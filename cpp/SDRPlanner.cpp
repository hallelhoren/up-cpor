#include "SDRPlanner.hpp"
#include "Evaluator.hpp"
#include "ActionApplier.hpp"
#include "SDRSampler.hpp"
#include "FFSolver.hpp"
#include "DeadEndManager.hpp"
#include "BFSSolver.hpp"
#include <iostream>


namespace CPOR {
    
SDRPlanner::SDRPlanner(const PartiallySpecifiedState& initial_state, const ProblemDef& problem)
    : belief(initial_state), 
      global_problem(problem), 
      next_action_index(0), 
      expecting_observation(false), 
      pending_sensing_action_id(-1) 
{
    // Initialization complete
}

// Helper Method: Hunts for an applicable sensing action to force an epistemic collapse
int SDRPlanner::find_loop_breaking_sensing_action(const PartiallySpecifiedState& current_state) {
    for (const auto& action : global_problem.actions) {
        // 1. Is it a sensing action?
        if (action.observe_predicate_id != -1) {
            // 2. Does it sense a variable we currently DO NOT know?
            if (current_state.is_unknown(action.observe_predicate_id)) {
                // 3. Are we physically capable of executing it right now?
                if (Evaluator::evaluate(action.precondition_rpn, current_state, EvalMode::PESSIMISTIC)) {
                    return action.id;
                }
            }
        }
    }
    return -1; // No loop-breaking observation is immediately available
}

bool SDRPlanner::check_goal(const PartiallySpecifiedState& state) const {
    // The goal must be mathematically guaranteed. 
    // PESSIMISTIC mode ensures we don't accidentally assume a goal is met 
    // due to unresolved VAL_UNKNOWN variables.
    return Evaluator::evaluate(global_problem.goal_rpn, state, EvalMode::PESSIMISTIC);
}

bool SDRPlanner::verify_contingent_deadends(const PartiallySpecifiedState& state) const {
    for (const auto& rpn : global_problem.deadend_rpns) {
        
        // Check 1: Confirmed DeadEnd (VAL_TRUE)
        // If it evaluates to true under PESSIMISTIC conditions, it is mathematically unavoidable.
        if (Evaluator::evaluate(rpn, state, EvalMode::PESSIMISTIC)) {
            return true; 
        }

        // Check 2: MaybeDeadEnd (VAL_UNKNOWN)
        // If it evaluates to true under OPTIMISTIC conditions (but failed PESSIMISTIC),
        // it means the dead-end is possible, but currently obscured by missing knowledge.
        if (Evaluator::evaluate(rpn, state, EvalMode::OPTIMISTIC)) {
            // We treat this as a MaybeDeadEnd. According to SDR algorithm rules, 
            // we do NOT abort the branch. The main execution loop in get_next_action() 
            // is responsible for catching this epistemic gap and injecting a sensing action.
            continue; 
        }
    }
    
    // The agent is safe from all known dead-ends
    return false;
}

std::vector<int> SDRPlanner::get_plan_from_solver(const PartiallySpecifiedState& state) const {
    // This acts as the deliberation phase, querying the classical planner.
    // Both FFSolver::search and BFSSolver::solve require a FULLY
    // DETERMINIZED state -- `state` (belief.get_current_state()) generally
    // isn't one: under this engine's open-world initialization, any fact
    // not yet resolved by an action/observation is genuinely UNKNOWN, not
    // arbitrarily true or false, so passing it directly leaves every
    // precondition referencing an unresolved fact evaluating false and the
    // classical search degenerately inapplicable from the very first step
    // (this was a real, latent bug: it made this instance-based
    // get_next_action() path fail outright on any domain with real initial
    // uncertainty, e.g. doors5, until this fix -- previously masked because
    // nothing exercised this path before it was wired up to
    // up_cpor.engine.SDRImpl).
    //
    // SDRPlanner::compute_linear_plan (the CPOR outer loop's stateless
    // OnlinePlan subroutine) avoids this by sampling a concrete witness via
    // SDRSampler before ever calling FFSolver::search; do the same single-
    // witness determinization here, matching the SDR paper's own Algorithm
    // 1/2 "sample a distinguished state, plan for it, execute until an
    // observation contradicts it" model. Note this is deliberately the
    // simpler, single-witness SDR model -- it does not carry
    // compute_linear_plan's own additional multi-witness safety validation
    // (4A/4B/4C in SDRPlanner::compute_linear_plan), since that machinery
    // is specific to CPOR's stack-based tree-building loop, not to SDR's
    // own one-action-at-a-time online replanning contract.
    std::vector<PartiallySpecifiedState> witnesses = SDRSampler::sample_concrete_states(state, global_problem, 1);
    if (witnesses.empty()) {
        return {};
    }
    const PartiallySpecifiedState& witness = witnesses[0];

    std::vector<int> plan = FFSolver::search(witness, global_problem);
    if (!plan.empty()) {
        return plan;
    }
    return BFSSolver::solve(witness, global_problem);
}

int SDRPlanner::get_next_action() {
    // 0. State Machine Lock Check
    // Prevent action dispatch while waiting for physical sensor resolution
    if (expecting_observation) {
        std::cerr << "CRITICAL: Cannot dispatch action. System is locked waiting for an observation." << std::endl;
        return -2; // Execution failure
    }

    // Cache the current epistemic state for fast, read-only evaluations
    const PartiallySpecifiedState& current_state = belief.get_current_state();

    // 1. Goal Check 
    // Assumes check_goal() evaluates the global_problem.goal_rpn in PESSIMISTIC mode.
    if (check_goal(current_state)) {
        std::cout << "[SDR] Goal reached successfully." << std::endl;
        return -1; // Standard return code for Goal Achieved
    }

    // 2. Contingent Dead-End Verification
    // This now actively injects sensing sub-goals into plan_queue if it hits a MaybeDeadEnd
    if (handle_contingent_deadends(current_state)) {
        return -2; // Confirmed failure
    }

    // 3. Plan Retrieval / Generation, with real-belief precondition
    // re-validation before ever proposing an action to the caller -- SDR
    // Algorithm 2's own safety check (regress the negation of the next
    // action's precondition through history; if inconsistent with the real
    // belief, abandon the plan and replan -- see SDRPlanner::compute_linear_plan's
    // step 4A for the same idea applied to CPOR's tree-building loop). The
    // plan in plan_queue was derived from a single SAMPLED witness (see
    // get_plan_from_solver), so its next step might depend on a fact that
    // witness merely guessed rather than one this belief actually knows.
    // Previously unchecked here: the caller could be handed an action that
    // was never actually applicable in the real state, only in the witness
    // that happened to produce it.
    int action_to_execute = -1;

    constexpr int MAX_REPLAN_ATTEMPTS = 5;
    for (int attempt = 0; attempt < MAX_REPLAN_ATTEMPTS; ++attempt) {
        if (next_action_index >= plan_queue.size()) {
            plan_queue = get_plan_from_solver(current_state);

            if (plan_queue.empty()) {
                std::cerr << "CRITICAL: Classical solver failed to find a path from the current belief state." << std::endl;
                return -2; // Execution failure
            }

            next_action_index = 0;
        }

        int candidate = plan_queue[next_action_index];
        if (Evaluator::evaluate(global_problem.actions[candidate].precondition_rpn, current_state, EvalMode::PESSIMISTIC)) {
            action_to_execute = candidate;
            break;
        }

        // This witness-derived plan's next step isn't actually known-applicable
        // against the real belief -- discard it and force a fresh sample +
        // classical-solve attempt on the next loop iteration.
        plan_queue.clear();
        next_action_index = 0;
    }

    if (action_to_execute == -1) {
        // Repeated resampling didn't help -- likely a structural gap rather
        // than an unlucky witness: FF has no incentive to ever choose a
        // sensing action when handed a single fully-determined witness
        // (every fact already has *some* value by construction), so if the
        // domain requires sensing before it's possible to know a
        // precondition at all (e.g. colorballs2-2: you must sense a ball's
        // color before any trash-it action's precondition is knowable),
        // every fresh witness reproduces the identical class of failure.
        // find_loop_breaking_sensing_action is already used for exactly
        // this "actively force an epistemic collapse" purpose during cycle
        // breaking below; reuse it as a general last resort here too.
        int sensing_action = find_loop_breaking_sensing_action(current_state);
        if (sensing_action != -1) {
            plan_queue = { sensing_action };
            next_action_index = 0;
            action_to_execute = sensing_action;
        }
    }

    if (action_to_execute == -1) {
        std::cerr << "CRITICAL: Repeated replanning (" << MAX_REPLAN_ATTEMPTS
                  << " attempts) could not find an action whose precondition is "
                  << "known-true against the real belief." << std::endl;
        return -2;
    }

    // 4. Physical Cycle Detection & SDR Loop Breaking
    // Predict the immediate physical result of this action using our stateless applier.
    PartiallySpecifiedState predicted_next_state = current_state;
    const GroundedAction& act = global_problem.actions[action_to_execute];
    ActionApplier::apply_action(act, predicted_next_state);

    // Verify if the agent is about to re-enter an identical physical/epistemic state
    if (visited_physical_states.find(predicted_next_state) != visited_physical_states.end()) {
        std::cout << "[SDR] WARNING: Physical cycle detected on action " << action_to_execute << "." << std::endl;
        
        int alternative_action = find_loop_breaking_sensing_action(current_state);
        
        if (alternative_action != -1) {
            std::cout << "[SDR] Forcing branching decision: Injecting sensing action " << alternative_action << std::endl;
            // Override the classical plan to force epistemic collapse
            plan_queue = { alternative_action };
            next_action_index = 0;
            action_to_execute = alternative_action;
        } else {
            std::cerr << "CRITICAL: Agent is trapped in a physical loop with no available sensing actions. Aborting." << std::endl;
            return -2; 
        }
    }

    // 5. Dispatch & State Tracking
    // Record departure state to prevent future cycle regressions. 
    // StateHasher includes known_mask, so learning new facts clears the cycle footprint.
    visited_physical_states.insert(current_state);
    
    // Advance the execution pointer for the next tick
    next_action_index++;

    // Lock the planner if this action requires environmental feedback
    if (global_problem.actions[action_to_execute].observe_predicate_id != -1) {
        expecting_observation = true;
        // apply_observation() requires this to be set (it looks up which
        // fluent was being sensed via global_problem.actions[pending_sensing_action_id]) --
        // previously left at its constructor default of -1 forever, so
        // apply_observation() always hit its own "not currently expecting
        // one" guard and failed on every call, regardless of expecting_observation
        // above being correctly true. Latent since nothing exercised this
        // instance-based get_next_action()/apply_observation() step-by-step
        // path until now (compute_linear_plan's stateless usage never
        // touches it).
        pending_sensing_action_id = action_to_execute;
    }

    return action_to_execute;
}

bool SDRPlanner::apply_observation(bool observation_value) {
    // 1. State Validation
    // Strictly prevent the ingestion of rogue or asynchronous observations 
    // when the engine has not explicitly dispatched a sensing action.
    if (!expecting_observation || pending_sensing_action_id == -1) {
        std::cerr << "CRITICAL ERROR: Received an observation, but the planner is not currently expecting one. "
                  << "The state machine is out of sync with the physical environment." << std::endl;
        return false;
    }

    const GroundedAction& sensing_action = global_problem.actions[pending_sensing_action_id];

    // Secondary architectural guard: ensure the pending action is actually capable of sensing
    if (sensing_action.observe_predicate_id == -1) {
        std::cerr << "CRITICAL ERROR: The pending action (ID: " << pending_sensing_action_id 
                  << ") is not mathematically defined as a sensing action." << std::endl;
        return false;
    }

    // 2. State Ingestion
    // Retrieve a mutable copy of the current epistemic state to apply the new physical truth.
    PartiallySpecifiedState updated_state = belief.get_current_state();
    
    // Inject the physical boolean response directly into the Dual-Mask bitset.
    // This transitions the specific fluent from VAL_UNKNOWN to VAL_TRUE/VAL_FALSE.
    updated_state.set_known_value(sensing_action.observe_predicate_id, observation_value);

    // 2b. Scoped Regression Fix: this is a genuine observation, so abduce
    // any conditional-effect conditions this now-known fact can resolve
    // backward (see State.hpp's apply_provenance_deductions) before the
    // OneOf closure below -- and once more after, in case OneOf deductions
    // themselves resolved a fact another pending provenance was waiting on.
    updated_state.apply_provenance_deductions();

    // 3. Epistemic Collapse (OneOf Deductions)
    // Now that a new concrete fact is known, propagate this knowledge across all
    // mutually exclusive (OneOf) groups to deduce hidden variables and collapse the belief space.
    bool is_logically_valid = updated_state.apply_oneof_deductions(global_problem.oneofs);

    updated_state.apply_provenance_deductions();

    if (!is_logically_valid) {
        std::cerr << "FATAL ERROR: The physical observation (" << observation_value 
                  << " for predicate ID " << sensing_action.observe_predicate_id 
                  << ") caused a mathematical contradiction with the global OneOf invariants. "
                  << "The problem definition or environment physics may be corrupted." << std::endl;
        return false;
    }

    // 4. State History Synchronization
    // Formally commit the newly collapsed state and the sensing action that caused it 
    // into the chronological history graph. This ensures regression checks remain accurate.
    belief.apply_forward_action(pending_sensing_action_id, updated_state);

    // 5. State Machine Release
    // Fully unlock the orchestration layer so `get_next_action()` can resume Deliberation/Reaction.
    expecting_observation = false;
    pending_sensing_action_id = -1;
    
    // Advance the execution pointer so the agent executes the next step of the cached plan
    // or triggers a replan if the newly deduced knowledge invalidates future preconditions.
    next_action_index++;

    return true;
}


std::vector<int> SDRPlanner::compute_linear_plan(
    const BeliefState& current_belief,
    const ProblemDef& problem,
    int sample_size,
    int* out_blocking_action_id,
    std::vector<int>* out_blocking_fact_tokens)
{
    if (out_blocking_action_id) *out_blocking_action_id = -1;
    if (out_blocking_fact_tokens) out_blocking_fact_tokens->clear();

    // 1. Determinization: Sample diverse concrete "witness" states
    std::vector<PartiallySpecifiedState> sampled_states =
        SDRSampler::sample_concrete_states(current_belief.get_current_state(), problem, sample_size);

    if (sampled_states.empty()) {
        return {}; // Logically invalid belief state or unrecoverable contradiction
    }

    // 2. Select the Primary Guide State (s')
    PartiallySpecifiedState primary_guide_state = sampled_states[0];

    // 3. Route to the Classical Forward Search (Native FF C-API Wrapper)
    std::vector<int> candidate_plan = FFSolver::search(primary_guide_state, problem);

    if (candidate_plan.empty()) {
        // FF found nothing at all for the primary guide witness -- there's no
        // specific action to blame (out_blocking_action_id stays -1), but
        // the caller can still often make progress: surface which of the
        // REAL belief's still-unresolved facts is most likely responsible,
        // so it can try a sensing branch instead of falling straight back to
        // the exhaustive search. See this method's header comment for why
        // goal tokens are tried first and a full unresolved-fact sweep only
        // as the fallback.
        if (out_blocking_fact_tokens) {
            const PartiallySpecifiedState& real_belief = current_belief.get_current_state();
            for (int token : problem.goal_rpn) {
                if (token >= 0 && real_belief.is_unknown(token)) {
                    out_blocking_fact_tokens->push_back(token);
                }
            }
            if (out_blocking_fact_tokens->empty()) {
                for (int pred_id = 0; pred_id < problem.total_predicates; ++pred_id) {
                    if (real_belief.is_unknown(pred_id)) {
                        out_blocking_fact_tokens->push_back(pred_id);
                    }
                }
            }
        }
        return {}; // No classical path found
    }

    // 4. Witness-State Plan Validation (Multi-State Soundness)
    std::vector<int> robust_plan;
    robust_plan.reserve(candidate_plan.size());

    for (int action_id : candidate_plan) {
        const GroundedAction& action = problem.actions[action_id];
        
        bool is_unsafe = false;
        bool causes_divergence = false;

        // 4A. Validate Preconditions (Pre-Action)
        for (const PartiallySpecifiedState& witness : sampled_states) {
            if (Evaluator::evaluate_rpn_raw(action.precondition_rpn, witness) != VAL_TRUE) {
                is_unsafe = true;
                break;
            }
        }

        if (is_unsafe) {
            if (out_blocking_action_id) *out_blocking_action_id = action_id;
            break; // Truncate BEFORE adding the unsafe action
        }

        // 4B. Safely apply the action to all realities
        for (PartiallySpecifiedState& witness : sampled_states) {
            ActionApplier::apply_action(action, witness);
        }
        ActionApplier::apply_action(action, primary_guide_state);

        // 4C. Check for Fatal Dead-Ends (Post-Action)
        for (const PartiallySpecifiedState& witness : sampled_states) {
            if (DeadEndManager::check_dead_ends(witness) == DeadEndStatus::FATAL) {
                is_unsafe = true;
                break;
            }
        }

        if (is_unsafe) {
            if (out_blocking_action_id) *out_blocking_action_id = action_id;
            break; // Action was valid to start, but led to a dead-end. Truncate BEFORE adding.
        }

        // 4D. The action is definitively safe. Add it to our robust plan.
        robust_plan.push_back(action_id);

        // 4E. Check Observational Parity (Post-Action Sensing Divergence)
        if (action.observe_predicate_id != -1) {
            uint8_t primary_observation = Evaluator::evaluate_rpn_raw({action.observe_predicate_id}, primary_guide_state);
            
            for (const PartiallySpecifiedState& witness : sampled_states) {
                uint8_t witness_observation = Evaluator::evaluate_rpn_raw({action.observe_predicate_id}, witness);
                if (witness_observation != primary_observation) {
                    causes_divergence = true;
                    break;
                }
            }
        }

        if (causes_divergence) {
            // We KEEP the sensing action (already pushed to robust_plan) because it is safe.
            // But we break here so CPOR can gracefully split the tree based on the divergent truths.
            break;
        }
    }

    return robust_plan;
}

int SDRPlanner::plan_to_observe_deadend(const PartiallySpecifiedState& current_state, const std::vector<int>& target_fluents) {
    // Basic Anti-Looping Mechanism for sensing attempts
    static std::unordered_map<int, int> sensing_attempt_tracker;
    const int MAX_SENSING_RETRIES = 3;

    for (int target_fluent : target_fluents) {
        if (sensing_attempt_tracker[target_fluent] >= MAX_SENSING_RETRIES) {
            std::cout << "[SDR] WARNING: Reached sensing limit for fluent " << target_fluent << ". Skipping." << std::endl;
            continue;
        }

        // 1. Immediate Applicability Check
        for (const auto& action : global_problem.actions) {
            if (action.observe_predicate_id == target_fluent) {
                if (Evaluator::evaluate(action.precondition_rpn, current_state, EvalMode::PESSIMISTIC)) {
                    sensing_attempt_tracker[target_fluent]++;
                    return action.id; // Return the immediate sensing action
                }
            }
        }
    }
    return -1; // No valid sensing action could be found
}

bool SDRPlanner::handle_contingent_deadends(const PartiallySpecifiedState& state) {
    bool maybe_deadend = false;
    std::vector<int> unknown_deadend_fluents;

    for (const auto& rpn : global_problem.deadend_rpns) {
        // Check 1: Confirmed DeadEnd (VAL_TRUE)
        if (Evaluator::evaluate(rpn, state, EvalMode::PESSIMISTIC)) {
            std::cerr << "CRITICAL: Agent is in a confirmed DeadEndTrue. Aborting." << std::endl;
            return true; // Unrecoverable failure
        }

        // Check 2: MaybeDeadEnd (VAL_UNKNOWN)
        if (Evaluator::evaluate(rpn, state, EvalMode::OPTIMISTIC)) {
            maybe_deadend = true;
            
            // Extract the specific fluents causing the uncertainty
            for (int token : rpn) {
                if (token >= 0 && state.is_unknown(token)) {
                    if (std::find(unknown_deadend_fluents.begin(), unknown_deadend_fluents.end(), token) == unknown_deadend_fluents.end()) {
                        unknown_deadend_fluents.push_back(token);
                    }
                }
            }
        }
    }

    if (maybe_deadend) {
        std::cout << "[SDR] MaybeDeadEnd detected. Suspending primary plan to resolve uncertainty." << std::endl;
        int sensing_action_id = plan_to_observe_deadend(state, unknown_deadend_fluents);
        
        if (sensing_action_id != -1) {
            // INJECTION: Override the cached plan and force immediate execution of the sensing path
            plan_queue = { sensing_action_id };
            next_action_index = 0;
            return false; // Not a failure, we successfully recovered
        } else {
            std::cerr << "CRITICAL: Agent is in a MaybeDeadEnd but cannot find a valid sensing action to resolve it. Aborting." << std::endl;
            return true; // Treated as failure if we can't observe it
        }
    }
    
    return false; // Safe
}


} // namespace CPOR