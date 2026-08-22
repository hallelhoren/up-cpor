#include "SDRPlanner.hpp"
#include "Evaluator.hpp"
#include "ActionApplier.hpp"
#include "SDRSampler.hpp"
#include "FFSolver.hpp"
#include "DeadEndManager.hpp"
#include "BFSSolver.hpp"
#include "Regression.hpp"
#include "DebugStats.hpp"
#include <iostream>
#include <chrono>


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
    if (Evaluator::evaluate(global_problem.goal_rpn, state, EvalMode::PESSIMISTIC)) {
        return true;
    }
    // The flat snapshot alone couldn't prove the goal -- try the full-history
    // regression fallback before giving up (see
    // BeliefState::verify_condition_safely): a goal fact can be genuinely
    // deducible from actions and observations already in this belief's own
    // history even though nothing ever wrote it directly into `state`'s
    // bitset (this is `state` == belief.get_current_state() at every actual
    // call site, so consulting `belief` here is exactly the same state, just
    // with its full history available).
    return belief.verify_condition_safely(global_problem.goal_rpn, global_problem);
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

std::vector<int> SDRPlanner::get_plan_from_solver(const PartiallySpecifiedState& state) {
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
    // Witness CONTINUITY, not fresh resampling every call, is what actually
    // matters here (a second, later fix): re-sampling a brand new witness
    // from `state` on every single call independently picks EVERY unknown
    // fact fresh via Z3, including ones the domain's own conditional effects
    // would otherwise tie together (e.g. localize5's `checking` action:
    // once a witness has committed to a specific hidden position and
    // `checking` has been dispatched, the *correct* witness must have
    // `free-up`/`free-down`/etc. set consistently with that exact position
    // -- but the belief itself never learns the position, via provenance or
    // otherwise, in this domain, so nothing forces a freshly-resampled
    // witness to respect what an EARLIER witness already committed to).
    // Concretely reproduced: sampling fresh at every call let step 1's
    // witness be internally consistent (one position, `checking` not yet
    // applied) while step 2's freshly-resampled witness had `ok=true` and
    // that SAME position, but `free-*` all arbitrarily false -- a genuine
    // dead end no classical planner (FF or BFS) can escape, even though the
    // real world reachable from that exact history is solvable. Verified by
    // direct comparison against unified_planning's own SimulatedExecutionEnvironment-style
    // single-persistent-world execution, which never hits this because it
    // only ever has ONE concrete world, applied to directly, never resampled.
    //
    // Fixed by maintaining guide_witness_ as a single hidden-state guess,
    // sampled once and then progressed forward via ActionApplier in lockstep
    // with every action get_next_action() actually dispatches (see its
    // step 5) -- exactly mirroring how the real belief itself is
    // progressed, so the witness's downstream conditional-effect
    // consequences (like free-*) are always derived correctly from the SAME
    // committed position, never independently re-guessed. Only invalidated
    // (forcing a fresh sample here) when apply_observation() learns
    // something that actually contradicts it -- the one case where the
    // current guess has been proven wrong by ground truth.
    if (!has_guide_witness_) {
        // Assert everything this belief's own history already implies (see
        // BeliefState::derive_learned_constraints) so the fresh witness
        // can't re-guess a hidden fact (like a position only ever observed
        // through its side effects) inconsistently with what's already been
        // ruled out -- exactly the gap that let the classical sub-solver
        // keep proposing the same already-failed physical move in a
        // localize5-style domain.
        std::vector<std::vector<int>> learned = belief.derive_learned_constraints(global_problem);
        std::vector<PartiallySpecifiedState> witnesses = SDRSampler::sample_concrete_states(state, global_problem, 1, learned);
        if (witnesses.empty()) {
            return {};
        }
        guide_witness_ = witnesses[0];
        has_guide_witness_ = true;
    }

    std::vector<int> plan = FFSolver::search(guide_witness_, global_problem);
    if (plan.empty()) {
        plan = BFSSolver::solve(guide_witness_, global_problem);
    }
    if (!plan.empty()) {
        return plan;
    }

    // The current guide witness is a genuine dead end: classically
    // unsolvable from it, not just momentarily unlucky. SDR's own algorithm
    // (Sec 4, Algorithm 2) calls for trying a different hidden-state
    // hypothesis in exactly this situation. Z3 sampling is deterministic
    // given the same belief (re-querying would just return this same
    // witness again), so request several diverse witnesses and adopt the
    // first the classical solver can actually make progress from.
    has_guide_witness_ = false;
    std::vector<std::vector<int>> learned_alt = belief.derive_learned_constraints(global_problem);
    std::vector<PartiallySpecifiedState> alternatives = SDRSampler::sample_concrete_states(state, global_problem, 5, learned_alt);
    for (const auto& alt : alternatives) {
        std::vector<int> alt_plan = FFSolver::search(alt, global_problem);
        if (alt_plan.empty()) alt_plan = BFSSolver::solve(alt, global_problem);
        if (!alt_plan.empty()) {
            guide_witness_ = alt;
            has_guide_witness_ = true;
            return alt_plan;
        }
    }
    return {};
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
    auto visited_it = visited_physical_states.find(predicted_next_state);
    if (visited_it != visited_physical_states.end()) {
        // The flat bitset repeats, but that alone doesn't mean we're
        // genuinely stuck: derive_learned_constraints() can capture position
        // -style knowledge (e.g. a hidden room narrowed, but not resolved to
        // one value, by a localization domain's per-room conditions) the
        // bitset itself never records. If strictly more is derivable now
        // than when this bitset was last visited, real epistemic progress
        // has happened since then -- proceed with the originally-chosen
        // action instead of treating this as a dead loop, and refresh the
        // stored count so the NEXT revisit is judged against this newer
        // baseline, not the stale one.
        size_t learned_now = belief.derive_learned_constraints(global_problem).size();
        if (learned_now > visited_it->second) {
            visited_it->second = learned_now;
        } else {
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
    }

    // 5. Dispatch & State Tracking
    // Record departure state to prevent future cycle regressions, alongside
    // how much derive_learned_constraints() can currently derive (see the
    // cycle check above for why this baseline matters, not just the bitset).
    // StateHasher includes known_mask, so learning new facts clears the cycle footprint.
    visited_physical_states[current_state] = belief.derive_learned_constraints(global_problem).size();

    // Advance the execution pointer for the next tick
    next_action_index++;

    // Progress the guide witness forward too, exactly like the real belief
    // is about to be (below, and in apply_observation() for sensing
    // actions) -- see get_plan_from_solver's comment for why this
    // continuity (rather than resampling fresh every call) is what makes
    // the witness's downstream conditional-effect consequences stay
    // consistent with whatever hidden-state guess it already committed to.
    // Applies uniformly regardless of which branch above chose
    // action_to_execute (the normal plan, the sensing-fallback, or the
    // cycle-breaking override): guide_witness_ is a single fully-concrete
    // world, so every action has a well-defined effect on it either way,
    // and apply_observation() is what actually detects and reacts to this
    // guess turning out wrong for a sensing action's observed fluent.
    if (has_guide_witness_) {
        ActionApplier::apply_action(act, guide_witness_);
    }

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
    } else {
        // Classical (non-sensing) action: commit its effect to belief right
        // now, mirroring what apply_observation() already does for sensing
        // actions below. predicted_next_state was already computed for this
        // exact action_to_execute at step 4 above and is still valid here --
        // the physical-cycle branch only ever reassigns action_to_execute to
        // a SENSING action (via find_loop_breaking_sensing_action), which
        // would have taken the `if` branch above instead of this `else`.
        //
        // Without this, belief.get_current_state() never advances past
        // whatever it was after the last *sensing* action (or the initial
        // state, if none has happened yet): every classical action dispatched
        // here was only ever used for this call's own cycle-detection check
        // and precondition re-validation, then silently discarded. The next
        // get_next_action() call would then plan from that same stale belief
        // again -- typically re-proposing the very same action, which the
        // real environment (e.g. unified_planning's SimulatedExecutionEnvironment)
        // correctly rejects as inapplicable, since it already advanced past
        // that state when the caller executed the first proposal. This was
        // the exact reproduction of the "SDR + UP Simulated Environment"
        // mode's crash: the first action succeeded, and the second -- still
        // planned from the pre-first-action belief -- was rejected as not
        // applicable by the simulator's real, already-advanced ground truth.
        belief.apply_forward_action(action_to_execute, predicted_next_state);
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

    // 1b. Witness continuity: the guide witness (see get_plan_from_solver)
    // is one concrete guess at the hidden state, already progressed through
    // this same sensing action at dispatch time (get_next_action's step 5),
    // so it already has a fixed, definite value for the fluent just
    // observed. If ground truth just disagreed with that guess, the guess
    // is proven wrong -- discard it so the next get_plan_from_solver call
    // samples fresh, now correctly respecting this observation (which is
    // about to be committed to belief below) instead of silently continuing
    // to reason from a hidden-state hypothesis reality already contradicted.
    if (has_guide_witness_ &&
        guide_witness_.is_true(sensing_action.observe_predicate_id) != observation_value) {
        has_guide_witness_ = false;
    }

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
    std::vector<int>* out_blocking_fact_tokens,
    bool* inout_has_guide_witness,
    PartiallySpecifiedState* inout_guide_witness)
{
    if (out_blocking_action_id) *out_blocking_action_id = -1;
    if (out_blocking_fact_tokens) out_blocking_fact_tokens->clear();

    const PartiallySpecifiedState& real_belief = current_belief.get_current_state();

    // Everything current_belief's own history already implies (see
    // BeliefState::derive_learned_constraints), asserted alongside every
    // sampling call below so a fresh witness can't re-guess a hidden fact
    // (like a position only ever observed through its side effects)
    // inconsistently with what's already been ruled out. Computed once
    // since it doesn't change across this call's several sampling attempts.
    std::vector<std::vector<int>> learned_constraints = current_belief.derive_learned_constraints(problem);

    // 1. Determinization: reuse an inherited guide witness across calls
    // instead of independently re-sampling one every time (see this
    // method's header comment for the exact witness-discontinuity bug this
    // avoids). Falls back to a fresh Z3 sample when the caller isn't
    // threading continuity at all (both pointers null) or has none yet.
    bool inherited_witness = inout_has_guide_witness && *inout_has_guide_witness && inout_guide_witness;

    std::vector<PartiallySpecifiedState> sampled_states;
    PartiallySpecifiedState primary_guide_state;

    if (inherited_witness) {
        primary_guide_state = *inout_guide_witness;
    } else {
        sampled_states = SDRSampler::sample_concrete_states(real_belief, problem, sample_size, learned_constraints);
        if (sampled_states.empty()) {
            if (inout_has_guide_witness) *inout_has_guide_witness = false;
            return {}; // Logically invalid belief state or unrecoverable contradiction
        }
        // 2. Select the Primary Guide State (s')
        primary_guide_state = sampled_states[0];
    }

    // 3. Route to the Classical Forward Search (Native FF C-API Wrapper)
    auto __dbg_cfs_t0 = DebugStats::enabled() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    std::vector<int> candidate_plan = FFSolver::search(primary_guide_state, problem);
    if (DebugStats::enabled()) {
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - __dbg_cfs_t0).count();
        DebugStats::record_ff_call(ms);
    }

    if (candidate_plan.empty() && inherited_witness) {
        // The inherited witness is a genuine classical dead end -- not an
        // artifact of this call's own sampling, since it wasn't sampled
        // here at all. Mirrors get_plan_from_solver's identical recovery:
        // discard it and try a fresh diverse batch, adopting the first one
        // FF can make progress from.
        if (inout_has_guide_witness) *inout_has_guide_witness = false;
        sampled_states = SDRSampler::sample_concrete_states(real_belief, problem, sample_size, learned_constraints);
        for (const auto& alt : sampled_states) {
            candidate_plan = FFSolver::search(alt, problem);
            if (!candidate_plan.empty()) {
                primary_guide_state = alt;
                break;
            }
        }
    }

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

    // Make sure there's a peer set for 4A/4C/4E's cross-witness validation
    // below even when the primary came from an inherited witness (which
    // skips the ordinary sampling step above) -- and that the primary
    // itself is one of the peers, exactly as when it's sampled_states[0] in
    // the non-continuity path.
    if (sampled_states.empty()) {
        sampled_states = SDRSampler::sample_concrete_states(real_belief, problem, sample_size, learned_constraints);
    }
    if (sampled_states.empty()) {
        sampled_states.push_back(primary_guide_state);
    } else if (inherited_witness) {
        sampled_states.push_back(primary_guide_state);
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

    if (inout_has_guide_witness && inout_guide_witness) {
        // primary_guide_state has already been progressed through every
        // action added to robust_plan via 4B's ActionApplier::apply_action
        // call above -- hand it onward exactly as-is so the caller's next
        // node/call in this same history starts from the same hidden-state
        // hypothesis instead of re-guessing it.
        *inout_guide_witness = primary_guide_state;
        *inout_has_guide_witness = true;
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
            // Regression through the full history may prove this dead-end
            // condition is definitely FALSE even though the flat snapshot
            // alone couldn't rule it out -- see
            // BeliefState::verify_condition_safely and
            // RegressionEngine::regress_rpn's multi-conditional-effect
            // disjunction handling. `state` is belief.get_current_state() at
            // this method's one call site, so this is the same state, just
            // with its history available.
            if (belief.verify_condition_safely(RegressionEngine::negate_rpn(rpn), global_problem)) {
                continue; // Proven false via full-history regression -- not a MaybeDeadEnd after all.
            }

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