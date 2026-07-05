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
    // It delegates the heavy lifting to the BFSSolver which treats the 
    // PartiallySpecifiedState as a determinized root node.
    return BFSSolver::solve(state, global_problem);
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

    // 3. Plan Retrieval / Generation
    int action_to_execute = -1;
    
    // If we have exhausted our current plan trajectory, we must replan
    if (next_action_index >= plan_queue.size()) {
        plan_queue = get_plan_from_solver(current_state);
        
        if (plan_queue.empty()) {
            std::cerr << "CRITICAL: Classical solver failed to find a path from the current belief state." << std::endl;
            return -2; // Execution failure
        }
        
        // Reset execution pointer for the new plan
        next_action_index = 0; 
    }

    action_to_execute = plan_queue[next_action_index];

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

    // 3. Epistemic Collapse (OneOf Deductions)
    // Now that a new concrete fact is known, propagate this knowledge across all 
    // mutually exclusive (OneOf) groups to deduce hidden variables and collapse the belief space.
    bool is_logically_valid = updated_state.apply_oneof_deductions(global_problem.oneofs);

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
    int sample_size) 
{
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