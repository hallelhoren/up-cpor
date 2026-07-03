#include "SDRPlanner.hpp"
#include "Evaluator.hpp"
#include "ActionApplier.hpp"
#include <iostream>

namespace CPOR {

int SDRPlanner::get_next_action() {
    // 0. State Machine Lock Check
    // Strictly prevent the system from dispatching new actions while waiting 
    // for a physical sensor response from the environment.
    if (expecting_observation) {
        std::cerr << "CRITICAL: Cannot get next action. System is locked waiting for an observation." << std::endl;
        return -2; // Engine locked error code
    }

    // Cache the current epistemic state as a const reference for fast, read-only evaluations
    const PartiallySpecifiedState& current_state = belief.get_current_state();

    // 1. Goal Check (Absolute Certainty Required)
    // The agent must mathematically guarantee all goal conditions are met (PESSIMISTIC mode).
    if (Evaluator::evaluate(global_problem.goal_rpn, current_state, EvalMode::PESSIMISTIC)) {
        return -1; // Standard return code for Goal Reached
    }

    // 2. Dead-End Verification (Safety Guard)
    // If ANY dead-end condition evaluates to True OR Unknown (OPTIMISTIC mode), 
    // the agent is mathematically compromised (or risks being compromised) and must halt execution.
    for (const auto& deadend_rpn : global_problem.deadend_rpns) {
        if (Evaluator::evaluate(deadend_rpn, current_state, EvalMode::OPTIMISTIC)) {
            std::cerr << "CRITICAL: Agent has entered a mathematically unrecoverable dead-end." << std::endl;
            return -2; // Execution failure
        }
    }

    // 3. Cache Validation (Precondition Verification)
    // Verify that the currently planned sequence is still mathematically sound 
    // given the current world knowledge (which may have collapsed after a recent observation).
    bool plan_invalid = false;
    if (next_action_index < plan_queue.size()) {
        int candidate_action_id = plan_queue[next_action_index];
        const GroundedAction& candidate_action = global_problem.actions[candidate_action_id];

        // If we are not absolutely certain (PESSIMISTIC) that preconditions hold, 
        // the cached plan is dangerous and must be discarded.
        if (!Evaluator::evaluate(candidate_action.precondition_rpn, current_state, EvalMode::PESSIMISTIC)) {
            plan_invalid = true;
        }
    }

    // 4. Replanning Trigger
    // Triggers if the plan was invalidated by world changes, or if we exhausted our cached queue.
    if (plan_invalid || next_action_index >= plan_queue.size()) {
        
        // Invoke the Z3 Determinization & BFS Pipeline. 
        // We pass a default sample_size of 5 for witness state generation.
        plan_queue = compute_linear_plan(belief, global_problem, 5); 
        next_action_index = 0; // Reset execution pointer to the start of the new plan

        // Handle total planner failure gracefully
        if (plan_queue.empty()) {
            std::cerr << "CRITICAL: Planner failed to find a valid robust path from the current state." << std::endl;
            return -2; // Planning failure
        }
    }

    // 5. Action Dispatch and State Machine Locking
    int action_to_execute = plan_queue[next_action_index];
    const GroundedAction& act_def = global_problem.actions[action_to_execute];

    if (act_def.observe_predicate_id != -1) {
        // Sensing Action Dispatch
        // Lock the engine. We cannot advance the belief state or the next_action_index 
        // until apply_observation() is called with the physical boolean result.
        expecting_observation = true;
        pending_sensing_action_id = action_to_execute;
    } else {
        // Standard Action Dispatch
        // The action is purely physical/manipulative. We compute the subsequent state.
        PartiallySpecifiedState next_state = current_state;
        ActionApplier::apply_action(act_def, next_state);

        // Implicit Auto-Observations ---
        // Scan the predefined list of auto-observable predicates.
        // If the agent naturally learns this information by being in the new state,
        // evaluate it and force an immediate epistemic collapse.
        for (int obs_id : global_problem.auto_observable_predicates) {
            uint8_t eval_result = Evaluator::evaluate_rpn_raw({obs_id}, next_state);
            
            // If the condition is mathematically determinable (True or False)
            if (eval_result != VAL_UNKNOWN) {
                next_state.set_known_value(obs_id, eval_result == VAL_TRUE);
                
                // Immediately trigger OneOf invariants to deduce further hidden variables
                bool valid_deduction = next_state.apply_oneof_deductions(global_problem.oneofs);
                
                if (!valid_deduction) {
                    std::cerr << "CRITICAL ERROR: Auto-observation led to a logical contradiction with OneOf invariants." << std::endl;
                    return -2; 
                }
            }
        }
        // -------------------------------------------
        
        // Push the mutated, fully-collapsed state into our chronological BeliefState history.
        
        // Push the mutated state into our chronological BeliefState history.
        // This maintains the historical graph for future regression safety checks.
        belief.apply_forward_action(action_to_execute, next_state);
        
        // Safely advance to the next action in the cache for the next turn.
        next_action_index++;
    }

    // Return the integer ID of the action to be translated back to Python/Physical interface
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

bool SDRPlanner::execute_full_simulation() {
    // Execution safeguard to prevent infinite loops in cyclic or unresolvable domains
    constexpr int MAX_STEPS = 1000;
    int step_count = 0;

    std::cout << "--- Starting SDR Full Execution Simulation ---" << std::endl;

    while (true) {
        // 1. Loop Safeguard Check
        if (step_count >= MAX_STEPS) {
            std::cerr << "CRITICAL TIMEOUT: Maximum execution steps (" << MAX_STEPS 
                      << ") exceeded. The agent is caught in an infinite loop or a degenerate domain." 
                      << std::endl;
            return false;
        }

        // 2. Deliberation and Execution Dispatch
        // This will either pull the next cached action, trigger a replan, or detect a terminal state.
        int action_id = get_next_action();

        // 3. Return Value Handling
        if (action_id == -1) {
            // Goal successfully reached
            std::cout << "SUCCESS: Goal state confirmed. Execution completed in " 
                      << step_count << " steps." << std::endl;
            return true;
        }

        if (action_id < -1) {
            // Fatal planning failure, dead-end, or engine lock
            std::cerr << "FATAL FAILURE: The simulation aborted at step " << step_count 
                      << " due to an unrecoverable planning error." << std::endl;
            return false;
        }

        std::cout << "[Step " << step_count << "] Dispatched Action ID: " << action_id << std::endl;

        // 4. Sensing Action Resolution
        // If the dispatched action was a sensor activation, the state machine is currently locked.
        if (expecting_observation) {
            const GroundedAction& sensing_action = global_problem.actions[action_id];
            
            // Construct a temporary RPN query for the observed predicate
            std::vector<int> observation_rpn = { sensing_action.observe_predicate_id };

            // Simulate the environment's physical response. 
            // Because a separate physical hidden ground-truth state is not passed to this simulation, 
            // we simulate the outcome by evaluating the predicate optimistically against our current belief.
            bool simulated_observation_result = Evaluator::evaluate(
                observation_rpn, 
                belief.get_current_state(), 
                EvalMode::OPTIMISTIC
            );

            std::cout << "  -> Sensing Action Detected. Simulating environmental response: " 
                      << (simulated_observation_result ? "TRUE" : "FALSE") << std::endl;

            // Ingest the physical observation back into the epistemic state
            bool observation_successful = apply_observation(simulated_observation_result);

            if (!observation_successful) {
                // If applying the observation causes a mathematical contradiction with the OneOf invariants
                std::cerr << "FATAL FAILURE: Epistemic collapse resulted in a logical contradiction at step " 
                          << step_count << "." << std::endl;
                return false;
            }
        }

        // Advance the execution cycle
        step_count++;
    }

    // Unreachable due to the infinite loop, but required by C++ standard for non-void functions
    return false; 
}

} // namespace CPOR