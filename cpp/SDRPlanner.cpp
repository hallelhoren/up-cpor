#include "SDRPlanner.hpp"
#include "Evaluator.hpp"
#include "ActionApplier.hpp"
#include "SDRSampler.hpp"
#include "FFSolver.hpp"
#include "DeadEndManager.hpp"
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

} // namespace CPOR