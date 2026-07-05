#pragma once
#include <vector>
#include "BeliefState.hpp"
#include <unordered_set>
#include "ProblemData.hpp"

namespace CPOR {

class SDRPlanner {
private:
    // Core state and environment references
    BeliefState belief;
    const ProblemDef& global_problem;
    
    // The Execution Cache (Replaces C# FutureActions and NextActionIndex)
    std::vector<int> plan_queue;
    size_t next_action_index;
    
    // Safety Locks (Replaces C# ExpectingObservation)
    bool expecting_observation;
    int pending_sensing_action_id;

    // Internal routing for deliberation
    std::vector<int> compute_linear_plan(const BeliefState& current_belief, const ProblemDef& problem, int sample_size);

    // Tracks states actually visited during execution to prevent physical loops
    std::unordered_set<PartiallySpecifiedState, StateHasher> visited_physical_states;

    int find_loop_breaking_sensing_action(const PartiallySpecifiedState& current_state); 
    bool check_goal(const PartiallySpecifiedState& state) const;
    bool verify_contingent_deadends(const PartiallySpecifiedState& state) const;
    std::vector<int> get_plan_from_solver(const PartiallySpecifiedState& state) const;

    bool handle_contingent_deadends(const PartiallySpecifiedState& state);
    int plan_to_observe_deadend(const PartiallySpecifiedState& current_state, const std::vector<int>& target_fluents);

public:
    // Constructor
    SDRPlanner(const PartiallySpecifiedState& initial_state, const ProblemDef& problem);

    // Phase 1: Deliberation and Execution Dispatch
    int get_next_action();

    // Phase 2: Environment Ingestion and Epistemic Collapse
    bool apply_observation(bool observation_value);

    bool execute_full_simulation();

    //DEBUG
    void set_plan_queue(const std::vector<int>& forced_plan) {
        plan_queue = forced_plan;
        next_action_index = 0; 
    }

    // Allows the execution environment to update the agent's history
    BeliefState& get_mutable_belief() {
        return belief;
    }
};

} // namespace CPOR