#pragma once
#include <vector>
#include "BeliefState.hpp"
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

public:
    // Constructor
    SDRPlanner(const PartiallySpecifiedState& initial_state, const ProblemDef& problem);

    // Phase 1: Deliberation and Execution Dispatch
    int get_next_action();

    // Phase 2: Environment Ingestion and Epistemic Collapse
    bool apply_observation(bool observation_value);

    int get_next_action();
};

} // namespace CPOR