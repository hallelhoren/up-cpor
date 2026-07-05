#include "BeliefState.hpp"
#include "Regression.hpp"
#include "Evaluator.hpp"

namespace CPOR {

void BeliefState::apply_forward_action(int action_id, const PartiallySpecifiedState& next_state) {
    // 1. Snapshot the belief state AT THE EXACT TIME the action was chosen.
    state_history.push_back(current_state); 
    
    // 2. Advance the timeline
    action_history.push_back(action_id);
    current_state = next_state;
}

bool BeliefState::verify_condition_safely(const std::vector<int>& rpn_condition, const ProblemDef& global_problem) const {
    if (rpn_condition.empty()) return true;

    std::vector<int> current_rpn = rpn_condition;

    // Regress chronologically backward through the action history using indices
    for (int i = static_cast<int>(action_history.size()) - 1; i >= 0; --i) {
        int action_id = action_history[i];
        const GroundedAction& action = global_problem.actions[action_id];
        
        // Retrieve the exact epistemic state the agent held right before this action
        const PartiallySpecifiedState& historical_belief = state_history[i];

        // Pass the historical belief to safely evaluate conditional effects
        current_rpn = RegressionEngine::regress_rpn(current_rpn, action, historical_belief);
    }

    // Evaluate the fully regressed condition against the concrete initial state
    // We use PESSIMISTIC mode: it must be absolutely TRUE, not UNKNOWN.
    return Evaluator::evaluate(current_rpn, initial_state, EvalMode::PESSIMISTIC);
}

} // namespace CPOR