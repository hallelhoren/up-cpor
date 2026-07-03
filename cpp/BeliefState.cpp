#include "BeliefState.hpp"
#include "Regression.hpp"
#include "Evaluator.hpp"

namespace CPOR {

void BeliefState::apply_forward_action(int action_id, const PartiallySpecifiedState& next_state) {
    action_history.push_back(action_id);
    current_state = next_state;
}

bool BeliefState::verify_condition_safely(const std::vector<int>& rpn_condition, const ProblemDef& global_problem) const {
    if (rpn_condition.empty()) return true;

    std::vector<int> current_rpn = rpn_condition;

    // Regress chronologically backward through the action history
    for (auto it = action_history.rbegin(); it != action_history.rend(); ++it) {
        int action_id = *it;
        const GroundedAction& action = global_problem.actions[action_id];
        current_rpn = RegressionEngine::regress_rpn(current_rpn, action);
    }

    // Evaluate the fully regressed condition against the concrete initial state
    // We use PESSIMISTIC mode: it must be absolutely TRUE, not UNKNOWN.
    return Evaluator::evaluate(current_rpn, initial_state, EvalMode::PESSIMISTIC);
}

} // namespace CPOR