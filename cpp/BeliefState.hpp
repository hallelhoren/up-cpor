#pragma once
#include <vector>
#include <cstdint>
#include "State.hpp"
#include "ProblemData.hpp"

namespace CPOR {

/**
 * @class BeliefState
 * @brief Manages the chronological history of the agent's state and observations.
 * Provides deep historical verification utilizing logical regression.
 */
class BeliefState {
private:
    PartiallySpecifiedState initial_state;
    PartiallySpecifiedState current_state;
    std::vector<int> action_history;
    std::vector<PartiallySpecifiedState> state_history;

public:
    BeliefState() = default;

    explicit BeliefState(const PartiallySpecifiedState& init_state)
        : initial_state(init_state), current_state(init_state) {}

    const PartiallySpecifiedState& get_current_state() const { return current_state; }
    const PartiallySpecifiedState& get_initial_state() const { return initial_state; }
    const std::vector<int>& get_action_history() const { return action_history; }

    /**
     * @brief Transitions the current belief state forward and logs the history.
     */
    void apply_forward_action(int action_id, const PartiallySpecifiedState& next_state);

    /**
     * @brief Checks if an RPN condition is guaranteed to hold true given the 
     * execution history, by regressing the condition backward to the initial state bounds.
     */
    bool verify_condition_safely(const std::vector<int>& rpn_condition, const ProblemDef& global_problem) const;
};

} // namespace CPOR