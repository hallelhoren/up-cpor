#pragma once

#include "State.hpp"
#include "ProblemData.hpp"
#include "Evaluator.hpp"

extern ProblemDef global_problem;

enum class DeadEndStatus {
    SAFE,       // Dead end is definitively FALSE
    FATAL,      // Dead end is definitively TRUE (Trigger Backpropagation!)
    MAYBE       // Dead end is UNKNOWN (Trigger Observation/Branching!)
};

/**
 * @class DeadEndManager
 * @brief Evaluates global invariants to prune the search space early.
 */
class DeadEndManager {
public:
    static DeadEndStatus check_dead_ends(const PartiallySpecifiedState& state) {
        // First pass: Are we definitely in a fatal state?
        for (const auto& rpn : global_problem.deadend_rpns) {
            uint8_t result = Evaluator::evaluate_rpn_raw(rpn, state);
            if (result == VAL_TRUE) return DeadEndStatus::FATAL;
        }
        
        // Second pass: If none are fatal, are we in danger of a phantom dead end?
        for (const auto& rpn : global_problem.deadend_rpns) {
            uint8_t result = Evaluator::evaluate_rpn_raw(rpn, state);
            if (result == VAL_UNKNOWN) return DeadEndStatus::MAYBE;
        }

        return DeadEndStatus::SAFE;
    }
};