#pragma once

#include "State.hpp"
#include "ProblemData.hpp"
#include "Evaluator.hpp"

/**
 * @class ActionApplier
 * @brief Stateless system that safely transitions a state using CPOR Knowledge Loss rules.
 */
class ActionApplier {
public:
    static void apply_action(const GroundedAction& action, PartiallySpecifiedState& state) {
        // 1. Apply Guaranteed Effects
        for (const auto& eff : action.guaranteed_effects) {
            uint64_t bit = 1ULL << (eff.first % 64);
            state.known_mask[eff.first / 64] |= bit; 
            if (eff.second) state.value_mask[eff.first / 64] |= bit; 
            else state.value_mask[eff.first / 64] &= ~bit; 
        }

        // 2. Apply Numeric Effects (Phase 3 Integration)
        for (const auto& num_eff : action.numeric_effects) {
            state.apply_numeric_effect(num_eff);
        }

        // 3. Apply Conditional Effects with Strict Knowledge Loss
        for (const auto& ce : action.conditional_effects) {
            uint8_t cond_result = Evaluator::evaluate_rpn_raw(ce.condition_rpn, state);
            
            if (cond_result == VAL_TRUE) {
                for (const auto& eff : ce.effects) {
                    uint64_t bit = 1ULL << (eff.first % 64);
                    state.known_mask[eff.first / 64] |= bit;
                    if (eff.second) state.value_mask[eff.first / 64] |= bit;
                    else state.value_mask[eff.first / 64] &= ~bit;
                }
            } 
            else if (cond_result == VAL_UNKNOWN) {
                for (const auto& eff : ce.effects) {
                    state.set_unknown(eff.first); 
                }
            }
        }

        // 4. Apply Non-Deterministic Degradation
        for (int nd_fact : action.non_deterministic_effects) {
            state.set_unknown(nd_fact);
        }
    }
};