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
            // Unconditional forward write -- any earlier pending abduction
            // for this fact is now moot (superseded, not confirmed).
            state.clear_provenance(eff.first);
            state.set_known_value(eff.first, eff.second);
        }

        // 2. Apply Numeric Effects
        for (const auto& num_eff : action.numeric_effects) {
            state.apply_numeric_effect(num_eff);
        }

        // 3. Apply Conditional Effects with Strict Knowledge Loss
        for (const auto& ce : action.conditional_effects) {
            uint8_t cond_result = Evaluator::evaluate_rpn_raw(ce.condition_rpn, state);

            if (cond_result == VAL_TRUE) {
                for (const auto& eff : ce.effects) {
                    state.clear_provenance(eff.first);
                    state.set_known_value(eff.first, eff.second);
                }
            }
            else if (cond_result == VAL_UNKNOWN) {
                // Scoped Regression Fix: if the condition is a single fact
                // (optionally negated), remember that this effect fact would
                // become eff.second if that one fact turns out true/false --
                // so a later direct observation of the effect fact can be
                // abduced backward into the condition. See
                // PartiallySpecifiedState::apply_provenance_deductions in
                // State.hpp for the deduction side and its scope/soundness
                // notes. Multi-fact conditions aren't tracked -- soundly
                // inverting an arbitrary RPN formula is out of scope here.
                bool trackable = false;
                int condition_fact_id = -1;
                bool condition_negated = false;
                if (ce.condition_rpn.size() == 1 && ce.condition_rpn[0] >= 0) {
                    trackable = true;
                    condition_fact_id = ce.condition_rpn[0];
                } else if (ce.condition_rpn.size() == 2 && ce.condition_rpn[0] >= 0 && ce.condition_rpn[1] == OP_NOT) {
                    trackable = true;
                    condition_fact_id = ce.condition_rpn[0];
                    condition_negated = true;
                }
                for (const auto& eff : ce.effects) {
                    state.set_unknown(eff.first);
                    if (trackable) {
                        state.record_conditional_provenance(eff.first, condition_fact_id, eff.second, condition_negated);
                    }
                }
            }
        }

        // 4. Apply Non-Deterministic Degradation
        for (int nd_fact : action.non_deterministic_effects) {
            state.set_unknown(nd_fact);
        }
    }
};