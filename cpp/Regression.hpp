#pragma once
#include <vector>
#include "ProblemData.hpp"
#include "Evaluator.hpp"
#include "State.hpp" // Required for PartiallySpecifiedState

namespace CPOR {

class RegressionEngine {
public:
    /**
     * @brief Regresses an RPN logic condition backwards through a specific action.
     * * @param current_rpn The flat RPN expression being pushed backward in time.
     * @param action The GroundedAction that was applied at this step in history.
     * @param current_belief The epistemic state of the world AT THE TIME the action was executed, 
     * used to validate if conditional effects definitively triggered.
     * @return std::vector<int> The transformed RPN representing the regressed condition.
     */
    static std::vector<int> regress_rpn(
        const std::vector<int>& current_rpn, 
        const GroundedAction& action, 
        const PartiallySpecifiedState& current_belief) 
    {
        std::vector<int> regressed_rpn;
        // Preallocate to prevent heap reallocations during backward chaining
        regressed_rpn.reserve(current_rpn.size());

        for (int token : current_rpn) {
            // If token >= 0, it's a physical fluent ID. Otherwise, it's a logical operator.
            if (token >= 0) {
                bool modified_by_action = false;
                
                // 1. Check Guaranteed Effects (Highest Priority)
                for (const auto& eff : action.guaranteed_effects) {
                    if (eff.first == token) {
                        modified_by_action = true;
                        // Substitute the variable with its absolute post-action truth value
                        regressed_rpn.push_back(eff.second ? OP_TRUE : OP_FALSE);
                        break;
                    }
                }
                
                // 2. Check Conditional Effects (The SDR Fix)
                if (!modified_by_action) {
                    for (const auto& ce : action.conditional_effects) {
                        
                        // Look-ahead: Does this conditional branch even touch our token?
                        bool affects_token = false;
                        bool effect_value = false;
                        for (const auto& eff : ce.effects) {
                            if (eff.first == token) {
                                affects_token = true;
                                effect_value = eff.second;
                                break;
                            }
                        }
                        
                        if (affects_token) {
                            // The token is affected. Evaluate the condition against the historical belief state.
                            uint8_t cond_eval = Evaluator::evaluate_rpn_raw(ce.condition_rpn, current_belief);
                            
                            if (cond_eval == VAL_TRUE) {
                                // The condition mathematically held. We deduce the effect was applied.
                                modified_by_action = true;
                                regressed_rpn.push_back(effect_value ? OP_TRUE : OP_FALSE);
                                break;
                            } 
                            else if (cond_eval == VAL_UNKNOWN) {
                                // The condition was unknown. This means the regression through this 
                                // specific action is non-deterministic (knowledge degraded).
                                modified_by_action = true;
                                regressed_rpn.push_back(token); // Pass as unresolved fluent
                                break;
                            }
                            // If VAL_FALSE, the effect didn't happen, continue loop to check other CE branches.
                        }
                    }
                }

                // 3. Check Non-Deterministic Effects (Knowledge Loss)
                if (!modified_by_action) {
                    for (int nd_fact : action.non_deterministic_effects) {
                        if (nd_fact == token) {
                            modified_by_action = true;
                            // A non-deterministic regression implies we cannot guarantee its prior state.
                            regressed_rpn.push_back(token);
                            break;
                        }
                    }
                }

                // 4. Fluent was completely untouched by the action
                if (!modified_by_action) {
                    regressed_rpn.push_back(token); 
                }
            } else {
                // Forward logical operators (AND, OR, NOT, etc.) directly to the new RPN
                regressed_rpn.push_back(token); 
            }
        }
        return regressed_rpn;
    }

    /**
     * @brief Negates an entire RPN array by appending the NOT operator.
     */
    static std::vector<int> negate_rpn(const std::vector<int>& rpn) {
        if (rpn.empty()) return {OP_FALSE};
        std::vector<int> negated;
        negated.reserve(rpn.size() + 1);
        negated.insert(negated.end(), rpn.begin(), rpn.end());
        negated.push_back(OP_NOT);
        return negated;
    }
};

} // namespace CPOR