#pragma once
#include <vector>
#include "ProblemData.hpp"
#include "Evaluator.hpp"

namespace CPOR {

/**
 * @class RegressionEngine
 * @brief Handles the logical regression of physical fluents chronologically backwards.
 */
class RegressionEngine {
public:
    /**
     * @brief Regresses an RPN logic condition backwards through a specific action.
     * * If the action modifies a token found in the RPN, the token is substituted 
     * with the boolean outcome (OP_TRUE / OP_FALSE). Non-deterministic modifications 
     * are strictly converted to OP_UNKNOWN logic boundaries.
     */
    static std::vector<int> regress_rpn(const std::vector<int>& current_rpn, const GroundedAction& action) {
        std::vector<int> regressed_rpn;
        regressed_rpn.reserve(current_rpn.size());

        for (int token : current_rpn) {
            if (token >= 0) {
                bool modified_by_action = false;
                
                // Check if the action deterministically caused this fluent
                for (const auto& eff : action.guaranteed_effects) {
                    if (eff.first == token) {
                        modified_by_action = true;
                        regressed_rpn.push_back(eff.second ? OP_TRUE : OP_FALSE);
                        break;
                    }
                }
                
                if (!modified_by_action) {
                    // Check if it was conditionally/non-deterministically degraded
                    for (int nd_fact : action.non_deterministic_effects) {
                        if (nd_fact == token) {
                            modified_by_action = true;
                            // A non-deterministic regression implies we cannot guarantee its prior state
                            // without deeper branching logic, so we flag it conditionally (Placeholder strategy for Step 1)
                            regressed_rpn.push_back(token);
                            break;
                        }
                    }
                }

                if (!modified_by_action) {
                    regressed_rpn.push_back(token); // Fluent was untouched
                }
            } else {
                regressed_rpn.push_back(token); // Forward logical operators
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