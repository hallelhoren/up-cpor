#pragma once
#include <vector>
#include <unordered_map>
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

        // Per-call memoization: what a given fluent token substitutes to is entirely
        // determined by (token, action, current_belief), all fixed for this whole
        // call -- so if the SAME token appears more than once in current_rpn (routine
        // once an earlier regression step has already expanded a query into a
        // multi-branch disjunction referencing the same underlying fact several
        // times, e.g. one `at(pX)` literal per branch across an OR-chain), computing
        // its substitution again for every repeat is pure waste. Confirmed
        // empirically to matter: chaining regression through several
        // conditional-effect-heavy actions (localize5's `checking`, called
        // repeatedly along a real search path) was measured to blow a single-fact
        // query up to hundreds of thousands of RPN tokens, almost entirely repeated
        // substructure from re-expanding the same handful of distinct tokens over
        // and over. This doesn't change what gets computed, only how many times --
        // a token's substitution is a pure function of the three call-invariant
        // inputs above, so caching it within one call cannot change the result.
        // unordered_map element references stay valid across further insertions
        // (only erase invalidates them), so storing the substitution vectors
        // directly as map values -- rather than in a separate vector-of-vectors
        // with pointers into it -- is both simpler and safe against the reallocate-
        // and-dangle hazard a growing std::vector<std::vector<int>> would create.
        std::unordered_map<int, std::vector<int>> token_substitution_cache;

        for (int token : current_rpn) {
            // If token >= 0, it's a physical fluent ID. Otherwise, it's a logical operator.
            if (token >= 0) {
                auto cached = token_substitution_cache.find(token);
                if (cached != token_substitution_cache.end()) {
                    regressed_rpn.insert(regressed_rpn.end(), cached->second.begin(), cached->second.end());
                    continue;
                }

                std::vector<int> this_token_substitute;
                bool modified_by_action = false;

                // 1. Check Guaranteed Effects (Highest Priority)
                for (const auto& eff : action.guaranteed_effects) {
                    if (eff.first == token) {
                        modified_by_action = true;
                        // Substitute the variable with its absolute post-action truth value
                        this_token_substitute.push_back(eff.second ? OP_TRUE : OP_FALSE);
                        break;
                    }
                }

                // 2. Check Conditional Effects (The SDR Fix)
                //
                // Collect every conditional effect that targets this token (a
                // domain like localize5 can have several -- one per mutually
                // exclusive room -- all writing the same fact) rather than
                // stopping at the first one, so a token gated on more than one
                // still-unresolved condition regresses to a genuine
                // disjunction instead of silently giving up. See
                // substitute_from_conditional_effects below for how that
                // disjunction is built.
                if (!modified_by_action) {
                    bool found_definite_true = false;
                    bool definite_true_value = false;
                    std::vector<std::pair<std::vector<int>, bool>> unresolved_ces; // (condition_rpn, effect_value)

                    for (const auto& ce : action.conditional_effects) {
                        bool affects_token = false;
                        bool effect_value = false;
                        for (const auto& eff : ce.effects) {
                            if (eff.first == token) {
                                affects_token = true;
                                effect_value = eff.second;
                                break;
                            }
                        }
                        if (!affects_token) continue;

                        uint8_t cond_eval = Evaluator::evaluate_rpn_raw(ce.condition_rpn, current_belief);
                        if (cond_eval == VAL_TRUE) {
                            // Mirrors ActionApplier's own forward semantics: the
                            // first (and, for a well-formed domain whose
                            // per-token conditions are mutually exclusive,
                            // only) definitely-true condition fully determines
                            // the result -- no need to consider the others.
                            found_definite_true = true;
                            definite_true_value = effect_value;
                            break;
                        } else if (cond_eval == VAL_UNKNOWN) {
                            unresolved_ces.emplace_back(ce.condition_rpn, effect_value);
                        }
                        // VAL_FALSE: this branch definitely didn't fire, keep checking the others.
                    }

                    if (found_definite_true) {
                        modified_by_action = true;
                        this_token_substitute.push_back(definite_true_value ? OP_TRUE : OP_FALSE);
                    } else if (!unresolved_ces.empty()) {
                        modified_by_action = true;
                        this_token_substitute = substitute_from_conditional_effects(unresolved_ces, token);
                    }
                    // If every affecting CE evaluated definitely FALSE, none of
                    // them fired -- fall through exactly like "untouched by
                    // any CE" below, so the token keeps regressing through
                    // earlier history.
                }

                // 3. Check Non-Deterministic Effects (Knowledge Loss)
                if (!modified_by_action) {
                    for (int nd_fact : action.non_deterministic_effects) {
                        if (nd_fact == token) {
                            modified_by_action = true;
                            // A non-deterministic regression implies we cannot guarantee its prior state.
                            this_token_substitute.push_back(token);
                            break;
                        }
                    }
                }

                // 4. Fluent was completely untouched by the action
                if (!modified_by_action) {
                    this_token_substitute.push_back(token);
                }

                regressed_rpn.insert(regressed_rpn.end(), this_token_substitute.begin(), this_token_substitute.end());
                token_substitution_cache.emplace(token, std::move(this_token_substitute));
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

private:
    // Folds a list of RPN sub-formulas into a single left-associative OR
    // chain: {a, b, c} -> "a b OR c OR" (valid RPN, since repeatedly
    // appending "operand OP" to a stack machine's program is always
    // well-formed regardless of how many operands preceded it).
    static std::vector<int> or_chain(const std::vector<std::vector<int>>& parts) {
        if (parts.empty()) return {OP_FALSE};
        std::vector<int> result = parts[0];
        for (size_t i = 1; i < parts.size(); ++i) {
            result.insert(result.end(), parts[i].begin(), parts[i].end());
            result.push_back(OP_OR);
        }
        return result;
    }

    static std::vector<int> and_chain(const std::vector<std::vector<int>>& parts) {
        if (parts.empty()) return {OP_TRUE};
        std::vector<int> result = parts[0];
        for (size_t i = 1; i < parts.size(); ++i) {
            result.insert(result.end(), parts[i].begin(), parts[i].end());
            result.push_back(OP_AND);
        }
        return result;
    }

    // Builds the regressed substitute for `token` given every conditional
    // effect that targets it whose condition is still unresolved (VAL_UNKNOWN)
    // against the historical belief -- i.e. none is known to have fired, but
    // none is ruled out either. Assumes (as ActionApplier's own forward
    // semantics already do) that these conditions are mutually exclusive in
    // any real historical state -- at most one actually held.
    //
    // token's regressed value is TRUE iff:
    //   - some condition whose effect sets the token TRUE actually held, OR
    //   - none of these conditions held, and the token's own PRE-action
    //     value (passed through for the caller's next, earlier regression
    //     step) was true.
    // This is strictly more informative than the old behavior of simply
    // giving up and passing `token` through unresolved: it captures that
    // e.g. a single still-unknown condition being confirmed true would
    // settle the token's value, which a bare pass-through discarded. With
    // several conditions (e.g. one per mutually exclusive room in a
    // localization domain), it produces a genuine disjunction that a
    // downstream Z3-backed entailment check can resolve once combined with
    // the domain's own mutual-exclusion (oneof) invariant -- see
    // Z3Manager::formula_is_entailed.
    static std::vector<int> substitute_from_conditional_effects(
        const std::vector<std::pair<std::vector<int>, bool>>& unresolved_ces,
        int token)
    {
        std::vector<std::vector<int>> all_conditions;
        std::vector<std::vector<int>> true_conditions;
        all_conditions.reserve(unresolved_ces.size());
        for (const auto& [condition_rpn, effect_value] : unresolved_ces) {
            all_conditions.push_back(condition_rpn);
            if (effect_value) true_conditions.push_back(condition_rpn);
        }

        std::vector<int> none_fired = negate_rpn(or_chain(all_conditions));
        std::vector<int> passthrough_term = and_chain({none_fired, {token}});

        if (true_conditions.empty()) {
            return passthrough_term;
        }
        return or_chain({or_chain(true_conditions), passthrough_term});
    }
};

} // namespace CPOR