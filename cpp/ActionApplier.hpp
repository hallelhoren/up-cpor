#pragma once

#include "State.hpp"
#include "ProblemData.hpp"
#include "Evaluator.hpp"
#include <memory>
#include <utility>
#include <unordered_set>

/**
 * @class ActionApplier
 * @brief Stateless system that safely transitions a state using CPOR Knowledge Loss rules.
 */
class ActionApplier {
private:
    // Returns {true, literals} if `rpn` is representable as a flat
    // conjunction of (possibly negated) fact literals combined only with
    // AND -- the same "simple problem" shape CPORSolver's own
    // try_extract_literal_conjunction targets for Plan Graph Compaction
    // (duplicated here rather than shared across translation units, to keep
    // this header self-contained with only State.hpp/Evaluator.hpp as
    // dependencies). Each literal is (fact_id, required_value): true means
    // the literal reads the fact directly, false means it reads NOT fact.
    // Returns {false, {}} for anything else -- OR/EQUALS/ONEOF anywhere, or
    // a NOT applied to more than a single literal -- exactly the same
    // "isn't flat-literal-representable" bailout as before this
    // multi-literal extension existed (see record_conditional_provenance's
    // old single-literal gate, which this generalizes).
    static std::pair<bool, std::vector<std::pair<int, bool>>> extract_flat_and_conjunction(const std::vector<int>& rpn) {
        std::vector<std::vector<std::pair<int, bool>>> stack;
        for (int token : rpn) {
            if (token >= 0) {
                stack.push_back({ {token, true} });
            } else if (token == OP_TRUE) {
                stack.push_back({});
            } else if (token == OP_NOT) {
                if (stack.empty()) return {false, {}};
                std::vector<std::pair<int, bool>> top = std::move(stack.back());
                stack.pop_back();
                if (top.size() != 1) return {false, {}}; // De Morgan'ing a conjunction isn't a flat literal.
                top[0].second = !top[0].second;
                stack.push_back(std::move(top));
            } else if (token == OP_AND) {
                if (stack.size() < 2) return {false, {}};
                std::vector<std::pair<int, bool>> right = std::move(stack.back()); stack.pop_back();
                std::vector<std::pair<int, bool>> left = std::move(stack.back()); stack.pop_back();
                left.insert(left.end(), right.begin(), right.end());
                stack.push_back(std::move(left));
            } else {
                // OP_OR, OP_EQUALS, OP_ONEOF, OP_FALSE: not a flat literal conjunction.
                return {false, {}};
            }
        }
        if (stack.size() != 1) return {false, {}};
        return {true, std::move(stack.back())};
    }

public:
    static void apply_action(const GroundedAction& action, PartiallySpecifiedState& state) {
        // PDDL/STRIPS simultaneous-effect semantics: every effect of an action
        // (guaranteed AND conditional) is evaluated against the state as it
        // was BEFORE the action fires, then all effects apply at once. If a
        // conditional effect's condition is evaluated against `state` only
        // after this same action's own guaranteed effects have already
        // mutated it in-place (the old behavior below), an action that both
        // unconditionally sets a fact and gates a conditional effect on that
        // same fact's PRE-action value can never fire that conditional effect
        // -- e.g. `(:effect (and (ok) (when (and (not (ok)) ...) ...)))`
        // always evaluates `(not (ok))` as false, because `ok` was already
        // flipped true by this call's own step 1. Snapshotting here so
        // conditions read pre-action values fixes that without changing
        // anything for actions whose conditions don't overlap their own
        // guaranteed effects.
        // Only pay for the copy when there's actually a condition to evaluate
        // against it; the vast majority of actions have no conditional
        // effects at all.
        std::unique_ptr<const PartiallySpecifiedState> pre_state_storage;
        const PartiallySpecifiedState* pre_state = &state;
        if (!action.conditional_effects.empty()) {
            pre_state_storage = std::make_unique<const PartiallySpecifiedState>(state);
            pre_state = pre_state_storage.get();
        }

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
        //
        // ambiguous_provenance_facts tracks, across THIS SAME apply_action
        // call only, which effect facts have already been degraded to
        // UNKNOWN by an earlier conditional effect below. A domain where two
        // (or more) of this action's conditional effects target the SAME
        // fact under DIFFERENT, mutually exclusive conditions (e.g. one
        // per-room `(when (and (not ok) (at room1)) (free-up ...))`,
        // `(when (and (not ok) (at room2)) (free-up ...))`, ... all writing
        // `free-up`) means that once BOTH conditions are simultaneously
        // unresolved, a later observation of `free-up` can no longer be
        // attributed to any ONE of them -- record_conditional_provenance
        // itself is keyed only by effect_fact_id, so a second call here
        // would otherwise silently overwrite the first's entry and let
        // apply_provenance_deductions confidently (and unsoundly) abduce
        // whichever condition happened to be processed LAST, even though it
        // was never the only candidate. Verified to manifest in practice:
        // localize5's `checking` action is exactly this shape, and without
        // this guard a later genuine sensing observation could contradict
        // the wrongly-abduced position outright (a hard OneOf contradiction
        // in SDRPlanner::apply_observation). Once a fact is flagged
        // ambiguous here, no further conditional effect in this action may
        // record provenance for it either -- see the eff loop below.
        std::unordered_set<int> ambiguous_provenance_facts;
        for (const auto& ce : action.conditional_effects) {
            uint8_t cond_result = Evaluator::evaluate_rpn_raw(ce.condition_rpn, *pre_state);

            if (cond_result == VAL_TRUE) {
                for (const auto& eff : ce.effects) {
                    state.clear_provenance(eff.first);
                    state.set_known_value(eff.first, eff.second);
                }
            }
            else if (cond_result == VAL_UNKNOWN) {
                // Scoped Regression Fix: if the condition is a flat AND
                // conjunction of (possibly negated) fact literals, remember
                // which of those literals were themselves still unknown at
                // apply time -- so a later direct observation of the effect
                // fact can be abduced backward into them. See
                // PartiallySpecifiedState::apply_provenance_deductions in
                // State.hpp for the deduction side (both the full-conjunction
                // and single-remaining-literal cases) and its scope/
                // soundness notes. A genuinely undecidable OR/EQUALS/ONEOF
                // condition, or one where 2+ literals remain unresolved on
                // the "conjunction was false" branch, still isn't (fully)
                // trackable -- soundly inverting those is out of scope here.
                auto extracted = extract_flat_and_conjunction(ce.condition_rpn);
                std::vector<std::pair<int, bool>> unknown_literals;
                if (extracted.first) {
                    for (const auto& lit : extracted.second) {
                        if (pre_state->is_unknown(lit.first)) {
                            unknown_literals.push_back(lit);
                        }
                    }
                }
                for (const auto& eff : ce.effects) {
                    bool already_unknown_this_call = !ambiguous_provenance_facts.insert(eff.first).second;
                    state.set_unknown(eff.first);
                    if (already_unknown_this_call) {
                        // See the class-level comment above: a second
                        // conditional effect in this same action degraded
                        // this fact too, so any provenance recorded for it
                        // (by this ce or an earlier one) is no longer
                        // attributable to a single condition -- discard it.
                        state.clear_provenance(eff.first);
                    } else if (!unknown_literals.empty()) {
                        state.record_conditional_provenance(eff.first, unknown_literals, eff.second);
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