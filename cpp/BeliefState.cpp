#include "BeliefState.hpp"
#include "Regression.hpp"
#include "Evaluator.hpp"
#include "Z3Manager.hpp"

namespace CPOR {

void BeliefState::apply_forward_action(int action_id, const PartiallySpecifiedState& next_state) {
    // Snapshot the belief state at the exact time the action was chosen.
    state_history.push_back(current_state);

    action_history.push_back(action_id);
    current_state = next_state;
}

bool BeliefState::verify_condition_safely(const std::vector<int>& rpn_condition, const ProblemDef& global_problem) const {
    if (rpn_condition.empty()) return true;
    return verify_condition_safely_with_constraints(rpn_condition, global_problem, derive_learned_constraints(global_problem));
}

std::vector<int> BeliefState::regress_to_root(const std::vector<int>& rpn_condition, const ProblemDef& global_problem) const {
    std::vector<int> current_rpn = rpn_condition;

    // Regress chronologically backward through the action history using indices
    for (int i = static_cast<int>(action_history.size()) - 1; i >= 0; --i) {
        int action_id = action_history[i];
        const GroundedAction& action = global_problem.actions[action_id];
        const PartiallySpecifiedState& historical_belief = state_history[i];

        current_rpn = RegressionEngine::regress_rpn(current_rpn, action, historical_belief);
    }
    return current_rpn;
}

bool BeliefState::verify_condition_safely_with_constraints(const std::vector<int>& rpn_condition, const ProblemDef& global_problem,
                                                             const std::vector<std::vector<int>>& precomputed_constraints) const {
    if (rpn_condition.empty()) return true;
    return verify_regressed_condition_safely(regress_to_root(rpn_condition, global_problem), global_problem, precomputed_constraints);
}

bool BeliefState::verify_regressed_condition_safely(const std::vector<int>& already_regressed_rpn, const ProblemDef& global_problem,
                                                      const std::vector<std::vector<int>>& precomputed_constraints) const {
    if (already_regressed_rpn.empty()) return true;

    // Evaluate the fully regressed condition against the concrete initial state.
    // We use PESSIMISTIC mode: it must be absolutely TRUE, not UNKNOWN.
    if (Evaluator::evaluate(already_regressed_rpn, initial_state, EvalMode::PESSIMISTIC)) {
        return true;
    }

    // The cheap, invariant-blind check couldn't prove it -- fall back to Z3
    // entailment against the initial state's known facts, the problem's
    // global invariants (oneofs), and everything this history's own
    // observations imply about the initial state (see
    // derive_learned_constraints -- without these, regressing `rpn_condition`
    // alone is a no-op for any token no action along this history ever wrote
    // to directly, e.g. a hidden identity fact like `at(room)` whose only
    // observable trace is its effect on OTHER facts a sensing action reads).
    // Needed together with RegressionEngine::regress_rpn's disjunction
    // handling for a localization-style domain's per-room conditions:
    // Evaluator alone has no notion that those rooms are mutually exclusive,
    // so it can never resolve "room1 OR room2 OR ... OR (none held AND
    // already-known)" even when the oneof invariant makes it provably true.
    // See Z3Manager::formula_is_entailed.
    return g_z3_entailment_manager.formula_is_entailed(already_regressed_rpn, initial_state, global_problem, precomputed_constraints);
}

std::vector<std::vector<int>> BeliefState::derive_learned_constraints(const ProblemDef& global_problem, size_t start_step) const {
    std::vector<std::vector<int>> constraints;

    for (size_t i = start_step; i < action_history.size(); ++i) {
        const PartiallySpecifiedState& before = state_history[i];
        const PartiallySpecifiedState& after = (i + 1 < state_history.size()) ? state_history[i + 1] : current_state;

        for (int fact_id = 0; fact_id < global_problem.total_predicates; ++fact_id) {
            if (!before.is_unknown(fact_id) || after.is_unknown(fact_id)) continue;

            // fact_id became newly known exactly at step i (either this
            // step's action directly/conditionally established it, or -- for
            // the final step -- it's a genuine sensing observation). Treat it
            // as a target query about the state immediately AFTER step i, and
            // regress it backward through step i itself and everything
            // earlier (steps i down to 0 inclusive) to get a formula about
            // the initial state -- mirroring exactly how
            // verify_condition_safely regresses its own target query through
            // the WHOLE history. Regressing only i-1..0 (skipping step i) is
            // wrong whenever step i's own action is what established the
            // fact (e.g. a guaranteed effect): the fact's truth right after
            // step i doesn't mean it held before step i, but skipping the
            // regression through step i asserted exactly that.
            std::vector<int> learned_rpn = after.is_true(fact_id)
                ? std::vector<int>{fact_id}
                : RegressionEngine::negate_rpn({fact_id});

            for (int j = static_cast<int>(i); j >= 0; --j) {
                const GroundedAction& action = global_problem.actions[action_history[j]];
                const PartiallySpecifiedState& historical_belief = state_history[j];
                learned_rpn = RegressionEngine::regress_rpn(learned_rpn, action, historical_belief);
            }

            constraints.push_back(std::move(learned_rpn));
        }
    }

    return constraints;
}

} // namespace CPOR