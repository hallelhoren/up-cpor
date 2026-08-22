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
    // state_history[i] is the belief immediately BEFORE action_history[i] fired (so
    // state_history[0] == initial_state always). Exposed for Z3BMCManager::build, which
    // needs the belief at every point in history, not just the endpoints, to anchor
    // genuine observations an action's own declared effects don't capture (e.g. a
    // sensing action's :observe outcome) -- see that class's own doc comment.
    const std::vector<PartiallySpecifiedState>& get_state_history() const { return state_history; }

    /**
     * @brief Transitions the current belief state forward and logs the history.
     */
    void apply_forward_action(int action_id, const PartiallySpecifiedState& next_state);

    /**
     * @brief Checks if an RPN condition is guaranteed to hold true given the
     * execution history, by regressing the condition backward to the initial state bounds.
     */
    bool verify_condition_safely(const std::vector<int>& rpn_condition, const ProblemDef& global_problem) const;

    // Same check as verify_condition_safely, but takes the "what history already
    // implies about the initial state" constraints as a precomputed argument instead
    // of deriving them internally via derive_learned_constraints(). Callers who
    // already have these cached (e.g. CPORSolver::get_node_learned_constraints's
    // incremental, per-node-idx cache) should use this to avoid paying for the same
    // O(depth)-ish regression work a second time -- calling the plain overload above
    // from a hot path defeats that caching entirely, since it always re-derives from
    // scratch regardless of what the caller already knows.
    bool verify_condition_safely_with_constraints(const std::vector<int>& rpn_condition, const ProblemDef& global_problem,
                                                    const std::vector<std::vector<int>>& precomputed_constraints) const;

    // Just the backward-regression half of verify_condition_safely (no evaluate, no
    // Z3) -- regresses rpn_condition through this belief's whole action_history down
    // to a formula about initial_state's own literals. Factored out so callers doing
    // their own incremental caching over the regression itself (e.g.
    // CPORSolver::get_regressed_query) can reuse this exact logic instead of
    // duplicating it.
    std::vector<int> regress_to_root(const std::vector<int>& rpn_condition, const ProblemDef& global_problem) const;

    // The second half of verify_condition_safely_with_constraints (evaluate against
    // initial_state, then Z3-entailment fallback) taking an ALREADY-regressed formula
    // directly -- for callers with their own incremental cache over regress_to_root
    // itself (e.g. CPORSolver::get_regressed_query), who'd otherwise pay for the same
    // regression a second time by going through the plain overload.
    bool verify_regressed_condition_safely(const std::vector<int>& already_regressed_rpn, const ProblemDef& global_problem,
                                             const std::vector<std::vector<int>>& precomputed_constraints) const;

    // Every fact that became newly known at some point in this history (an
    // action's guaranteed/definite effect, or a genuine sensing observation),
    // regressed backward through this history down to a formula about the
    // initial state. See verify_condition_safely's .cpp comment for why this
    // -- not just regressing a single target query -- is what actually lets
    // an observation inform a query about a fact the observing action never
    // directly wrote to (e.g. a hidden identity fact like `at(room)` that
    // only the *consequences* of being at that room, not `at(room)` itself,
    // ever get observed).
    //
    // Public (not just verify_condition_safely's own internal use) so
    // witness sampling can assert the SAME accumulated knowledge when
    // determinizing a belief -- see SDRSampler::sample_concrete_states'
    // extra_constraints parameter and its callers in SDRPlanner. Without
    // this, a freshly-sampled witness can re-guess a fact (like a hidden
    // position) inconsistently with everything this history has already
    // ruled out, steering the classical sub-solver toward a plan that keeps
    // repeating an already-failed physical move instead of a genuinely new
    // one.
    //
    // start_step (default 0, i.e. the whole history) restricts which history
    // steps are scanned for newly-learned facts to i >= start_step -- letting
    // a caller who already has the constraints for a shorter prefix of this
    // SAME belief (e.g. its parent node in a search tree, since these
    // constraints are all phrased as facts about the shared initial_state and
    // so never get invalidated by history that happens afterward) ask only
    // for what's NEW since that prefix, instead of re-deriving everything
    // from scratch. Each newly-learned fact still gets regressed all the way
    // back through steps i..0 regardless of start_step -- that inner walk is
    // what makes the result a valid initial-state-level formula, and cannot
    // be shortened. Cutting the OUTER scan is still a real asymptotic win
    // when called incrementally once per history step, rather than once per
    // query at whatever depth the query happens to be at: it turns the total
    // cost of deriving constraints for every node from the root down to depth
    // D from O(D^2) into O(D) (each step only pays for its own new facts,
    // regressed back through however-deep it is, instead of every step's
    // facts being re-regressed from scratch on every subsequent query).
    std::vector<std::vector<int>> derive_learned_constraints(const ProblemDef& global_problem, size_t start_step = 0) const;
};

} // namespace CPOR