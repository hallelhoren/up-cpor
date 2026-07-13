#pragma once
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <iostream>
#include <limits>
#include <new>

#include "State.hpp"
#include "Evaluator.hpp"
#include "ProblemData.hpp"
#include "SDRSampler.hpp"
#include "BFSSolver.hpp"
#include "ActionApplier.hpp"
#include "SDRPlanner.hpp"
#include "BeliefState.hpp"

namespace CPOR { 

struct PlanNode {
    int action_id{-1}; 
    PartiallySpecifiedState state;
    
    // For classical actions (OR nodes)
    int single_child_idx{-1};
    
    // For sensing actions (AND nodes)
    int true_child_idx{-1};
    int false_child_idx{-1};

    int parent_idx{-1};
    int generating_action_id{-1};

    bool is_solved{false};
    bool is_failed{false};
    int chosen_action_id{-1};

    PlanNode(const PartiallySpecifiedState& s, int act_id) : state(s), action_id(act_id) {}
};

class CPORSolver {
private:
    std::vector<PlanNode> node_pool;
    std::unordered_map<PartiallySpecifiedState, int, StateHasher> solved_cache;
    std::unordered_set<PartiallySpecifiedState, StateHasher> failed_cache;
    // Memoizes compute_heuristic() by belief state: the candidate-generation loop in
    // solve_from_node calls it once per applicable action per node, so the same
    // projected state reached via different candidates/branches would otherwise pay
    // for a fresh bounded BFS every time.
    std::unordered_map<PartiallySpecifiedState, int, StateHasher> heuristic_cache;
    
    struct ActionCandidate {
        int action_idx;
        int h_score;
        
        bool operator<(const ActionCandidate& other) const {
            return h_score < other.h_score;
        }
    };

    // Internal Epistemic Logic
    void apply_oneof_deductions(PartiallySpecifiedState& state);
    void expand_sensing_node(int node_idx, int action_id, int observe_fluent, const PartiallySpecifiedState& current_state);
    
    bool is_action_applicable(const GroundedAction& action, const PartiallySpecifiedState& state);
    int compute_heuristic(const PartiallySpecifiedState& state);

    // -------------------------------------------------------------------
    // CPOR outer-loop machinery (solve_cpor_loop and its helpers below)
    // -------------------------------------------------------------------
    // How many diverse Z3 witness states SDRPlanner::compute_linear_plan samples
    // per OnlinePlan call. Larger catches more epistemic-disjunction mismatches
    // before the outer loop's own precondition re-check has to catch them (see
    // solve_cpor_loop's comment), at the cost of more Z3 solver.check() calls
    // per deliberation.
    static constexpr int CPOR_LOOP_WITNESS_SAMPLE_COUNT = 5;

    // Counts how many times fallback_to_exhaustive_search actually ran during
    // the most recent solve_cpor_loop call. Reset implicitly every time a new
    // CPORSolver is constructed (native_bridge.cpp's solve_native_cpor_loop()
    // always allocates a fresh instance), so callers can tell whether the
    // layered fallback (Option 1) was needed at all for a given problem.
    int fallback_invocation_count{0};

    // Runs solve_from_node for node_idx's whole subtree as a safety net when
    // the online loop can't make sound progress there (see solve_cpor_loop's
    // class-level comment for why this is required rather than optional), then
    // notifies node_idx's parent via close_node_and_propagate either way.
    // Returns node_idx's resulting is_solved value.
    bool fallback_to_exhaustive_search(int node_idx);

    // Bottom-up closure propagation for the CPOR outer loop (Algorithm 4 in
    // Maliah, Komarnitski & Shani 2022). The outer loop processes a sensing
    // node's two children asynchronously (they're popped from the stack on
    // separate, later iterations, unlike solve_from_node's synchronous
    // recursive return), so closing either child must check whether its
    // parent is now fully resolved and, if so, close the parent too and
    // recurse upward. If the parent is a sensing (AND) node whose two
    // children disagree (one solved, one failed), that specific committed
    // sensing action cannot work -- falls back to solve_from_node for the
    // parent's whole subtree rather than giving up (see solve_cpor_loop).
    void close_node_and_propagate(int node_idx);

    // When a proposed action's precondition evaluates to VAL_UNKNOWN (not
    // VAL_FALSE) against the real belief -- i.e. it looked true across the
    // sampled witnesses but isn't actually known -- looks for an applicable
    // sensing action observing one of the still-unknown fluents referenced in
    // that precondition, mirroring SDRPlanner::plan_to_observe_deadend's
    // "extract the blocking unknown fluents, find a sensing action for one of
    // them" pattern. Returns the sensing action's id, or -1 if none is
    // currently applicable.
    int find_resolving_sensing_action(const std::vector<int>& blocked_precondition_rpn, const PartiallySpecifiedState& belief);

    // Attempts to turn a blocked/truncated action (blocking_action_id, as
    // surfaced by SDRPlanner::compute_linear_plan's out_blocking_action_id,
    // or discovered directly during partial-plan replay) into a genuine
    // sensing branch at cursor_idx: looks up a resolving sensing action via
    // find_resolving_sensing_action and, if one is applicable, branches via
    // expand_sensing_node and pushes both new children onto open_stack.
    // Returns true if a branch was created (caller must not also fall back
    // for this node); false if blocking_action_id is -1 or no resolving
    // sensing action is currently applicable (caller should fall back).
    // Shared by all three of solve_cpor_loop's truncation exit points so the
    // "found a blocked action -> try to resolve it via sensing" logic can't
    // drift out of sync between them.
    bool try_resolve_via_sensing_branch(int cursor_idx, int blocking_action_id, const PartiallySpecifiedState& belief,
                                         const ProblemDef& problem, std::vector<int>& open_stack);

public:
    CPORSolver() {
        node_pool.reserve(500000); // Pre-allocate Arena to avoid vector reallocation invalidating indices
    }

    const PlanNode& get_node(int idx) const {
        return node_pool[idx];
    }

    // How many times fallback_to_exhaustive_search actually ran during the
    // most recent solve_cpor_loop call. Zero means the online-planner loop
    // resolved the entire problem on its own, without needing the layered
    // fallback's safety net at all.
    int get_fallback_invocation_count() const { return fallback_invocation_count; }

    int create_root_node(const PartiallySpecifiedState& initial_state);
    // out_lowlink (if non-null) reports a Tarjan-style low-link value for this call's
    // result, in the same units as `depth`/stack position within current_path_indices:
    //   - NO_CYCLE (INT_MAX): the result is a fully proven conclusion (goal reached,
    //     a cache hit, or a failure whose only cyclic dependencies bottomed out within
    //     this node's own subtree) -- safe to cache permanently.
    //   - Any other value P: the result genuinely depends on an ancestor at stack
    //     position P that has not finished being resolved yet (P < depth means a
    //     proper ancestor above this node; P == depth means this node's own position,
    //     still self-contained -- see solve_from_node's definition of NO_CYCLE). Must
    //     not be cached at this level; the caller propagates it further up via its own
    //     out_lowlink so the ancestor at position P can eventually resolve it.
    //   - -1 is used for a depth-cap cutoff: a horizon limit, not a real cycle, so it
    //     never satisfies "self-contained" at any node and permanently blocks caching
    //     along the whole current path, exactly like the prior always-non-definitive
    //     behavior for the depth cap specifically.
    static constexpr int NO_CYCLE = std::numeric_limits<int>::max();
    bool solve_from_node(int node_idx, std::vector<int>& current_path_indices, int depth = 0, int* out_lowlink = nullptr);

    // Builds the plan tree using the stack-based CPOR outer loop (Maliah,
    // Komarnitski & Shani, "Computing Contingent Plan Graphs using Online
    // Planning", TAAS 2022, Algorithm 2), matching CPORPlanner.OfflinePlanning's
    // shape in the legacy C# engine: pop an open node, ask an online planner
    // (SDRPlanner::compute_linear_plan) for a partial plan, execute it action
    // by action re-validating each precondition against the actual belief
    // before committing, and push both children of a sensing action back onto
    // the stack for later, independent resolution. Populates the SAME
    // node_pool/PlanNode structure and Python-facing extraction contract
    // (get_chosen_action/get_single_child/get_true_child/get_false_child) as
    // solve_from_node, so native_bridge.cpp and problem_grounder.py need no
    // changes -- they don't know or care which algorithm built the tree.
    //
    // Layered fallback (deliberately NOT a pure replacement of solve_from_node):
    // the CPOR paper's own soundness proof (Theorem 4.1 / Lemma 2) assumes
    // deadend-free domains and offers no recovery when a committed
    // sensing-action choice makes one branch unrecoverable -- but this engine
    // has explicit deadend_rpns and genuinely reachable dead ends. Whenever the
    // online loop can't make sound progress at a node -- OnlinePlan proposes
    // nothing, proposes an action whose precondition isn't actually satisfied
    // and no sensing action can resolve the gap, or a sensing commitment
    // leaves one child branch definitively failed -- that node's whole
    // subtree is handed to fallback_to_exhaustive_search instead of
    // being given up on. solve_from_node itself is unmodified and remains
    // fully available on its own (native_bridge.cpp can still call it
    // directly), so this is purely additive.
    bool solve_cpor_loop(int root_idx);

    size_t get_node_count() const { return node_pool.size(); }
    void print_conditional_plan(int node_idx, int indent = 0) const;
};

} // namespace CPOR