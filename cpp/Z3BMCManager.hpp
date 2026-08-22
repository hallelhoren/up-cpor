#pragma once
#include <z3++.h>
#include <vector>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <stdexcept>
#include <optional>
#include "ProblemData.hpp"
#include "State.hpp"
#include "Evaluator.hpp"
#include "DebugStats.hpp"
#include <chrono>
#include <cstdint>

namespace CPOR {

/**
 * @class Z3BMCManager
 * @brief A persistent, TREE-INDEXED Bounded Model Checking (BMC) cache: answers
 * "is fact_id == desired_value impossible given node_idx's history" and "sample a
 * concrete world consistent with node_idx's history plus one more action" without
 * ever re-deriving a node's ancestor history from scratch.
 *
 * HISTORY: the first version of this class was time-indexed (one variable row per
 * step 0..T of a SINGLE linear history, rebuilt fresh via BeliefState for every
 * node_idx that needed it) -- a huge win over RegressionEngine::regress_rpn's flat
 * formula substitution (which was blowing up past a million tokens by depth ~20),
 * but it still paid O(depth x total_predicates) to rebuild each node's own encoding
 * from t=0, every time, with zero sharing across nodes. Since solve_from_node's
 * search is a TREE (many nodes sharing long common ancestor prefixes, not one
 * linear path), that rebuild-from-scratch cost was still being paid once per node,
 * uncached across the tree -- confirmed to still dominate wall-clock time even
 * after the time-indexed version fixed the original formula-size blowup.
 *
 * This version fixes that the same way get_node_learned_constraints fixed the
 * analogous problem for forward constraint derivation: index by NODE instead of by
 * time step, and cache each node's assertion list as an INCREMENTAL EXTENSION of
 * its parent's already-cached list, not a rebuild. The key enabler is that all
 * z3::expr objects here live in ONE shared z3::context for the whole tree -- once
 * built, a z3::expr is a cheap, reference-counted handle to an AST node that can be
 * asserted into any number of different z3::solver instances for near-zero
 * additional cost (no re-parsing, no re-deriving). So "node_idx inherits its
 * parent's encoding" is implemented as: copy the parent's std::vector<z3::expr>
 * (cheap -- copying handles, not rebuilding the formulas they point to) and append
 * only THIS node's own new transition's assertions, built once. A node with a
 * cached entry costs O(total_predicates) to add to the tree, not O(depth x
 * total_predicates) -- turning what was still an O(depth) cost per node into
 * O(1)-amortized per node across the whole search tree, matching
 * get_node_learned_constraints's own asymptotic fix for the forward direction.
 *
 * Root is always node_idx 0 (CPORSolver::create_root_node's own contract). Callers
 * build the tree top-down as solve_from_node naturally visits nodes (a node's
 * parent is always visited, and thus already cached, before the node itself), so
 * build_child never needs to recurse to find an uncached ancestor.
 */
class Z3BMCManager {
public:
    explicit Z3BMCManager(int total_predicates)
        : total_predicates_(total_predicates), path_solver_(ctx_) {}

    bool has(int node_idx) const { return vars_by_node_.find(node_idx) != vars_by_node_.end(); }

    // Frees node_idx's own cached vars/assertions once its solve_from_node call
    // has definitively returned (see CPORSolver.cpp's BmcNodeScope, which calls
    // this from its destructor right after exit_node()). Safe unconditionally:
    // - Never called while node_idx is still on path_stack_ (BmcNodeScope's
    //   RAII lifetime IS enter_node/exit_node's own scope, so eviction can only
    //   happen after this node has already left the active path -- exactly
    //   when enter_node's FAST PATH would need a DIFFERENT, still-open node's
    //   entry, never a just-closed one).
    // - A node_idx CAN be legitimately re-entered later (see solve_from_node's
    //   own doc comment on fallback_to_exhaustive_search re-invocation) --
    //   ensure_node_bmc's existing has()-check already treats a missing entry
    //   as "not yet built" and transparently rebuilds it (recursing through any
    //   equally-evicted ancestors first, via the exact same logic that already
    //   handles a node visited for the very first time), from node_pool's own
    //   state/action_id/parent_idx, which this never touches. So eviction only
    //   ever costs a bounded, deterministic rebuild if re-entry happens -- it
    //   can never produce a wrong answer, since a given node_idx's rebuilt
    //   encoding is always bit-for-bit identical to what it replaced.
    // Deliberately does NOT touch impossible_cache_: entries there stay valid
    // forever once written (same node_idx -> same rebuilt encoding -> same
    // query answer), so a stale entry for an evicted-then-later-rebuilt node is
    // still correct, and node_idx values are never reused, so an entry for a
    // node that's simply gone for good just sits unused (a handful of bytes
    // per entry -- negligible next to the z3::expr vectors this targets).
    void evict(int node_idx) {
        vars_by_node_.erase(node_idx);
        assertions_by_node_.erase(node_idx);
        DebugStats::record_bmc_eviction();
    }

    // Registers the root (node_idx 0): fresh vars for the initial state, its own
    // known facts asserted, oneof invariants asserted.
    void build_root(int node_idx, const PartiallySpecifiedState& initial_state, const ProblemDef& problem) {
        std::vector<z3::expr> row = fresh_var_row(node_idx);
        std::vector<z3::expr> assertions;

        for (int f = 0; f < total_predicates_; ++f) {
            if (!initial_state.is_unknown(f)) {
                assertions.push_back(initial_state.is_true(f) ? row[f] : !row[f]);
            }
        }
        append_oneofs(assertions, row, problem);

        vars_by_node_.emplace(node_idx, std::move(row));
        assertions_by_node_.emplace(node_idx, std::move(assertions));
    }

    // Registers node_idx as parent_idx's child via `action`, whose resulting flat
    // state is `next_state`. parent_idx MUST already be registered (via build_root
    // or an earlier build_child) -- see class doc comment for why the natural
    // top-down visitation order of solve_from_node always guarantees this.
    // For facts an action doesn't touch, the child's variable is literally the
    // SAME Z3 constant as the parent's (not a fresh one tied to it by a frame-
    // axiom equality). This matters because assertions_by_node_ accumulates down
    // the whole ancestor chain: with a fresh constant every node, every oneof
    // group and every frame axiom had to be re-asserted at EVERY node even for
    // facts that never change along a long run of non-moving actions (confirmed
    // via CPOR_DEBUG_STATS: bmc_readd_avg_assertions plateaued around ~2570 on
    // localize5 even after fixing the oneof O(n^2) blowup, dominated by exactly
    // this kind of redundant re-assertion). Reusing the identical handle makes
    // the fact's constraints (oneof membership included) inherited for free --
    // nothing new needs to be asserted for it at this node at all.
    void build_child(int node_idx, int parent_idx, const GroundedAction& action,
                      const PartiallySpecifiedState& next_state, const ProblemDef& problem) {
        const std::vector<z3::expr>& parent_row = vars_by_node_.at(parent_idx);
        std::unordered_set<int> touched = compute_touched(action, problem);

        std::vector<z3::expr> row;
        row.reserve(total_predicates_);
        for (int f = 0; f < total_predicates_; ++f) {
            row.push_back(touched.count(f) ? fresh_var(node_idx, f) : parent_row[f]);
        }

        // Copy (not reference) the parent's list: cheap, since each element is a
        // lightweight handle into the shared context, not the formula itself.
        std::vector<z3::expr> assertions = assertions_by_node_.at(parent_idx);

        append_transition(assertions, parent_row, row, action, problem);
        append_oneofs(assertions, row, problem, &touched);
        for (int f = 0; f < total_predicates_; ++f) {
            if (!next_state.is_unknown(f)) {
                assertions.push_back(next_state.is_true(f) ? row[f] : !row[f]);
            }
        }

        vars_by_node_.emplace(node_idx, std::move(row));
        assertions_by_node_.emplace(node_idx, std::move(assertions));
    }

    // Advances the single persistent path_solver_ so its assertion stack
    // represents exactly node_idx's own cumulative history, and remembers
    // node_idx as the new top of the path. Every call MUST be matched by
    // exactly one exit_node() call, even across exceptions -- callers use an
    // RAII guard (see CPORSolver.cpp's BmcNodeScope) so this invariant can
    // never be violated by an early return.
    //
    // FAST PATH (current_top_ == parent_idx): this is a genuine recursive
    // descent from an already-open parent frame -- exactly what
    // solve_from_node's own call structure produces for its three recursive
    // call sites. push() ONE new scope and assert only the tail this node
    // appended beyond its parent's own list: O(this node's own delta), not
    // O(depth). This is what actually eliminates the O(depth) re-assertion
    // that used to dominate every single query (the old make_solver(), which
    // rebuilt a fresh solver by re-adding the ENTIRE cumulative list on every
    // call -- confirmed via CPOR_DEBUG_STATS to cost ~20s of a 60s run on
    // localize5-tamer even after the assertion list itself was already
    // minimal).
    //
    // SLOW PATH (anything else -- the very first call, or entry from outside
    // the natural recursion, e.g. fallback_to_exhaustive_search jumping into
    // an arbitrary node never visited by the currently-open recursion): reset()
    // the whole solver back to empty and push() ONE level containing node_idx's
    // FULL cumulative assertion list -- no worse than the old make_solver(),
    // and every subsequent recursive descent from here on gets the fast path.
    void enter_node(int node_idx, int parent_idx) {
        auto t0 = DebugStats::enabled() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        const auto& my_assertions = assertions_by_node_.at(node_idx);
        long added = 0;
        if (parent_idx != -1 && current_top_ == parent_idx) {
            size_t parent_size = assertions_by_node_.at(parent_idx).size();
            path_solver_.push();
            for (size_t k = parent_size; k < my_assertions.size(); ++k) {
                path_solver_.add(my_assertions[k]);
            }
            added = static_cast<long>(my_assertions.size() - parent_size);
        } else {
            path_solver_.reset();
            path_solver_.push();
            for (const auto& e : my_assertions) path_solver_.add(e);
            added = static_cast<long>(my_assertions.size());
            path_stack_.clear();
        }
        if (DebugStats::enabled()) {
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            DebugStats::record_bmc_readd(ms, added);
        }
        path_stack_.push_back(node_idx);
        current_top_ = node_idx;
    }

    // Pops exactly the one scope the matching enter_node() call pushed.
    void exit_node() {
        path_solver_.pop();
        path_stack_.pop_back();
        current_top_ = path_stack_.empty() ? -1 : path_stack_.back();
    }

    // True iff fact_id == desired_value is provably impossible at node_idx's own
    // state, given everything cached for it (its whole ancestor history).
    // node_idx MUST be the current path_solver_ top (i.e. the caller's own
    // enter_node(node_idx, ...) must already be active) -- queries use a
    // temporary push()/pop() scoped on top of the ALREADY-SYNCED path solver,
    // instead of rebuilding a fresh solver from node_idx's full assertion list
    // on every call.
    //
    // Memoized: node_idx's cached assertions never change once built (build_child/
    // build_root are each called at most once per node_idx, and nothing else ever
    // mutates assertions_by_node_), so is_impossible is a pure function of
    // (node_idx, fact_id, desired_value) -- safe to cache unconditionally. This
    // catches cases where the same query recurs (e.g. a rescue-path resample
    // re-deriving something a branch-admissibility check already proved for the
    // same node/fact) without paying for another solver push/check/pop.
    bool is_impossible(int node_idx, int fact_id, bool desired_value) {
        if (node_idx != current_top_) {
            throw std::runtime_error("Z3BMCManager::is_impossible: node_idx is not the current path top -- caller must enter_node() first");
        }
        int64_t key = (static_cast<int64_t>(node_idx) << 32) | (static_cast<int64_t>(fact_id) << 1) | (desired_value ? 1 : 0);
        auto it = impossible_cache_.find(key);
        if (it != impossible_cache_.end()) return it->second;

        path_solver_.push();
        path_solver_.add(vars_by_node_.at(node_idx)[fact_id] == ctx_.bool_val(desired_value));
        auto t0 = DebugStats::enabled() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        bool result = path_solver_.check() == z3::unsat;
        if (DebugStats::enabled()) {
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            DebugStats::record_bmc_check(ms);
        }
        path_solver_.pop();
        impossible_cache_.emplace(key, result);
        return result;
    }

    // General-formula counterpart to is_impossible: true iff `rpn` can never
    // be true given node_idx's own history (same current_top_/enter_node
    // requirement, same push/check/pop pattern). Needed because is_impossible
    // only takes a single fact_id -- fine for branch-admissibility (always a
    // single observed predicate) but not for an arbitrary goal formula.
    //
    // Added specifically to close a real gap: solve_from_node's own goal
    // check (Evaluator::evaluate(goal_rpn, current_state, PESSIMISTIC)) only
    // ever looks at the flat belief bits, with no history-aware fallback --
    // unlike SDRPlanner::check_goal, which already falls back to
    // BeliefState::verify_condition_safely's Z3 regression-entailment for
    // exactly this reason. On localize5, a fact like at(p5-5) can be
    // PROVABLY true given the full sensing history (tree_bmc already proves
    // this correctly for branch-admissibility) while remaining permanently
    // UNKNOWN in the flat state, because record_conditional_provenance's own
    // (correct, sound) ambiguity guard can never abduce it: `checking`'s 19
    // position-conditional effects all target the same four free-* facts, so
    // more than one of those 19 conditions is simultaneously unknown at
    // EVERY apply-time, discarding provenance recording before any
    // observation is even known -- provenance abduction can only ever help
    // once ambiguity is already down to a single candidate, which this
    // domain's signature-elimination reasoning can't produce on its own (only
    // tree_bmc's Z3 encoding, comparing the observed signature against every
    // remaining candidate at once, actually performs that elimination).
    // Confirmed as the reason goal_reached_successes stayed at 0 across a
    // 5-minute localize5-tamer run despite a fully sound, well-guided search:
    // the goal was reachable in principle, but literally unrecognizable by
    // the flat-state-only check that was the only thing solve_from_node ever
    // consulted.
    bool is_rpn_impossible(int node_idx, const std::vector<int>& rpn) {
        if (node_idx != current_top_) {
            throw std::runtime_error("Z3BMCManager::is_rpn_impossible: node_idx is not the current path top -- caller must enter_node() first");
        }
        path_solver_.push();
        path_solver_.add(rpn_to_z3_with(rpn, vars_by_node_.at(node_idx)));
        auto t0 = DebugStats::enabled() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        bool result = path_solver_.check() == z3::unsat;
        if (DebugStats::enabled()) {
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            DebugStats::record_bmc_check(ms);
        }
        path_solver_.pop();
        return result;
    }

    // Extends node_idx's cached encoding by exactly one more, TRANSIENT transition
    // (`action`, resulting in `next_state`) and returns a fully determinized,
    // history-consistent sample, or std::nullopt if the resulting state is
    // impossible. Transient: nothing here gets cached under a node_idx of its own
    // (that only happens via build_child, when the search actually commits to a
    // real child node) -- scoped via a temporary push()/pop() on the already-
    // synced path_solver_, same as is_impossible. node_idx MUST be the current
    // path_solver_ top, same requirement as is_impossible.
    // Like build_child (see its own doc comment), only builds a genuinely fresh
    // scratch variable for facts `action` actually touches -- untouched facts
    // reuse node_idx's own current row directly, so append_transition_to_solver's
    // frame axioms and append_oneofs_to_solver's oneof re-assertion are both
    // skipped for them automatically (same identical-handle mechanism). This
    // matters here specifically because extend_and_sample is called once per
    // rescue candidate (thousands of times per solve on localize5-tamer,
    // confirmed via CPOR_DEBUG_STATS to dominate wall time once the path-solver
    // fix eliminated the O(depth) re-assertion cost elsewhere) -- most rescue
    // candidates are actions that only touch a handful of facts, so this avoids
    // needlessly allocating and asserting ~total_predicates fresh Z3 constants on
    // every single call.
    std::optional<PartiallySpecifiedState> extend_and_sample(int node_idx, const GroundedAction& action,
                                                                const PartiallySpecifiedState& next_state,
                                                                const ProblemDef& problem) {
        if (node_idx != current_top_) {
            throw std::runtime_error("Z3BMCManager::extend_and_sample: node_idx is not the current path top -- caller must enter_node() first");
        }
        const std::vector<z3::expr>& before = vars_by_node_.at(node_idx);
        std::unordered_set<int> touched = compute_touched(action, problem);
        std::vector<z3::expr> row;
        row.reserve(total_predicates_);
        for (int f = 0; f < total_predicates_; ++f) {
            row.push_back(touched.count(f) ? fresh_scratch_var(f) : before[f]);
        }
        path_solver_.push();
        append_transition_to_solver(path_solver_, before, row, action, problem);
        append_oneofs_to_solver(path_solver_, row, problem, &touched);
        for (int f = 0; f < total_predicates_; ++f) {
            if (!next_state.is_unknown(f)) {
                path_solver_.add(next_state.is_true(f) ? row[f] : !row[f]);
            }
        }

        auto t0 = DebugStats::enabled() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        bool is_sat = path_solver_.check() == z3::sat;
        if (DebugStats::enabled()) {
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            DebugStats::record_bmc_check(ms);
        }

        std::optional<PartiallySpecifiedState> result;
        if (is_sat) {
            z3::model model = path_solver_.get_model();
            PartiallySpecifiedState concrete(problem.total_predicates);
            for (int f = 0; f < total_predicates_; ++f) {
                concrete.set_known_value(f, model.eval(row[f], true).is_true());
            }
            result = concrete;
        }
        path_solver_.pop();
        return result;
    }

private:
    z3::context ctx_;
    int total_predicates_;
    std::unordered_map<int, std::vector<z3::expr>> vars_by_node_;       // vars_by_node_[node_idx][fact_id]
    std::unordered_map<int, std::vector<z3::expr>> assertions_by_node_; // cumulative, inherited + own
    std::unordered_map<int64_t, bool> impossible_cache_;                // (node_idx,fact_id,desired_value) -> result

    // Persistent, incrementally-extended solver mirroring the CURRENT recursion
    // path through the search tree (see enter_node/exit_node). path_stack_[i] is
    // the node_idx that pushed path_solver_'s i-th scope; current_top_ is its
    // back() (or -1 when empty).
    z3::solver path_solver_;
    std::vector<int> path_stack_;
    int current_top_ = -1;

    int scratch_counter_ = 0; // disambiguates fresh_var_row(-1) calls from each other

    std::vector<z3::expr> fresh_var_row(int node_idx) {
        std::string tag = (node_idx >= 0) ? ("n" + std::to_string(node_idx))
                                           : ("scratch" + std::to_string(scratch_counter_++));
        std::vector<z3::expr> row;
        row.reserve(total_predicates_);
        for (int f = 0; f < total_predicates_; ++f) {
            std::string name = "bmc_" + tag + "_f" + std::to_string(f);
            row.push_back(ctx_.bool_const(name.c_str()));
        }
        return row;
    }

    z3::expr fresh_var(int node_idx, int fact_id) {
        std::string name = "bmc_n" + std::to_string(node_idx) + "_f" + std::to_string(fact_id);
        return ctx_.bool_const(name.c_str());
    }

    // One-off variable for extend_and_sample's transient scratch row -- never
    // cached under a real node_idx, so just needs a name distinct from every
    // other scratch variable ever created (scratch_counter_ never resets).
    z3::expr fresh_scratch_var(int fact_id) {
        std::string name = "bmc_scratch" + std::to_string(scratch_counter_++) + "_f" + std::to_string(fact_id);
        return ctx_.bool_const(name.c_str());
    }

    // The set of facts `action` actually writes to (guaranteed effects,
    // conditional-effect targets, non-deterministic effects) -- everything else
    // passes through unchanged. Shared between build_child's decision of which
    // facts need a genuinely fresh variable and append_transition's own
    // guaranteed/conditional-effect bookkeeping.
    std::unordered_set<int> compute_touched(const GroundedAction& action, const ProblemDef& problem) {
        std::unordered_set<int> touched;
        for (const auto& eff : action.guaranteed_effects) touched.insert(eff.first);
        for (const auto& ce : action.conditional_effects) {
            for (const auto& eff : ce.effects) touched.insert(eff.first);
        }
        for (int nd_fact : action.non_deterministic_effects) touched.insert(nd_fact);
        return touched;
    }

    // rpn_to_z3, resolving fluent tokens against a specific node's own variable row.
    z3::expr rpn_to_z3_with(const std::vector<int>& rpn, const std::vector<z3::expr>& vars) {
        std::vector<z3::expr> stack;
        stack.reserve(rpn.size());
        for (int token : rpn) {
            if (token >= 0) {
                stack.push_back(vars[token]);
            } else if (token == OP_TRUE) {
                stack.push_back(ctx_.bool_val(true));
            } else if (token == OP_FALSE) {
                stack.push_back(ctx_.bool_val(false));
            } else if (token == OP_NOT) {
                if (stack.empty()) throw std::runtime_error("Z3BMCManager: RPN underflow on OP_NOT");
                z3::expr a = stack.back(); stack.pop_back();
                stack.push_back(!a);
            } else {
                if (stack.size() < 2) throw std::runtime_error("Z3BMCManager: RPN underflow on binary op");
                z3::expr right = stack.back(); stack.pop_back();
                z3::expr left = stack.back(); stack.pop_back();
                if (token == OP_AND) stack.push_back(left && right);
                else if (token == OP_OR) stack.push_back(left || right);
                else if (token == OP_EQUALS) stack.push_back(left == right);
                else throw std::runtime_error("Z3BMCManager: unsupported RPN opcode");
            }
        }
        if (stack.size() != 1) throw std::runtime_error("Z3BMCManager: malformed RPN");
        return stack.back();
    }

    // Mutual exclusion used to be O(group_size^2) pairwise NOT(AND) clauses,
    // re-asserted fresh at EVERY node (each node has its own var row, so this is
    // unavoidable per-node work). For a group of size n that's O(n^2) new
    // assertions per node, and since assertions_by_node_ accumulates across the
    // whole ancestor chain, the cumulative list was O(depth * n^2) -- confirmed via
    // CPOR_DEBUG_STATS's bmc_readd_avg_assertions reaching ~8800 by depth ~20 on
    // localize5's oneof group, which dominated both the make_solver() re-assert
    // loop AND solver.check() itself. Z3's native at-most-1 cardinality constraint
    // compiles the same mutual-exclusion semantics into a single assertion object
    // (internally a sequential/commander encoding, not O(n^2) explicit clauses),
    // cutting per-node oneof cost from O(n^2) to O(1) in the assertions vector.
    // touched_filter, when non-null: skip a group entirely if NONE of its members
    // are touched by this transition. Safe because build_child only reaches here
    // with untouched members already set to the IDENTICAL expr as the parent's
    // row -- the parent's own cached assertions already constrain that exact
    // handle, so re-asserting the same mk_or/atmost over it again would be a
    // no-op, just paid for again on every subsequent node. nullptr (root, and the
    // extend_and_sample scratch path) always asserts every group, since there's
    // no parent encoding to inherit from.
    void append_oneofs(std::vector<z3::expr>& assertions, const std::vector<z3::expr>& vars, const ProblemDef& problem,
                        const std::unordered_set<int>* touched_filter = nullptr) {
        for (const auto& group : problem.oneofs) {
            if (group.empty()) continue;
            if (touched_filter != nullptr) {
                bool any_touched = false;
                for (int fluent_id : group) {
                    if (touched_filter->count(fluent_id)) { any_touched = true; break; }
                }
                if (!any_touched) continue;
            }
            z3::expr_vector group_vars(ctx_);
            for (int fluent_id : group) group_vars.push_back(vars[fluent_id]);
            assertions.push_back(z3::mk_or(group_vars));
            if (group.size() > 1) {
                assertions.push_back(z3::atmost(group_vars, 1));
            }
        }
    }

    void append_oneofs_to_solver(z3::solver& solver, const std::vector<z3::expr>& vars, const ProblemDef& problem,
                                  const std::unordered_set<int>* touched_filter = nullptr) {
        std::vector<z3::expr> tmp;
        append_oneofs(tmp, vars, problem, touched_filter);
        for (const auto& e : tmp) solver.add(e);
    }

    // Builds one transition's worth of assertions (guaranteed/conditional/
    // non-deterministic effects + frame axioms for untouched facts) from `before`
    // vars to `after` vars, appending them to `assertions`. Mirrors
    // RegressionEngine::regress_rpn's own four-case structure exactly, just as Z3
    // implications between two named variable rows instead of RPN substitution.
    //
    // Frame axioms are skipped for any fact where before[f] and after[f] are
    // already the IDENTICAL Z3 handle (before[f].id() == after[f].id()): that's
    // build_child's reuse-parent's-variable case (see its own doc comment) --
    // the equality is trivially true by construction, so asserting it would be
    // pure overhead. The extend_and_sample scratch path always builds a fully
    // fresh `after` row (never reused), so ids never coincide there and every
    // untouched fact still gets its frame axiom exactly as before.
    void append_transition(std::vector<z3::expr>& assertions, const std::vector<z3::expr>& before,
                            const std::vector<z3::expr>& after, const GroundedAction& action,
                            const ProblemDef& problem) {
        std::unordered_set<int> touched;

        for (const auto& eff : action.guaranteed_effects) {
            assertions.push_back(after[eff.first] == ctx_.bool_val(eff.second));
            touched.insert(eff.first);
        }

        std::unordered_map<int, std::vector<const ConditionalEffect*>> ces_by_fact;
        for (const auto& ce : action.conditional_effects) {
            for (const auto& eff : ce.effects) {
                if (touched.count(eff.first)) continue;
                ces_by_fact[eff.first].push_back(&ce);
            }
        }
        for (const auto& [fact_id, ces] : ces_by_fact) {
            touched.insert(fact_id);
            z3::expr_vector any_condition(ctx_);
            for (const ConditionalEffect* ce : ces) {
                bool effect_value = false;
                for (const auto& eff : ce->effects) {
                    if (eff.first == fact_id) { effect_value = eff.second; break; }
                }
                z3::expr condition = rpn_to_z3_with(ce->condition_rpn, before);
                any_condition.push_back(condition);
                assertions.push_back(z3::implies(condition, after[fact_id] == ctx_.bool_val(effect_value)));
            }
            assertions.push_back(z3::implies(!z3::mk_or(any_condition), after[fact_id] == before[fact_id]));
        }

        for (int nd_fact : action.non_deterministic_effects) {
            touched.insert(nd_fact); // left unconstrained, just excluded from frame axioms
        }

        for (int f = 0; f < problem.total_predicates; ++f) {
            if (!touched.count(f) && after[f].id() != before[f].id()) {
                assertions.push_back(after[f] == before[f]);
            }
        }
    }

    void append_transition_to_solver(z3::solver& solver, const std::vector<z3::expr>& before,
                                       const std::vector<z3::expr>& after, const GroundedAction& action,
                                       const ProblemDef& problem) {
        std::vector<z3::expr> tmp;
        append_transition(tmp, before, after, action, problem);
        for (const auto& e : tmp) solver.add(e);
    }
};

} // namespace CPOR
