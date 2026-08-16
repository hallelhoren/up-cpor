#include "CPORSolver.hpp"
#include "FFBridge.hpp"

namespace CPOR {

int CPORSolver::create_root_node(const PartiallySpecifiedState& initial_state) {
    node_pool.emplace_back(initial_state, -1);
    return 0;
}

// -----------------------------------------------------------------------------
// SDR Epistemic Deduction (Deductive Closure)
// -----------------------------------------------------------------------------
void CPORSolver::apply_oneof_deductions(PartiallySpecifiedState& state) {
    const ProblemDef& problem = get_global_problem();
    bool changed = true;
    
    // Loop until fixpoint is reached
    while (changed) {
        changed = false;
        for (const auto& group : problem.oneofs) {
            int true_count = 0;
            int unknown_count = 0;
            int last_unknown = -1;
            
            for (int p : group) {
                if (state.is_true(p)) true_count++;
                else if (state.is_unknown(p)) {
                    unknown_count++;
                    last_unknown = p;
                }
            }
            
            // Rule 1: If exactly one is true, all other unknowns MUST be false
            if (true_count > 0 && unknown_count > 0) {
                for (int p : group) {
                    if (state.is_unknown(p)) {
                        state.set_known_value(p, false);
                        changed = true;
                    }
                }
            } 
            // Rule 2: If all are false except one unknown, the unknown MUST be true
            else if (true_count == 0 && unknown_count == 1) {
                state.set_known_value(last_unknown, true);
                changed = true;
            }
        }
    }
}

// -----------------------------------------------------------------------------
// AND-Node Expansion (Epistemic Branching)
// -----------------------------------------------------------------------------
void CPORSolver::expand_sensing_node(int node_idx, int action_id, int observe_fluent, const PartiallySpecifiedState& current_state) {
    // 1. Create TRUE Universe
    PartiallySpecifiedState true_state = current_state;
    true_state.set_known_value(observe_fluent, true);
    // Scoped Regression Fix: this is a genuine observation, so abduce any
    // conditional-effect conditions this now-known fact can resolve
    // backward (see State.hpp's apply_provenance_deductions), then let
    // that feed the existing OneOf closure and vice versa.
    true_state.apply_provenance_deductions();
    apply_oneof_deductions(true_state);
    true_state.apply_provenance_deductions();

    int t_idx = static_cast<int>(node_pool.size());
    node_pool.emplace_back(true_state, action_id);
    node_pool[t_idx].parent_idx = node_idx;

    // 2. Create FALSE Universe
    PartiallySpecifiedState false_state = current_state;
    false_state.set_known_value(observe_fluent, false);
    false_state.apply_provenance_deductions();
    apply_oneof_deductions(false_state);
    false_state.apply_provenance_deductions();
    
    int f_idx = static_cast<int>(node_pool.size());
    node_pool.emplace_back(false_state, action_id);
    node_pool[f_idx].parent_idx = node_idx;

    // Link back to AND node
    node_pool[node_idx].true_child_idx = t_idx;
    node_pool[node_idx].false_child_idx = f_idx;
}

bool CPORSolver::is_action_applicable(const GroundedAction& action, const PartiallySpecifiedState& state) {
    // PESSIMISTIC mode: If a precondition is UNKNOWN, we mathematically CANNOT apply the action.
    return Evaluator::evaluate(action.precondition_rpn, state, EvalMode::PESSIMISTIC);
}

int CPORSolver::compute_heuristic(const PartiallySpecifiedState& state) {
    // The candidate-generation loop in solve_from_node calls this once per applicable
    // action per node -- memoize by belief state so repeated/overlapping branches don't
    // re-pay for sampling + a bounded BFS on a state we've already scored.
    auto cached = heuristic_cache.find(state);
    if (cached != heuristic_cache.end()) {
        return cached->second;
    }

    const ProblemDef& problem = get_global_problem();

    // Extract exactly ONE determinized reality from the belief state to feed the classical solver
    auto samples = SDRSampler::sample_concrete_states(state, problem, 1);
    if (samples.empty()) {
        heuristic_cache[state] = 999999; // Mathematically Dead State (Zero valid models)
        return 999999;
    }

    // Prefer FF's relaxed-planning-graph heuristic: it's a real informed heuristic
    // (near-linear per call) rather than a full blind search. FFBridge::build() is
    // called once per problem load (native_bridge.cpp's solve_native()) and only
    // reports available when the WHOLE problem -- every action's precondition,
    // every conditional effect's condition, and the goal -- translated soundly into
    // FF's STRIPS model; a partial/lossy translation would silently understate
    // reachability, so it's all-or-nothing. When unavailable (e.g. the domain has a
    // precondition that's a genuine disjunction FF's model can't express), fall back
    // to the bounded BFS, which is slower but always sound for any RPN formula.
    std::vector<int> path;
    if (FFBridge::is_available()) {
        path = FFBridge::search(samples[0]);
    } else {
        // This is a heuristic signal, not a plan-extraction search: bound it hard and
        // mute its logging. An unbounded BFS here (the previous behavior) turns
        // candidate generation -- called once per applicable action per search node --
        // into a second full solver, which is what was driving the 100k-expansion
        // timeouts on larger domains. BFSSolver::solve's other callers (SDRPlanner's
        // actual replanning, the native BFS POC entry point) are unaffected and keep
        // their full default budget.
        // Bumped from the original 2000: several viplan_hh domains (e.g. putting_away_toys)
        // have >=1 container-typed object, which makes navigate-to's `(forall (?c -
        // container) (or (not (inside ?o ?c)) (open ?c)))` precondition a genuine
        // disjunction once grounded -- FFBridge (soundly) refuses to represent any
        // disjunctive formula, so FFBridge::build() reports the whole problem's FF
        // heuristic unavailable and every heuristic call falls back to this bounded BFS.
        // 2000 was too tight for those domains' larger state spaces, causing systematic
        // false dead-end reports (h=999999) that silently excluded valid candidates from
        // ever being tried -- see compute_heuristic's dead-state comment. Pushing this
        // higher (tried 100000) helps some domains (e.g. organizing_file_cabinet) but not
        // enough to be worth the added per-call latency; 20000 is the point that fixed
        // putting_away_toys without materially slowing FF-available domains (doors7/unix9
        // never exercise this path at all, since FFBridge::build() succeeds for both).
        constexpr int HEURISTIC_MAX_EXPANSIONS = 20000;
        path = BFSSolver::solve(samples[0], problem, HEURISTIC_MAX_EXPANSIONS, /*verbose=*/false);
    }

    int h;
    if (path.empty()) {
        // If empty, check if it's already the goal. If not, treat it as a dead end
        // within this bounded search horizon.
        h = Evaluator::evaluate(problem.goal_rpn, samples[0], EvalMode::PESSIMISTIC) ? 0 : 999999;
    } else {
        h = static_cast<int>(path.size());
    }

    heuristic_cache[state] = h;
    return h;
}

// -----------------------------------------------------------------------------
// Core Contingent AND/OR Search Logic
// -----------------------------------------------------------------------------
bool CPORSolver::solve_from_node(int node_idx, std::vector<int>& current_path_indices, int depth, int* out_lowlink) {
    const ProblemDef& problem = get_global_problem();
    PartiallySpecifiedState current_state = node_pool[node_idx].state;

    // Depth Protection: hitting the bound means the search is inconclusive here,
    // not that the state is proven unsolvable -- must never be cached as a failure.
    // Reported as -1, a position "before the root": it is always less than every
    // real stack position (which start at 0), so it can never satisfy the
    // self-containment check below and permanently blocks caching for every
    // ancestor on this path, exactly like the depth cap always has.
    if (depth > 50) {
        if (out_lowlink) *out_lowlink = -1;
        return false;
    }

    // Cache / Loop Verification
    auto solved_it = solved_cache.find(current_state);
    if (solved_it != solved_cache.end()) {
        // This exact belief-state was already solved via a different node_idx
        // reached by a different search path. Mirror that node's solution
        // structure onto this node_idx too -- leaving chosen_action_id/
        // single_child_idx/true_child_idx/false_child_idx at their -1
        // defaults (as before this fix) made this node's action and any
        // downstream branches unextractable: problem_grounder.py's
        // extract_node() treats chosen_action_id == -1 as "no plan here" and
        // silently drops the branch, which is wrong when the branch is
        // reachable and solved, just solved elsewhere. This surfaced
        // concretely as a sensing node missing one of its two observation
        // branches once the earlier stale-AND-pointer bug (see the candidate
        // loop's reset above) was fixed and larger, more structurally
        // symmetric problems (e.g. doors7) started reusing cached solutions
        // across search paths more often than the smaller existing domains
        // ever did.
        const PlanNode& original = node_pool[solved_it->second];
        node_pool[node_idx].is_solved = true;
        node_pool[node_idx].chosen_action_id = original.chosen_action_id;
        node_pool[node_idx].single_child_idx = original.single_child_idx;
        node_pool[node_idx].true_child_idx = original.true_child_idx;
        node_pool[node_idx].false_child_idx = original.false_child_idx;
        // Carry over Plan Graph Compaction relevance too (safe: node_idx's
        // belief is bit-identical to `original`'s, since solved_cache keys
        // on exact PartiallySpecifiedState equality) -- otherwise a parent
        // relying on node_idx as its child would see has_relevance == false
        // and lose an otherwise-valid compaction opportunity. This function
        // itself never CALLS compute_and_register_relevance (Option B's
        // hooks are scoped to solve_cpor_loop/close_node_and_propagate
        // only), so this copy-through is the only way solve_from_node's own
        // cache hits stay compaction-eligible.
        node_pool[node_idx].relevant_known = original.relevant_known;
        node_pool[node_idx].relevant_hidden = original.relevant_hidden;
        node_pool[node_idx].has_relevance = original.has_relevance;
        if (out_lowlink) *out_lowlink = NO_CYCLE;
        return true;
    }
    if (failed_cache.find(current_state) != failed_cache.end()) {
        if (out_lowlink) *out_lowlink = NO_CYCLE;
        return false;
    }
    // Every candidate expansion allocates a brand-new node_pool slot, so node_idx is
    // always unique and can never already be in current_path_indices -- comparing
    // indices here can never detect a cycle. A cycle means the *epistemic state* has
    // recurred along the current path, regardless of which node instance holds it.
    // Record WHICH ancestor position matched (not just whether one did): a Tarjan-style
    // low-link, propagated back through the recursion below, distinguishes a cycle that
    // bottoms out locally (e.g. a reversible action's own trivial round trip) from one
    // that genuinely depends on an outer ancestor still being resolved -- only the
    // latter actually needs to stay uncached.
    int ancestor_position = -2; // sentinel: "no match found"
    for (size_t p = 0; p < current_path_indices.size(); ++p) {
        if (node_pool[current_path_indices[p]].state == current_state) {
            ancestor_position = static_cast<int>(p);
            break;
        }
    }
    if (ancestor_position != -2) {
        // A cycle only proves this path is unproductive, not that the state is
        // unsolvable in general -- its true status is tied to whether the matched
        // ancestor is itself ultimately resolved (by other, non-cyclic means). Report
        // that dependency upward instead of unconditionally refusing to cache; see the
        // self-containment check after the candidate loop below.
        if (out_lowlink) *out_lowlink = ancestor_position;
        return false;
    }

    // Goal Check
    if (Evaluator::evaluate(problem.goal_rpn, current_state, EvalMode::PESSIMISTIC)) {
        node_pool[node_idx].is_solved = true;
        solved_cache[current_state] = node_idx;
        if (out_lowlink) *out_lowlink = NO_CYCLE;
        return true;
    }

    current_path_indices.push_back(node_idx);
    std::vector<ActionCandidate> candidates;

    // Generate Candidates
    for (size_t i = 0; i < problem.actions.size(); ++i) {
        const GroundedAction& action = problem.actions[i];
        if (is_action_applicable(action, current_state)) {

            // Lookahead determinization logic
            PartiallySpecifiedState projected_state = current_state;
            ActionApplier::apply_action(action, projected_state);

            int h = compute_heuristic(projected_state);
            if (h < 999999) {
                candidates.push_back({static_cast<int>(i), h});
            }
        }
    }

    // Sort by Best-First Heuristic
    std::sort(candidates.begin(), candidates.end());
    // Tarjan-style low-link for this node: the minimum ancestor stack-position that
    // any candidate explored below touches, via either a direct cycle or a deeper
    // descendant's own unresolved low-link. Starts at NO_CYCLE (no dependency of its
    // own). If it never drops below `depth` (this node's own stack position) once every
    // candidate has been tried, every cycle touched anywhere in this node's subtree
    // bottoms out at or below this node itself -- this node is the root of its own
    // cyclic component, so its failure conclusion is self-contained and safe to cache,
    // exactly like an ordinary acyclic failure. If it drops below `depth`, resolution
    // genuinely still depends on a proper ancestor above this node and must propagate
    // further up unresolved.
    int node_lowlink = NO_CYCLE;

    // Attempt Execution
    for (const auto& candidate : candidates) {
        const GroundedAction& action = problem.actions[candidate.action_idx];

        // Clear any child links a previously-tried-and-abandoned candidate for
        // this same node left behind. Without this, a sensing candidate that
        // fails after expand_sensing_node() has already written
        // true_child_idx/false_child_idx, followed by a later OR-node
        // candidate that succeeds, leaves the node with chosen_action_id
        // pointing at the successful classical action while true_child_idx/
        // false_child_idx still point at the abandoned sensing attempt's
        // orphaned children -- extract_node (problem_grounder.py) trusts
        // true_child_idx/false_child_idx != -1 to mean "this is a sensing
        // node" and crashes looking for an observed fluent on an action that
        // doesn't have one.
        node_pool[node_idx].single_child_idx = -1;
        node_pool[node_idx].true_child_idx = -1;
        node_pool[node_idx].false_child_idx = -1;

        // OR Node (Classical Action)
        if (action.observe_predicate_id == -1) {
            PartiallySpecifiedState next_state = current_state;
            ActionApplier::apply_action(action, next_state);

            int child_idx = static_cast<int>(node_pool.size());
            node_pool.emplace_back(next_state, action.id);
            node_pool[child_idx].parent_idx = node_idx;
            node_pool[node_idx].single_child_idx = child_idx;

            int child_lowlink = NO_CYCLE;
            if (solve_from_node(child_idx, current_path_indices, depth + 1, &child_lowlink)) {
                node_pool[node_idx].is_solved = true;
                node_pool[node_idx].chosen_action_id = action.id;
                solved_cache[current_state] = node_idx;
                current_path_indices.pop_back();
                if (out_lowlink) *out_lowlink = NO_CYCLE;
                return true;
            }
            node_lowlink = std::min(node_lowlink, child_lowlink);
        }
        // AND Node (Sensing Action)
        else {
            // Optimization: If we already know the observation, it's just a regular action.
            if (!current_state.is_unknown(action.observe_predicate_id)) continue;

            expand_sensing_node(node_idx, action.id, action.observe_predicate_id, current_state);

            int t_idx = node_pool[node_idx].true_child_idx;
            int f_idx = node_pool[node_idx].false_child_idx;

            // BOTH branches must yield a valid plan
            int t_lowlink = NO_CYCLE;
            bool t_solved = solve_from_node(t_idx, current_path_indices, depth + 1, &t_lowlink);
            if (!t_solved) {
                node_lowlink = std::min(node_lowlink, t_lowlink);
                continue; // Early short-circuit
            }

            int f_lowlink = NO_CYCLE;
            bool f_solved = solve_from_node(f_idx, current_path_indices, depth + 1, &f_lowlink);
            node_lowlink = std::min(node_lowlink, f_lowlink);

            if (t_solved && f_solved) {
                node_pool[node_idx].is_solved = true;
                node_pool[node_idx].chosen_action_id = action.id;
                solved_cache[current_state] = node_idx;
                current_path_indices.pop_back();
                if (out_lowlink) *out_lowlink = NO_CYCLE;
                return true;
            }
        }
    }

    current_path_indices.pop_back();

    if (node_lowlink >= depth) {
        // Self-contained: safe to cache permanently -- see the low-link comment above.
        failed_cache.insert(current_state);
        if (out_lowlink) *out_lowlink = NO_CYCLE;
    } else {
        // Still depends on an unresolved ancestor above this node -- propagate upward,
        // must not cache here.
        if (out_lowlink) *out_lowlink = node_lowlink;
    }

    return false;
}

// -----------------------------------------------------------------------------
// CPOR Outer Loop (Algorithm 2 in Maliah, Komarnitski & Shani 2022) + Layered
// Fallback to the exhaustive AND/OR search above -- see solve_cpor_loop's
// declaration in CPORSolver.hpp for the full rationale.
// -----------------------------------------------------------------------------

int CPORSolver::find_resolving_sensing_action(const std::vector<int>& blocked_precondition_rpn, const PartiallySpecifiedState& belief) {
    const ProblemDef& problem = get_global_problem();
    for (int token : blocked_precondition_rpn) {
        if (token < 0 || !belief.is_unknown(token)) continue;
        for (const auto& action : problem.actions) {
            if (action.observe_predicate_id == token &&
                Evaluator::evaluate(action.precondition_rpn, belief, EvalMode::PESSIMISTIC)) {
                return action.id;
            }
        }
    }
    return -1;
}

int CPORSolver::find_resolving_sensing_action_via_prefix(const std::vector<int>& blocked_precondition_rpn,
                                                           const PartiallySpecifiedState& belief,
                                                           std::vector<int>& out_prefix_actions) {
    // Depth 0: the cheap, common case (doors/wumpus/ebtcs-shaped domains,
    // where the resolving sensing action is already applicable right now).
    int direct = find_resolving_sensing_action(blocked_precondition_rpn, belief);
    if (direct != -1) return direct;

    const ProblemDef& problem = get_global_problem();

    struct SearchState {
        PartiallySpecifiedState belief;
        std::vector<int> prefix;
    };

    std::vector<SearchState> frontier;
    frontier.push_back({belief, {}});
    std::unordered_set<PartiallySpecifiedState, StateHasher> visited;
    visited.insert(belief);

    // Total visited-state budget across the whole search, independent of the
    // depth cap: guards against a domain with a very large per-step branching
    // factor still blowing up within MAX_SENSING_PREFIX_DEPTH levels. This is
    // meant to stay cheap and non-exhaustive -- if the budget runs out, the
    // caller falls back to solve_from_node exactly as before this method
    // existed, so running out here never sacrifices correctness, only misses
    // a potential shortcut.
    constexpr size_t MAX_VISITED_STATES = 2000;

    for (int depth = 0; depth < MAX_SENSING_PREFIX_DEPTH && !frontier.empty(); ++depth) {
        std::vector<SearchState> next_frontier;
        for (const auto& node : frontier) {
            for (const auto& action : problem.actions) {
                // Only chain through classical actions: a sensing action here
                // would itself branch on an unknown outcome, which this
                // simple linear-prefix search doesn't attempt to reason
                // about (that's what the AND-node machinery in solve_from_node
                // and solve_cpor_loop's own partial-plan execution already do).
                if (action.observe_predicate_id != -1) continue;
                if (!is_action_applicable(action, node.belief)) continue;

                PartiallySpecifiedState next_belief = node.belief;
                ActionApplier::apply_action(action, next_belief);
                if (visited.count(next_belief)) continue;
                if (visited.size() >= MAX_VISITED_STATES) return -1;
                visited.insert(next_belief);

                std::vector<int> next_prefix = node.prefix;
                next_prefix.push_back(action.id);

                int resolver = find_resolving_sensing_action(blocked_precondition_rpn, next_belief);
                if (resolver != -1) {
                    out_prefix_actions = std::move(next_prefix);
                    return resolver;
                }
                next_frontier.push_back({std::move(next_belief), std::move(next_prefix)});
            }
        }
        frontier = std::move(next_frontier);
    }
    return -1;
}

bool CPORSolver::fallback_to_exhaustive_search(int node_idx) {
    ++fallback_invocation_count;
    std::vector<int> path;
    path.reserve(64);
    // A fresh, empty path means this call's own internal cycle detection is
    // always relative to itself (depth starts at 0), so any failure it proves
    // is self-contained and gets cached exactly as solve_from_node already
    // does for a normal top-level call -- see the Tarjan low-link comment on
    // solve_from_node's declaration.
    bool result;
    try {
        result = solve_from_node(node_idx, path);
    } catch (const std::exception&) {
        // The exhaustive search can combinatorially explode on some domains
        // (FF's relaxed-planning-graph fixpoint computation and/or this
        // search's own node_pool growth have no protection against it -- a
        // known, pre-existing characteristic of solve_from_node/FF, out of
        // scope to fix here), which throws std::bad_alloc. Evaluator/
        // Z3Manager can also throw plain std::runtime_error on a malformed
        // RPN formula reaching them -- a grounding-contract violation that
        // should never happen for a validly-grounded problem, but is a
        // dynamic-boundary risk, not a static one, since RPN arrays cross
        // the ctypes boundary as raw int arrays with no schema check on this
        // side. Left uncaught, either would cross the extern "C"/ctypes
        // boundary as an unhandled C++ exception, which the runtime turns
        // into std::terminate() -> abort(), crashing the whole host process.
        // Graceful degradation instead: treat this subtree as a failure --
        // NOT cached in failed_cache, since neither exception proves the
        // belief itself is unsolvable, only that this attempt couldn't
        // finish; a cheaper path reaching the same belief later should get
        // to try again, not be told it's already known unsolvable.
        result = false;
    }
    if (!result) {
        // Must be set explicitly: solve_from_node's own contract only
        // inserts into failed_cache (a global, state-keyed set) on a
        // self-contained failure -- it never touches the PER-NODE is_failed
        // flag on node_idx itself. close_node_and_propagate's AND-node-parent
        // logic checks is_solved || is_failed to decide whether a child is
        // "done"; without this, a failed fallback silently leaves its parent
        // waiting forever, since neither flag would ever become true.
        node_pool[node_idx].is_failed = true;
    }
    close_node_and_propagate(node_idx);
    return result;
}

void CPORSolver::close_node_and_propagate(int node_idx) {
    int parent_idx = node_pool[node_idx].parent_idx;
    if (parent_idx == -1) return; // Root of this tree-build call; nothing above it.

    bool parent_is_and_node = node_pool[parent_idx].true_child_idx != -1 ||
                               node_pool[parent_idx].false_child_idx != -1;

    if (!parent_is_and_node) {
        // Classical-action (OR) node parent: single_child_idx is always
        // exactly node_idx (populated synchronously, within the same
        // partial-plan execution pass that created it, by solve_cpor_loop).
        // Its resolution mirrors node_idx's directly -- but node_idx itself
        // may only just now be resolving via a LATER, independent stack pop
        // (e.g. a multi-step partial plan that ran several classical actions
        // before ending in a sensing branch further down: the intermediate
        // OR-node parents never get marked solved by anything else, since
        // only the branch's eventual descendants trigger this function).
        // Without this case, is_solved never climbs back up such a chain and
        // the whole tree is silently reported unsolved despite containing a
        // genuinely valid plan.
        if (node_pool[node_idx].is_solved) {
            node_pool[parent_idx].is_solved = true;
            solved_cache[node_pool[parent_idx].state] = parent_idx;
            if (get_global_problem().is_simple) {
                compute_and_register_relevance(parent_idx);
            }
            close_node_and_propagate(parent_idx);
        } else {
            // node_idx (the only child) failed -- give the exhaustive search
            // a fresh chance from parent_idx's own belief, exactly like the
            // "aborted"/"goal not reached" cases in solve_cpor_loop itself.
            fallback_to_exhaustive_search(parent_idx);
        }
        return;
    }

    int true_idx = node_pool[parent_idx].true_child_idx;
    int false_idx = node_pool[parent_idx].false_child_idx;
    bool true_done = true_idx != -1 && (node_pool[true_idx].is_solved || node_pool[true_idx].is_failed);
    bool false_done = false_idx != -1 && (node_pool[false_idx].is_solved || node_pool[false_idx].is_failed);
    if (!true_done || !false_done) return; // Still waiting on the other branch.

    bool true_solved = node_pool[true_idx].is_solved;
    bool false_solved = node_pool[false_idx].is_solved;

    if (true_solved && false_solved) {
        node_pool[parent_idx].is_solved = true;
        solved_cache[node_pool[parent_idx].state] = parent_idx;
        if (get_global_problem().is_simple) {
            compute_and_register_relevance(parent_idx);
        }
        close_node_and_propagate(parent_idx);
    } else {
        // One branch of this committed sensing action is a definitive dead
        // end. CPOR's own algorithm has no recovery for this (its soundness
        // proof assumes deadend-free domains); fall back to the exhaustive
        // solver for the parent's whole subtree instead -- it may still be
        // solvable via a different action choice the single-path OnlinePlan
        // proposal never considered. fallback_to_exhaustive_search re-enters
        // this function for parent_idx once it resolves, continuing the
        // propagation upward as needed.
        fallback_to_exhaustive_search(parent_idx);
    }
}

bool CPORSolver::try_resolve_via_sensing_branch(int cursor_idx, int blocking_action_id, const PartiallySpecifiedState& belief,
                                                 const ProblemDef& problem, std::vector<int>& open_stack,
                                                 const std::vector<int>* blocking_fact_tokens) {
    std::vector<int> prefix_actions;
    int resolver = -1;

    if (blocking_action_id != -1) {
        const GroundedAction& blocked_action = problem.actions[blocking_action_id];
        resolver = find_resolving_sensing_action_via_prefix(blocked_action.precondition_rpn, belief, prefix_actions);
    }
    if (resolver == -1 && blocking_fact_tokens != nullptr && !blocking_fact_tokens->empty()) {
        resolver = find_resolving_sensing_action_via_prefix(*blocking_fact_tokens, belief, prefix_actions);
    }
    if (resolver == -1) return false;

    // Materialize any navigate-then-sense prefix as an ordinary chain of
    // classical (OR) nodes, exactly like solve_cpor_loop's own partial-plan
    // execution does -- empty when the resolver was already applicable
    // directly at `belief` (the pre-existing, common case). node_pool is a
    // std::vector, so each emplace_back below can reallocate and invalidate
    // any reference into it; chain_cursor/child_idx are plain indices and
    // chain_belief is a by-value copy, so nothing here holds a stale
    // reference across that call (see solve_from_node's identical caution).
    int chain_cursor = cursor_idx;
    PartiallySpecifiedState chain_belief = belief;
    for (int action_id : prefix_actions) {
        const GroundedAction& action = problem.actions[action_id];
        ActionApplier::apply_action(action, chain_belief);
        int child_idx = static_cast<int>(node_pool.size());
        node_pool.emplace_back(chain_belief, action.id);
        node_pool[child_idx].parent_idx = chain_cursor;
        node_pool[chain_cursor].single_child_idx = child_idx;
        node_pool[chain_cursor].chosen_action_id = action.id;
        chain_cursor = child_idx;
    }

    const GroundedAction& sense_action = problem.actions[resolver];
    expand_sensing_node(chain_cursor, resolver, sense_action.observe_predicate_id, chain_belief);
    node_pool[chain_cursor].chosen_action_id = resolver;
    open_stack.push_back(node_pool[chain_cursor].false_child_idx);
    open_stack.push_back(node_pool[chain_cursor].true_child_idx);
    return true;
}

// -----------------------------------------------------------------------------
// Plan Graph Compaction (Option B): CPOR TAAS 2022 Algorithm 3/4 port.
// See CPORSolver.hpp's class-level comment on this section for the full
// design rationale (is_simple gating, why O(n,l) is omitted).
// -----------------------------------------------------------------------------

std::pair<bool, std::vector<CPORSolver::RelevantLiteral>> CPORSolver::try_extract_literal_conjunction(const std::vector<int>& rpn) {
    if (rpn.empty()) return {true, {}}; // Evaluator treats an empty RPN as trivially true.

    // Each stack entry is a flat conjunction of literals (possibly empty,
    // meaning "trivially true"). OP_AND merges two entries; anything else
    // that isn't a plain literal or a NOT of a single literal means the
    // formula isn't representable this way -- bail rather than guess.
    std::vector<std::vector<RelevantLiteral>> stack;
    for (int token : rpn) {
        if (token >= 0) {
            stack.push_back({ {token, true} });
        } else if (token == OP_TRUE) {
            stack.push_back({});
        } else if (token == OP_NOT) {
            if (stack.empty()) return {false, {}};
            std::vector<RelevantLiteral> top = std::move(stack.back());
            stack.pop_back();
            if (top.size() != 1) return {false, {}}; // De Morgan'ing a conjunction isn't a flat literal.
            top[0].second = !top[0].second;
            stack.push_back(std::move(top));
        } else if (token == OP_AND) {
            if (stack.size() < 2) return {false, {}};
            std::vector<RelevantLiteral> right = std::move(stack.back()); stack.pop_back();
            std::vector<RelevantLiteral> left = std::move(stack.back()); stack.pop_back();
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

void CPORSolver::compute_and_register_relevance(int node_idx) {
    const ProblemDef& problem = get_global_problem();
    if (!problem.is_simple) return;

    PlanNode& node = node_pool[node_idx];
    if (node.has_relevance || !node.is_solved) return;

    std::vector<RelevantLiteral> combined_known;
    std::vector<int> combined_hidden;

    if (node.chosen_action_id == -1) {
        // Goal leaf: reused whenever a future belief already satisfies the
        // goal formula's own literals.
        auto extracted = try_extract_literal_conjunction(problem.goal_rpn);
        if (!extracted.first) return;
        combined_known = std::move(extracted.second);
    } else {
        const GroundedAction& action = problem.actions[node.chosen_action_id];
        // is_simple guarantees no action anywhere has conditional effects,
        // but stay defensive rather than assume the grounder honored that.
        if (!action.conditional_effects.empty()) return;

        auto pre_extracted = try_extract_literal_conjunction(action.precondition_rpn);
        if (!pre_extracted.first) return;
        combined_known = pre_extracted.second;

        if (node.true_child_idx != -1 || node.false_child_idx != -1) {
            // Sensing (AND) node. expand_sensing_node never applies the
            // action's own guaranteed/non-deterministic effects (it only
            // resolves the observed fluent + deductions), so no effect
            // regression is needed here -- just union both children's
            // externally-required facts (minus the observed fluent itself,
            // which isn't known pre-sensing by definition). Union, not
            // intersection: reusing this WHOLE subtree means committing to
            // BOTH children's plans verbatim, so a candidate belief must
            // directly satisfy everything either branch relies on -- see
            // the class comment for why relying on deduction closure alone
            // here would be unsound.
            int t_idx = node.true_child_idx;
            int f_idx = node.false_child_idx;
            if (t_idx == -1 || f_idx == -1) return;
            const PlanNode& tchild = node_pool[t_idx];
            const PlanNode& fchild = node_pool[f_idx];
            if (!tchild.has_relevance || !fchild.has_relevance) return;

            int observe_fluent = action.observe_predicate_id;
            if (observe_fluent < 0) return;

            for (const auto& lit : tchild.relevant_known) {
                if (lit.first != observe_fluent) combined_known.push_back(lit);
            }
            for (const auto& lit : fchild.relevant_known) {
                if (lit.first != observe_fluent) combined_known.push_back(lit);
            }

            combined_hidden.push_back(observe_fluent);
            for (int fid : tchild.relevant_hidden) if (fid != observe_fluent) combined_hidden.push_back(fid);
            for (int fid : fchild.relevant_hidden) if (fid != observe_fluent) combined_hidden.push_back(fid);
        } else {
            // Classical (OR) node: regress the single child's requirements
            // back through this action's effects.
            int child_idx = node.single_child_idx;
            if (child_idx == -1) return;
            const PlanNode& child = node_pool[child_idx];
            if (!child.has_relevance) return;

            std::unordered_set<int> det_written;
            for (const auto& eff : action.guaranteed_effects) det_written.insert(eff.first);
            std::unordered_set<int> nondet_written;
            for (int fid : action.non_deterministic_effects) nondet_written.insert(fid);

            for (const auto& lit : child.relevant_known) {
                if (det_written.count(lit.first)) continue; // guaranteed by this action regardless of pre-state
                if (nondet_written.count(lit.first)) return; // self-consistency violation; bail defensively
                combined_known.push_back(lit);
            }
            for (int fid : child.relevant_hidden) {
                if (nondet_written.count(fid)) continue; // guaranteed unknown by this action regardless of pre-state
                if (det_written.count(fid)) return; // self-consistency violation; bail defensively
                combined_hidden.push_back(fid);
            }
        }
    }

    // Dedupe combined_known (identical duplicate literals are harmless; a
    // duplicate fact with CONFLICTING required values means the formula can
    // never be satisfied by any real belief -- bail rather than register
    // something vacuous).
    std::vector<RelevantLiteral> deduped_known;
    for (const auto& lit : combined_known) {
        bool conflict = false;
        bool already_present = false;
        for (const auto& existing : deduped_known) {
            if (existing.first == lit.first) {
                already_present = true;
                if (existing.second != lit.second) conflict = true;
                break;
            }
        }
        if (conflict) return;
        if (!already_present) deduped_known.push_back(lit);
    }
    std::vector<int> deduped_hidden;
    for (int fid : combined_hidden) {
        if (std::find(deduped_hidden.begin(), deduped_hidden.end(), fid) == deduped_hidden.end()) {
            deduped_hidden.push_back(fid);
        }
    }
    // A fact can't be simultaneously required known and required hidden.
    for (const auto& lit : deduped_known) {
        if (std::find(deduped_hidden.begin(), deduped_hidden.end(), lit.first) != deduped_hidden.end()) return;
    }

    // Final safety net: K(n)/H(n) must actually hold against node's OWN
    // belief (reflexivity) -- catches any bug in the regression above rather
    // than risk registering an unsound signature. Cheap relative to the
    // search itself, and this function only runs on solved nodes under
    // problem.is_simple.
    for (const auto& lit : deduped_known) {
        bool holds = lit.second ? node.state.is_true(lit.first) : node.state.is_false(lit.first);
        if (!holds) return;
    }
    for (int fid : deduped_hidden) {
        if (!node.state.is_unknown(fid)) return;
    }

    node.relevant_known = std::move(deduped_known);
    node.relevant_hidden = std::move(deduped_hidden);
    node.has_relevance = true;
    closed_registry.push_back(node_idx);
}

int CPORSolver::find_closed_relevance_match(const PartiallySpecifiedState& belief) const {
    for (int idx : closed_registry) {
        const PlanNode& candidate = node_pool[idx];
        bool ok = true;
        for (const auto& lit : candidate.relevant_known) {
            bool holds = lit.second ? belief.is_true(lit.first) : belief.is_false(lit.first);
            if (!holds) { ok = false; break; }
        }
        if (!ok) continue;
        for (int fid : candidate.relevant_hidden) {
            if (!belief.is_unknown(fid)) { ok = false; break; }
        }
        if (ok) return idx;
    }
    return -1;
}

bool CPORSolver::solve_cpor_loop(int root_idx) {
    const ProblemDef& problem = get_global_problem();
    std::vector<int> open_stack;
    open_stack.push_back(root_idx);

    while (!open_stack.empty()) {
        int node_idx = open_stack.back();
        open_stack.pop_back();

        if (node_pool[node_idx].is_solved || node_pool[node_idx].is_failed) {
            continue; // Already resolved, e.g. via close_node_and_propagate.
        }

        // fallback_to_exhaustive_search's own std::bad_alloc guard only covers
        // its direct solve_from_node call -- node_pool's already-allocated
        // capacity never shrinks just because that catch fired, so sustained
        // memory pressure from an earlier fallback can still make a LATER,
        // completely unrelated node_pool.emplace_back below (e.g. materializing
        // a navigate-then-sense prefix chain, or an ordinary classical-action
        // child) throw std::bad_alloc with nothing above it to catch it,
        // crossing the extern "C"/ctypes boundary uncaught and crashing the
        // whole host process exactly like the original, narrower gap this
        // mirrors (see fallback_to_exhaustive_search's own comment). Degrade
        // the same way: treat this node as failed and let the caller's
        // eventual fallback attempt (if any) try again once some of the
        // arena's failed subtree work has been abandoned.
        try {

        // Copy the belief before any node_pool-mutating call below: node_pool
        // is a std::vector, and emplace_back may reallocate, invalidating any
        // reference/pointer into it (see BFSSolver::solve's identical caution).
        const PartiallySpecifiedState node_belief = node_pool[node_idx].state;

        // 0. Closed-node reuse -- identical semantics to solve_from_node's
        // solved_cache/failed_cache lookup, so a belief already resolved by
        // either algorithm short-circuits here too.
        auto solved_it = solved_cache.find(node_belief);
        if (solved_it != solved_cache.end()) {
            const PlanNode& original = node_pool[solved_it->second];
            node_pool[node_idx].is_solved = true;
            node_pool[node_idx].chosen_action_id = original.chosen_action_id;
            node_pool[node_idx].single_child_idx = original.single_child_idx;
            node_pool[node_idx].true_child_idx = original.true_child_idx;
            node_pool[node_idx].false_child_idx = original.false_child_idx;
            // See the identical copy-through in the "0b." block below and in
            // solve_from_node's own cache-hit mirror: without this, a node
            // resolved via this exact-belief cache hit would silently never
            // become compaction-eligible (has_relevance stays false), which
            // then blocks EVERY ancestor relying on it as a child from ever
            // computing its own relevance either -- this was the actual
            // dominant cause of low compaction observed on doors5 (bailed
            // as "child lacks relevance" far more often than any other
            // reason) until this copy-through was added.
            node_pool[node_idx].relevant_known = original.relevant_known;
            node_pool[node_idx].relevant_hidden = original.relevant_hidden;
            node_pool[node_idx].has_relevance = original.has_relevance;
            close_node_and_propagate(node_idx);
            continue;
        }
        if (failed_cache.find(node_belief) != failed_cache.end()) {
            node_pool[node_idx].is_failed = true;
            close_node_and_propagate(node_idx);
            continue;
        }

        // 0b. Plan Graph Compaction (Option B): a previously closed node
        // whose K(n')/H(n') both hold against this belief can be aliased
        // wholesale -- see CPORSolver.hpp's "Plan Graph Compaction" section
        // for the full soundness argument. find_closed_relevance_match is
        // always safe to call (returns -1, a no-op, whenever closed_registry
        // is empty -- e.g. every non-simple problem, since
        // compute_and_register_relevance itself is a no-op there too), but
        // gate on problem.is_simple anyway as an explicit, redundant
        // safeguard against ever compacting a non-simple domain.
        if (problem.is_simple) {
            int relevance_match = find_closed_relevance_match(node_belief);
            if (relevance_match != -1) {
                const PlanNode& original = node_pool[relevance_match];
                node_pool[node_idx].is_solved = true;
                node_pool[node_idx].chosen_action_id = original.chosen_action_id;
                node_pool[node_idx].single_child_idx = original.single_child_idx;
                node_pool[node_idx].true_child_idx = original.true_child_idx;
                node_pool[node_idx].false_child_idx = original.false_child_idx;
                node_pool[node_idx].relevant_known = original.relevant_known;
                node_pool[node_idx].relevant_hidden = original.relevant_hidden;
                node_pool[node_idx].has_relevance = original.has_relevance;
                close_node_and_propagate(node_idx);
                continue;
            }
        }

        // 1. Goal check.
        if (Evaluator::evaluate(problem.goal_rpn, node_belief, EvalMode::PESSIMISTIC)) {
            node_pool[node_idx].is_solved = true;
            solved_cache[node_belief] = node_idx;
            if (problem.is_simple) {
                compute_and_register_relevance(node_idx);
            }
            close_node_and_propagate(node_idx);
            continue;
        }

        // 2. OnlinePlan: ask SDRPlanner's witness-validated deliberation for a
        // partial plan ending at the goal or at a safe, informative sensing
        // action.
        BeliefState wrapped_belief(node_belief);
        int blocking_action_id = -1;
        std::vector<int> blocking_fact_tokens;
        std::vector<int> partial_plan = SDRPlanner::compute_linear_plan(wrapped_belief, problem, CPOR_LOOP_WITNESS_SAMPLE_COUNT, &blocking_action_id, &blocking_fact_tokens);

        if (partial_plan.empty()) {
            // compute_linear_plan may have truncated to nothing because its
            // very first candidate action was blocked on a fact this belief
            // hasn't resolved yet (blocking_action_id != -1), or because
            // FFSolver::search itself found nothing for the guide witness
            // (blocking_action_id == -1, blocking_fact_tokens covers this
            // case instead) -- not because no plan exists at all. Try
            // sensing before assuming the online loop is stuck.
            if (!try_resolve_via_sensing_branch(node_idx, blocking_action_id, node_belief, problem, open_stack, &blocking_fact_tokens)) {
                fallback_to_exhaustive_search(node_idx);
            }
            continue;
        }

        // 3. Execute the partial plan action by action, re-validating each
        // action's precondition against the ACTUAL current belief before
        // committing (CPOR Algorithm 2 line 15's soundness check -- here a
        // direct Evaluator call against the fully-materialized belief rather
        // than the C# original's regress-to-root, since this engine already
        // keeps a full PartiallySpecifiedState at every node).
        int cursor_idx = node_idx;
        PartiallySpecifiedState cursor_belief = node_belief;
        bool branched = false;
        bool aborted = false;

        for (int action_id : partial_plan) {
            const GroundedAction& action = problem.actions[action_id];

            if (!Evaluator::evaluate(action.precondition_rpn, cursor_belief, EvalMode::PESSIMISTIC)) {
                // OnlinePlan's proposal doesn't actually hold against the true
                // belief -- it only looked valid across the sampled witnesses.
                // Try to resolve the gap with a sensing action instead of
                // discarding the whole node.
                if (try_resolve_via_sensing_branch(cursor_idx, action_id, cursor_belief, problem, open_stack)) {
                    branched = true;
                } else {
                    aborted = true;
                }
                break;
            }

            if (action.observe_predicate_id != -1 && cursor_belief.is_unknown(action.observe_predicate_id)) {
                // Genuine sensing branch: stop executing this partial plan here
                // -- the two children get their own independent OnlinePlan
                // calls on later stack pops.
                expand_sensing_node(cursor_idx, action.id, action.observe_predicate_id, cursor_belief);
                node_pool[cursor_idx].chosen_action_id = action.id;
                open_stack.push_back(node_pool[cursor_idx].false_child_idx);
                open_stack.push_back(node_pool[cursor_idx].true_child_idx);
                branched = true;
                break;
            }

            // Classical action (or a sensing action whose value we already
            // know, which behaves like one -- same optimization solve_from_node
            // applies).
            ActionApplier::apply_action(action, cursor_belief);
            int child_idx = static_cast<int>(node_pool.size());
            node_pool.emplace_back(cursor_belief, action.id);
            node_pool[child_idx].parent_idx = cursor_idx;
            node_pool[cursor_idx].single_child_idx = child_idx;
            node_pool[cursor_idx].chosen_action_id = action.id;
            cursor_idx = child_idx;
        }

        if (aborted) {
            fallback_to_exhaustive_search(node_idx);
            continue;
        }

        if (!branched) {
            // Ran off the end of partial_plan without hitting a sensing
            // branch -- per OnlinePlan's own contract this should mean the
            // goal now holds. Verified rather than assumed: if it doesn't,
            // there are three possible reasons, cheapest-to-check first:
            //  1. compute_linear_plan truncated the tail of its own
            //     candidate_plan on a witness-blocked action -- blocking_action_id
            //     names it.
            //  2. FFSolver::search found nothing at all for the guide
            //     witness -- blocking_fact_tokens (from that same, now-stale
            //     top-of-loop compute_linear_plan call) names candidates.
            //  3. Neither of the above: candidate_plan was non-empty and
            //     passed 4A/4C's multi-witness validation cleanly (every
            //     sampled witness already agreed on it), yet still doesn't
            //     establish the actual goal against the REAL belief -- e.g.
            //     the goal references a fact that happened to already hold
            //     in every sampled witness by chance, so no action in the
            //     plan was ever needed (or chosen) to establish it, even
            //     though it's still genuinely unknown here. blocking_action_id
            //     and the stale blocking_fact_tokens both miss this case (it
            //     never truncated anything), so re-scan the goal directly
            //     against cursor_belief -- the actual belief at this point,
            //     not the one compute_linear_plan reasoned about before any
            //     of partial_plan executed.
            if (Evaluator::evaluate(problem.goal_rpn, cursor_belief, EvalMode::PESSIMISTIC)) {
                node_pool[cursor_idx].is_solved = true;
                solved_cache[cursor_belief] = cursor_idx;
                if (problem.is_simple) {
                    compute_and_register_relevance(cursor_idx);
                }
                close_node_and_propagate(cursor_idx);
            } else {
                std::vector<int> goal_blocking_tokens;
                for (int token : problem.goal_rpn) {
                    if (token >= 0 && cursor_belief.is_unknown(token)) {
                        goal_blocking_tokens.push_back(token);
                    }
                }
                if (goal_blocking_tokens.empty()) {
                    // The goal formula itself doesn't mention any currently-unknown
                    // fact -- the block is one step removed (e.g. the chosen
                    // plan's own object bindings only make sense for whichever
                    // witness FF happened to be guided by, so the goal was
                    // technically reachable *for that witness* without ever
                    // resolving some other, non-goal fact this belief still
                    // doesn't know). Fall back to every unresolved fact in the
                    // real belief, exactly like compute_linear_plan's own
                    // candidate_plan.empty() fallback (see its comment).
                    for (int pred_id = 0; pred_id < problem.total_predicates; ++pred_id) {
                        if (cursor_belief.is_unknown(pred_id)) {
                            goal_blocking_tokens.push_back(pred_id);
                        }
                    }
                }
                const std::vector<int>* tokens_to_try = !goal_blocking_tokens.empty() ? &goal_blocking_tokens : &blocking_fact_tokens;
                if (!try_resolve_via_sensing_branch(cursor_idx, blocking_action_id, cursor_belief, problem, open_stack, tokens_to_try)) {
                    fallback_to_exhaustive_search(node_idx);
                }
            }
        }
        // If branched, the two new children are already on open_stack and
        // will eventually resolve node_idx via close_node_and_propagate.
        } catch (const std::exception&) {
            // See fallback_to_exhaustive_search's identical catch: covers
            // both std::bad_alloc (out-of-memory during this iteration's own
            // node_pool growth or FF's fixpoint computation) and a malformed
            // RPN's std::runtime_error from Evaluator/Z3Manager. Not inserted
            // into failed_cache: neither exception proves node_idx's true
            // solvability, only that this attempt couldn't finish.
            node_pool[node_idx].is_failed = true;
            close_node_and_propagate(node_idx);
        }
    }

    return node_pool[root_idx].is_solved;
}

void CPORSolver::print_conditional_plan(int node_idx, int indent) const {
    if (node_idx < 0 || node_idx >= node_pool.size()) return;
    const PlanNode& node = node_pool[node_idx];

    std::string padding(indent * 2, ' ');
    
    if (node.is_solved && node.chosen_action_id != -1) {
        const GroundedAction& action = get_global_problem().actions[node.chosen_action_id];
        
        if (action.observe_predicate_id == -1) {
            std::cout << padding << "ACTION: " << node.chosen_action_id << "\n";
            print_conditional_plan(node.single_child_idx, indent);
        } else {
            std::cout << padding << "SENSE: " << node.chosen_action_id << " (Fluent " << action.observe_predicate_id << ")\n";
            std::cout << padding << "├─ [TRUE Branch]:\n";
            print_conditional_plan(node.true_child_idx, indent + 2);
            std::cout << padding << "└─ [FALSE Branch]:\n";
            print_conditional_plan(node.false_child_idx, indent + 2);
        }
    } else if (node.is_solved) {
        std::cout << padding << "-> [GOAL REACHED]\n";
    } else {
        std::cout << padding << "-> [DEAD END]\n";
    }
}

} // namespace CPOR