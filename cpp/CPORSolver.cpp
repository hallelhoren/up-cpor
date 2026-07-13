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
    apply_oneof_deductions(true_state);
    
    int t_idx = static_cast<int>(node_pool.size());
    node_pool.emplace_back(true_state, action_id);
    node_pool[t_idx].parent_idx = node_idx;

    // 2. Create FALSE Universe
    PartiallySpecifiedState false_state = current_state;
    false_state.set_known_value(observe_fluent, false);
    apply_oneof_deductions(false_state);
    
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
    } catch (const std::bad_alloc&) {
        // The exhaustive search can combinatorially explode on some domains
        // (FF's relaxed-planning-graph fixpoint computation and/or this
        // search's own node_pool growth have no protection against it -- a
        // known, pre-existing characteristic of solve_from_node/FF, out of
        // scope to fix here). Left uncaught, this would cross the
        // extern "C"/ctypes boundary as an unhandled C++ exception, which
        // the runtime turns into std::terminate() -> abort(), crashing the
        // whole host process. Graceful degradation instead: treat this
        // subtree as a failure -- NOT cached in failed_cache, since running
        // out of memory proves nothing about the belief's true solvability,
        // only that this attempt couldn't finish; a cheaper path reaching
        // the same belief later should get to try again, not be told it's
        // already known unsolvable.
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
                                                 const ProblemDef& problem, std::vector<int>& open_stack) {
    if (blocking_action_id == -1) return false;

    const GroundedAction& blocked_action = problem.actions[blocking_action_id];
    int resolver = find_resolving_sensing_action(blocked_action.precondition_rpn, belief);
    if (resolver == -1) return false;

    const GroundedAction& sense_action = problem.actions[resolver];
    expand_sensing_node(cursor_idx, resolver, sense_action.observe_predicate_id, belief);
    node_pool[cursor_idx].chosen_action_id = resolver;
    open_stack.push_back(node_pool[cursor_idx].false_child_idx);
    open_stack.push_back(node_pool[cursor_idx].true_child_idx);
    return true;
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
            close_node_and_propagate(node_idx);
            continue;
        }
        if (failed_cache.find(node_belief) != failed_cache.end()) {
            node_pool[node_idx].is_failed = true;
            close_node_and_propagate(node_idx);
            continue;
        }

        // 1. Goal check.
        if (Evaluator::evaluate(problem.goal_rpn, node_belief, EvalMode::PESSIMISTIC)) {
            node_pool[node_idx].is_solved = true;
            solved_cache[node_belief] = node_idx;
            close_node_and_propagate(node_idx);
            continue;
        }

        // 2. OnlinePlan: ask SDRPlanner's witness-validated deliberation for a
        // partial plan ending at the goal or at a safe, informative sensing
        // action.
        BeliefState wrapped_belief(node_belief);
        int blocking_action_id = -1;
        std::vector<int> partial_plan = SDRPlanner::compute_linear_plan(wrapped_belief, problem, CPOR_LOOP_WITNESS_SAMPLE_COUNT, &blocking_action_id);

        if (partial_plan.empty()) {
            // compute_linear_plan may have truncated to nothing because its
            // very first candidate action was blocked on a fact this belief
            // hasn't resolved yet (blocking_action_id != -1), not because no
            // plan exists at all -- try sensing that fact before assuming the
            // online loop is stuck.
            if (!try_resolve_via_sensing_branch(node_idx, blocking_action_id, node_belief, problem, open_stack)) {
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
            // goal now holds. Verified rather than assumed: if it doesn't
            // (most commonly because compute_linear_plan truncated the tail
            // of its own candidate_plan on a witness-blocked action --
            // blocking_action_id names it -- rather than the plan ending
            // cleanly), try resolving that blocked action via sensing before
            // falling back and discarding the safe prefix already executed.
            if (Evaluator::evaluate(problem.goal_rpn, cursor_belief, EvalMode::PESSIMISTIC)) {
                node_pool[cursor_idx].is_solved = true;
                solved_cache[cursor_belief] = cursor_idx;
                close_node_and_propagate(cursor_idx);
            } else if (!try_resolve_via_sensing_branch(cursor_idx, blocking_action_id, cursor_belief, problem, open_stack)) {
                fallback_to_exhaustive_search(node_idx);
            }
        }
        // If branched, the two new children are already on open_stack and
        // will eventually resolve node_idx via close_node_and_propagate.
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