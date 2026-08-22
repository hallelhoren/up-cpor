#include "CPORSolver.hpp"
#include "FFBridge.hpp"
#include "Regression.hpp"
#include "DebugStats.hpp"
#include <cstdlib>
#include <iostream>
#include <chrono>

namespace CPOR {

namespace {
// RAII bracket for Z3BMCManager::enter_node/exit_node: guarantees exactly one
// exit_node() per enter_node(), even when solve_from_node returns early (it has
// many early-return paths -- depth cap, solved_cache/failed_cache hits, cycle
// detection) or when an exception unwinds through it (fallback_to_exhaustive_search
// catches std::bad_alloc from a runaway FF call; the destructor still fires
// correctly during unwinding, keeping path_solver_'s push/pop stack balanced).
// Also owns memory reclamation: once this node's own solve_from_node call
// returns (this scope closes), its Z3BMCManager cache entry is freed --
// see Z3BMCManager::evict's own doc comment for why this is always safe
// (ensure_node_bmc transparently rebuilds a missing entry on any later
// re-entry, from node_pool's own state, at a bounded, deterministic cost).
// This is what actually bounds the search's Z3-side memory to roughly the
// current active path plus whatever's still open, instead of the whole
// search tree ever visited -- confirmed as the fix for doors15-tamer's
// ~14GB cgroup OOM, which was accumulating exactly this: every closed
// subtree's cached assertion vectors, held forever.
//
// A lazy-activation version of this (defer build+enter to the first actual
// tree_bmc query) was tried and reverted: the goal-check below is
// unconditional on every node for every domain, so it activates tree_bmc
// immediately regardless -- lazy activation delivered no measured benefit
// for doors15-tamer (the one domain it was meant to help) while adding real
// complexity, so eager construction stays the simpler, equally-safe choice.
struct BmcNodeScope {
    Z3BMCManager& bmc;
    int node_idx;
    BmcNodeScope(Z3BMCManager& b, int idx, int parent_idx) : bmc(b), node_idx(idx) { bmc.enter_node(node_idx, parent_idx); }
    ~BmcNodeScope() { bmc.exit_node(); bmc.evict(node_idx); }
    BmcNodeScope(const BmcNodeScope&) = delete;
    BmcNodeScope& operator=(const BmcNodeScope&) = delete;
};

// One-shot tracing for the specific question of WHY solve_cpor_loop's cheap
// online loop gave up and reached for fallback_to_exhaustive_search -- gated
// on CPOR_DEBUG_FALLBACK and capped at a handful of prints so a real run
// stays readable instead of flooding with the same pattern hundreds of times.
void trace_fallback_trigger(const char* reason, int node_idx, int depth,
                             const PartiallySpecifiedState& belief, const ProblemDef& problem,
                             int blocking_action_id, const std::vector<int>* blocking_fact_tokens) {
    static int remaining = std::getenv("CPOR_DEBUG_FALLBACK") ? 6 : 0;
    if (remaining <= 0) return;
    --remaining;
    std::cerr << "[DBG-FALLBACK] reason=" << reason << " node_idx=" << node_idx << " depth=" << depth << "\n";
    if (blocking_action_id != -1) {
        const GroundedAction& a = problem.actions[blocking_action_id];
        std::cerr << "  blocking_action_id=" << blocking_action_id
                   << " observe_predicate_id=" << a.observe_predicate_id
                   << " precondition_rpn_size=" << a.precondition_rpn.size() << "\n";
        std::cerr << "  precondition tokens: ";
        for (int t : a.precondition_rpn) {
            if (t >= 0) std::cerr << t << "(" << (belief.is_unknown(t) ? "?" : (belief.is_true(t) ? "T" : "F")) << ") ";
        }
        std::cerr << "\n";
    }
    if (blocking_fact_tokens != nullptr) {
        std::cerr << "  blocking_fact_tokens (" << blocking_fact_tokens->size() << "): ";
        for (int t : *blocking_fact_tokens) std::cerr << t << " ";
        std::cerr << "\n";
    }
    int known_true = 0, known_false = 0, unknown = 0;
    for (int f = 0; f < problem.total_predicates; ++f) {
        if (belief.is_unknown(f)) unknown++;
        else if (belief.is_true(f)) known_true++;
        else known_false++;
    }
    std::cerr << "  belief: known_true=" << known_true << " known_false=" << known_false << " unknown=" << unknown
               << " / total=" << problem.total_predicates << "\n";
}
} // namespace

int CPORSolver::create_root_node(const PartiallySpecifiedState& initial_state) {
    node_pool.emplace_back(initial_state, -1);
    return 0;
}

BeliefState CPORSolver::build_belief_state_for_node(int node_idx) const {
    std::vector<int> chain; // node_pool indices, node_idx first, root last -- reversed below.
    for (int cursor = node_idx; cursor != -1; cursor = node_pool[cursor].parent_idx) {
        chain.push_back(cursor);
    }
    std::reverse(chain.begin(), chain.end()); // now chain.front() is the root, chain.back() is node_idx.

    BeliefState belief(node_pool[chain.front()].state);
    for (size_t i = 1; i < chain.size(); ++i) {
        belief.apply_forward_action(node_pool[chain[i]].action_id, node_pool[chain[i]].state);
    }
    return belief;
}

void CPORSolver::ensure_node_bmc(int node_idx) {
    const ProblemDef& problem = get_global_problem();
    if (!tree_bmc.has_value()) {
        tree_bmc.emplace(problem.total_predicates);
    }
    if (tree_bmc->has(node_idx)) return;

    int parent_idx = node_pool[node_idx].parent_idx;
    if (parent_idx == -1) {
        tree_bmc->build_root(node_idx, node_pool[node_idx].state, problem);
        return;
    }
    ensure_node_bmc(parent_idx); // fills in any not-yet-cached ancestors first
    tree_bmc->build_child(node_idx, parent_idx, problem.actions[node_pool[node_idx].action_id],
                           node_pool[node_idx].state, problem);
}

int CPORSolver::find_cycle_ancestor_position(const PartiallySpecifiedState& state, const std::vector<int>& current_path_indices) const {
    for (size_t p = 0; p < current_path_indices.size(); ++p) {
        if (node_pool[current_path_indices[p]].state == state) {
            return static_cast<int>(p);
        }
    }
    return -2;
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
    // re-pay for a fresh FF/BFS search on a state we've already scored.
    auto cached = heuristic_cache.find(state);
    if (cached != heuristic_cache.end()) {
        return cached->second;
    }

    const ProblemDef& problem = get_global_problem();
    int h = score_concrete_sample(state, problem);
    heuristic_cache[state] = h;
    return h;
}

int CPORSolver::score_sensing_candidate(const PartiallySpecifiedState& state, int observe_fluent) {
    // Mirror expand_sensing_node's own deduction pipeline exactly (see its doc
    // comment): apply_provenance_deductions() lets this genuine observation get
    // abduced backward into whichever single conditional-effect condition it
    // resolves (e.g. narrowing a position fact via a prior `checking`'s
    // now-resolvable provenance), and apply_oneof_deductions() then closes the
    // loop the other way (a narrowed oneof can make another provenance entry
    // resolvable, and vice versa -- hence bracketing oneof on both sides,
    // exactly as expand_sensing_node does). Without this, this scoring function
    // was blind to sensing's real narrowing power: it could only ever see the
    // OBSERVED fact itself change, never any position fact it happens to
    // resolve, so a sense that genuinely collapses a 9-way ambiguity down to 1
    // scored identically to one that resolves nothing at all -- confirmed as
    // the reason goal_reached_successes stayed at 0 across a 5-minute
    // localize5-tamer run despite a fully sound, well-ordered search: the
    // heuristic could never distinguish real disambiguation progress from
    // none, since expand_sensing_node's own (correct) narrowing was invisible
    // to it.
    PartiallySpecifiedState branch_true = state;
    branch_true.set_known_value(observe_fluent, true);
    branch_true.apply_provenance_deductions();
    apply_oneof_deductions(branch_true);
    branch_true.apply_provenance_deductions();
    int h_true = compute_heuristic(branch_true);

    PartiallySpecifiedState branch_false = state;
    branch_false.set_known_value(observe_fluent, false);
    branch_false.apply_provenance_deductions();
    apply_oneof_deductions(branch_false);
    branch_false.apply_provenance_deductions();
    int h_false = compute_heuristic(branch_false);

    // A branch the cheap FF/BFS search can't see a bounded path through isn't
    // proof that branch is truly dead (see score_concrete_sample's own note on
    // BFSSolver's bounded horizon) -- clamp instead of propagating the 999999
    // "definitively excluded" sentinel, so a sensing candidate whose one hard
    // branch just needs more (BMC-driven) work later doesn't get silently
    // dropped from consideration the way a genuinely dead classical candidate
    // would be.
    constexpr int UNRESOLVED_BRANCH_PENALTY = 200;
    if (h_true >= 999999) h_true = UNRESOLVED_BRANCH_PENALTY;
    if (h_false >= 999999) h_false = UNRESOLVED_BRANCH_PENALTY;

    // Combine as MIN(h_true, h_false) + 1, not their SUM. This is a ranking
    // signal compared directly against classical candidates' own h = path.size()
    // (a single-branch count, not an AND-node total) -- summing both branches
    // put sensing on a structurally larger scale than any classical action
    // could ever show, so best-first ordering silently starved it: confirmed on
    // blocks7-tamer (a genuinely contingent domain with real senseON/senseCLEAR/
    // senseONTABLE actions), where branch_admissibility_checks stayed at 0 for
    // 30s straight -- every classical-looking-but-ultimately-futile action chain
    // got tried, to the full depth cap, before sensing was ever once attempted.
    // MIN keeps sensing's reported cost on the same order of magnitude as a
    // single classical step, so it competes fairly instead of being drowned out
    // by its own two-branch bookkeeping.
    return 1 + std::min(h_true, h_false);
}

int CPORSolver::score_concrete_sample(const PartiallySpecifiedState& sample, const ProblemDef& problem) {
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
    auto __dbg_ff_t0 = DebugStats::enabled() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    if (FFBridge::is_available()) {
        path = FFBridge::search(sample);
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
        path = BFSSolver::solve(sample, problem, HEURISTIC_MAX_EXPANSIONS, /*verbose=*/false);
    }
    if (DebugStats::enabled()) {
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - __dbg_ff_t0).count();
        DebugStats::record_ff_call(ms);
    }

    int h;
    if (path.empty()) {
        // If empty, check if it's already the goal. If not, treat it as a dead end
        // within this bounded search horizon.
        h = Evaluator::evaluate(problem.goal_rpn, sample, EvalMode::PESSIMISTIC) ? 0 : 999999;
    } else {
        h = static_cast<int>(path.size());
    }
    return h;
}

// -----------------------------------------------------------------------------
// Core Contingent AND/OR Search Logic
// -----------------------------------------------------------------------------
bool CPORSolver::solve_from_node(int node_idx, std::vector<int>& current_path_indices, int depth, int* out_lowlink) {
    const ProblemDef& problem = get_global_problem();
    PartiallySpecifiedState current_state = node_pool[node_idx].state;

    // Keeps tree_bmc's persistent path solver synced to node_idx for the whole
    // duration of this call (see ensure_node_bmc's own doc comment and
    // Z3BMCManager::enter_node's doc comment for why this eliminates the
    // O(depth) re-assertion that used to dominate every branch-admissibility
    // check and rescue-path query). Bracketing the WHOLE function body (instead
    // of only the AND-node/rescue call sites that actually query it) means the
    // three recursive calls below (child_idx/t_idx/f_idx) always find
    // current_top_ already sitting at their own parent when THEY enter, which is
    // exactly the fast path -- solve_from_node's own call structure is a real
    // DFS, so this mirrors it exactly.
    ensure_node_bmc(node_idx);
    BmcNodeScope __bmc_scope(*tree_bmc, node_idx, node_pool[node_idx].parent_idx);

    if (std::getenv("CPOR_DEBUG_STATE_DUMP") && node_idx == std::atoi(std::getenv("CPOR_DEBUG_STATE_DUMP"))) {
        const ProblemDef& p = get_global_problem();
        std::cerr << "[DBG-STATE] node_idx=" << node_idx << " depth=" << depth << ": ";
        for (int id = 0; id < p.total_predicates; ++id) {
            if (current_state.is_true(id)) std::cerr << id << "=T ";
            else if (current_state.is_false(id)) std::cerr << id << "=F ";
        }
        std::cerr << std::endl;
    }

    // Depth Protection: hitting the bound means the search is inconclusive here,
    // not that the state is proven unsolvable -- must never be cached as a failure.
    // Reported as -1, a position "before the root": it is always less than every
    // real stack position (which start at 0), so it can never satisfy the
    // self-containment check below and permanently blocks caching for every
    // ancestor on this path, exactly like the depth cap always has.
    DebugStats::record_solve_from_node_call(depth);
    if (depth > MAX_SEARCH_DEPTH) {
        DebugStats::record_depth_cap_hit();
        // Diagnostic only: is the search genuinely closing in on the goal when
        // it runs out of room, or wandering through low-value action chains
        // that never really progress? See DebugStats::record_depth_cap_h's own
        // doc comment. Gated on enabled() so this extra compute_heuristic call
        // (cheap, memoized, but still real work) never fires outside an
        // explicit CPOR_DEBUG_STATS diagnostic run.
        if (DebugStats::enabled()) {
            DebugStats::record_depth_cap_h(compute_heuristic(current_state));
        }
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
        DebugStats::record_solved_cache_hit();
        if (out_lowlink) *out_lowlink = NO_CYCLE;
        return true;
    }
    if (failed_cache.find(current_state) != failed_cache.end()) {
        DebugStats::record_failed_cache_hit();
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
    int ancestor_position = find_cycle_ancestor_position(current_state, current_path_indices);
    if (ancestor_position != -2) {
        DebugStats::record_cycle_hit();
        // A cycle only proves this path is unproductive, not that the state is
        // unsolvable in general -- its true status is tied to whether the matched
        // ancestor is itself ultimately resolved (by other, non-cyclic means). Report
        // that dependency upward instead of unconditionally refusing to cache; see the
        // self-containment check after the candidate loop below.
        if (out_lowlink) *out_lowlink = ancestor_position;
        return false;
    }

    // Goal Check.
    //
    // A regression-fallback goal check (build_belief_state_for_node +
    // verify_condition_safely) was tried here once before, using the OLD
    // O(depth)-per-node regression machinery, and reverted -- see git history
    // for that attempt's own reasoning. tree_bmc changes the cost picture
    // entirely: node_idx's path solver is already synced (see
    // ensure_node_bmc/BmcNodeScope above), so this fallback is one more
    // push/check/pop against an already-built encoding, not a fresh O(depth)
    // walk. Needed for real, not speculative: confirmed root cause of
    // goal_reached_successes staying at 0 across a 5-minute localize5-tamer
    // run despite a fully sound, well-guided search -- a fact like at(p5-5)
    // can be PROVABLY true given the sensing history (tree_bmc already proves
    // this correctly for branch-admissibility) while permanently UNKNOWN in
    // the flat belief bits, because record_conditional_provenance's own
    // (sound) ambiguity guard can never abduce it for a domain like this
    // one's shared-signature `checking` action -- see
    // Z3BMCManager::is_rpn_impossible's own doc comment for the full
    // reasoning. This vacuous conclusion is exactly like an AND-node's
    // BMC-vacuous branch (see NO_CYCLE's own doc comment): history-dependent,
    // not derivable from current_state's flat bits alone, so it must use the
    // -1 "never self-contained" sentinel rather than NO_CYCLE, or a
    // DIFFERENT node_idx reaching this identical flat state via a different
    // history could wrongly inherit this node's cached solved_cache entry.
    // Both branches below are a genuine goal LEAF: chosen_action_id stays at
    // its -1 default (extract_node's own contract for "solved, no further
    // action needed" -- see problem_grounder.py). But node_idx can be
    // RE-ENTERED into solve_from_node more than once (e.g. close_node_and_
    // propagate's own "give the exhaustive search a fresh chance from
    // parent_idx" re-invoking fallback_to_exhaustive_search(node_idx) after
    // an earlier attempt already tried and abandoned some AND-node candidate,
    // leaving true_child_idx/false_child_idx pointing at that attempt's own
    // (possibly still-unresolved) t_idx/f_idx) -- the candidate loop below
    // clears these defensively on every iteration for exactly this reason
    // ("Clear any child links a previously-tried-and-abandoned candidate"),
    // but an early return here skips that loop entirely. Without this reset,
    // extract_node can reach a stale, never-resolved child through this
    // node's own leftover true_child_idx/false_child_idx even though THIS
    // node is correctly marked solved -- confirmed as the cause of
    // extract_node's "reached an unsolved node" assertion once the BMC
    // fallback below started actually proving goals true.
    node_pool[node_idx].single_child_idx = -1;
    node_pool[node_idx].true_child_idx = -1;
    node_pool[node_idx].false_child_idx = -1;

    if (Evaluator::evaluate(problem.goal_rpn, current_state, EvalMode::PESSIMISTIC)) {
        DebugStats::record_goal_reached_success();
        node_pool[node_idx].is_solved = true;
        solved_cache[current_state] = node_idx;
        if (out_lowlink) *out_lowlink = NO_CYCLE;
        return true;
    }
    if (tree_bmc->is_rpn_impossible(node_idx, RegressionEngine::negate_rpn(problem.goal_rpn))) {
        DebugStats::record_goal_reached_success();
        node_pool[node_idx].is_solved = true;
        // Vacuous/history-dependent -- see this whole check's doc comment.
        if (out_lowlink) *out_lowlink = -1;
        return true;
    }

    current_path_indices.push_back(node_idx);
    std::vector<ActionCandidate> candidates;

    // Lazy/selective history-aware rescue for candidates the cheap heuristic marks
    // dead. compute_heuristic's cheap path samples a concrete world consistent only
    // with this node's own flat bitset + the problem's static oneof/or constraints --
    // nothing ties an unresolved fact (e.g. a still-unknown `at` position) to other
    // facts ALREADY known via history (e.g. sensor readings), because that link only
    // exists procedurally inside an action's conditional effects, never as a standing
    // Z3 axiom. So the cheap sampler can silently pick a physically-inconsistent
    // witness -- one where a known fact and an unresolved one couldn't actually
    // co-occur under the domain's own dynamics -- and score every candidate as dead
    // from it, even when a real solution exists.
    //
    // Escalating on EVERY candidate would fix this but reintroduces the O(depth)
    // build_belief_state_for_node walk plus a full derive_learned_constraints
    // regression on every single heuristic call -- confirmed experimentally to turn
    // a sub-second (wrong-answer) solve into one that doesn't finish in 10 minutes.
    // Since h is never trusted as ground truth (only used to admit/order candidates;
    // real correctness is enforced by is_action_applicable's sound Evaluator check
    // and, for any committed action, by solve_cpor_loop's belief-based validation),
    // there is nothing to lose by trying the cheap path first everywhere and only
    // paying for the expensive, history-aware resample on the specific candidates it
    // was about to discard -- candidates the cheap path already scores as viable never
    // pay this cost at all.
    //
    // Even lazy-per-candidate wasn't enough on its own via the old regress_rpn-based
    // constraint derivation: re-deriving node_idx's own (pre-candidate) constraints,
    // and each rescued candidate's own new step, meant converting increasingly large
    // RPN formulas to Z3 expressions on every rescue call -- confirmed to reach over a
    // million tokens and >150ms/call by depth ~20 in localize5-tamer. tree_bmc (a
    // CPORSolver member, persistent across the WHOLE solve -- see its own declaration
    // and ensure_node_bmc's doc comment) replaces that whole pipeline: node_idx's
    // history is encoded once, ever, incrementally extending its parent's already-
    // cached encoding rather than rebuilding from the root every time, and each
    // rescued candidate just extends node_idx's own cached encoding by its one action
    // via extend_and_sample -- see Z3BMCManager's own doc comment for the full
    // rationale.

    // Generate Candidates
    for (size_t i = 0; i < problem.actions.size(); ++i) {
        const GroundedAction& action = problem.actions[i];
        if (is_action_applicable(action, current_state)) {
            auto __dbg_cand_t0 = DebugStats::enabled() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            // Lookahead determinization logic
            PartiallySpecifiedState projected_state = current_state;
            ActionApplier::apply_action(action, projected_state);

            // For a classical action (not sensing), projected_state IS exactly
            // the OR-node child's would-be state -- if it trivially cycles back
            // to an already-visited ancestor, the "Attempt Execution" loop below
            // will discard it via the identical check anyway (see
            // find_cycle_ancestor_position's own doc comment), so skip paying
            // for compute_heuristic (and possibly the BMC rescue path) on it
            // here too. Sensing actions are excluded: their actual AND-node
            // children (t_idx/f_idx) come from expand_sensing_node forcing the
            // observed fact definitively, not from this optimistic projection,
            // so this check wouldn't be checking the right state for them.
            if (action.observe_predicate_id == -1 &&
                find_cycle_ancestor_position(projected_state, current_path_indices) != -2) {
                if (DebugStats::enabled()) {
                    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - __dbg_cand_t0).count();
                    DebugStats::record_candidate_block(ms);
                }
                continue;
            }

            int h;
            if (action.observe_predicate_id != -1) {
                // ActionApplier::apply_action is a no-op for a pure sensing
                // action, so projected_state == current_state here -- scoring
                // it via compute_heuristic would show zero improvement no
                // matter how valuable the sense actually is. score_sensing_candidate
                // looks at what each hypothetical observation outcome actually
                // unlocks instead (see its own doc comment).
                h = score_sensing_candidate(current_state, action.observe_predicate_id);
            } else {
                h = compute_heuristic(projected_state);
                if (h == 999999) {
                    DebugStats::record_rescue_call();
                    // Extends node_idx's cached encoding by this ONE candidate's own
                    // action, scoped to its own throwaway solver (node_idx's own cached
                    // encoding is untouched, ready for the next candidate or a future
                    // descendant) -- returns a fully determinized, history-consistent
                    // sample if one exists, or nullopt if the resulting state is
                    // genuinely impossible given history (the same "Mathematically Dead
                    // State" conclusion compute_heuristic's own cheap path reaches, just
                    // soundly re-derived with history taken into account instead of
                    // guessed at from an uninformed sample).
                    auto __dbg_ns_t0 = DebugStats::enabled() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                    auto rescued_sample = tree_bmc->extend_and_sample(node_idx, action, projected_state, problem);
                    if (DebugStats::enabled()) {
                        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - __dbg_ns_t0).count();
                        DebugStats::record_new_step_call(ms);
                    }
                    h = rescued_sample.has_value() ? score_concrete_sample(*rescued_sample, problem) : 999999;
                    if (std::getenv("CPOR_DEBUG_RESCUE")) {
                        std::cerr << "[DBG-RESCUE] node_idx=" << node_idx << " depth=" << depth
                                  << " candidate_action=" << i << " bmc_sample=" << rescued_sample.has_value()
                                  << " rescued_h=" << h << std::endl;
                    }
                }
            }
            if (h < 999999) {
                candidates.push_back({static_cast<int>(i), h});
            }
            if (DebugStats::enabled()) {
                double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - __dbg_cand_t0).count();
                DebugStats::record_candidate_block(ms);
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

            // Cheap pre-filter: skip the node_pool allocation + full recursive
            // solve_from_node call (its own BMC enter_node/exit_node push/pop,
            // heuristic computation, cache lookups) for a candidate whose
            // resulting state trivially cycles back to an already-visited
            // ancestor on this path -- solve_from_node would immediately
            // rediscover exactly this at its own top and return false anyway.
            // See find_cycle_ancestor_position's own doc comment: confirmed via
            // CPOR_DEBUG_STATS this was over half of ALL solve_from_node calls
            // on localize5-tamer (cycle_hits=16704 of 31926), every one paying
            // full recursion cost -- including a node_pool entry that then sat
            // in memory forever -- just to hit this same conclusion.
            int pre_ancestor_position = find_cycle_ancestor_position(next_state, current_path_indices);
            if (pre_ancestor_position != -2) {
                DebugStats::record_cycle_hit();
                node_lowlink = std::min(node_lowlink, pre_ancestor_position);
                continue;
            }

            int child_idx = static_cast<int>(node_pool.size());
            node_pool.emplace_back(next_state, action.id);
            node_pool[child_idx].parent_idx = node_idx;
            node_pool[node_idx].single_child_idx = child_idx;

            int child_lowlink = NO_CYCLE;
            bool child_solved = solve_from_node(child_idx, current_path_indices, depth + 1, &child_lowlink);
            if (std::getenv("CPOR_DEBUG_RESCUE")) {
                std::cerr << "[DBG-ATTEMPT] node_idx=" << node_idx << " depth=" << depth
                          << " OR action=" << action.id << " h=" << candidate.h_score
                          << " child_idx=" << child_idx << " child_solved=" << child_solved << std::endl;
            }
            if (child_solved) {
                DebugStats::record_or_node_success();
                node_pool[node_idx].is_solved = true;
                node_pool[node_idx].chosen_action_id = action.id;
                // Only cache if child_lowlink says this success is genuinely
                // self-contained -- see the AND-node branch below for why this
                // matters (same reasoning, though OR-node children never get a
                // vacuous BMC marking themselves; child_lowlink already reflects
                // whatever taint its OWN recursive resolution picked up, via
                // this same fix applied recursively).
                if (child_lowlink >= depth) {
                    solved_cache[current_state] = node_idx;
                }
                current_path_indices.pop_back();
                if (out_lowlink) *out_lowlink = child_lowlink;
                return true;
            }
            node_lowlink = std::min(node_lowlink, child_lowlink);
        }
        // AND Node (Sensing Action)
        else {
            // Optimization: If we already know the observation, it's just a regular action.
            if (!current_state.is_unknown(action.observe_predicate_id)) continue;

            // Branch admissibility: an AND-node normally requires BOTH the observed-true
            // and observed-false outcomes to independently yield a plan. But the flat
            // bitset alone can't see that one of those outcomes may already be
            // impossible given history -- e.g. in localize5, once free-up and free-down
            // are both known true, no real position also allows free-left true, yet
            // without this check the search still opens that branch and requires it to
            // find its own plan, which of course it never can (confirmed via a direct
            // state dump: a node reachable in the tree had all four of
            // free-up/down/left/right known true simultaneously, a combination no
            // position in the domain supports). A branch Z3-entailment proves impossible
            // given this node's own history can never actually be reached at execution
            // time, so it's vacuously satisfied rather than something requiring its own
            // solution -- exactly the same "provably true, not proven by the flat bits
            // alone" reasoning the goal-check/dead-end-check machinery already relies on
            // elsewhere, just applied before opening a branch instead of after.
            // tree_bmc (see its own declaration, and ensure_node_bmc's doc comment):
            // node_idx's encoding is built at most once, ever, across the whole
            // search tree, incrementally extending its parent's already-cached
            // encoding -- replaces the get_regressed_query/regress_rpn path, whose
            // formula size grew combinatorially with history depth (measured to
            // exceed a million RPN tokens and ~150ms/call by depth ~20), and the
            // first (time-indexed, rebuilt-per-node) BMC version, which stayed
            // linear-sized per query but still paid O(depth) to rebuild from the
            // root for every node with zero sharing across the tree.
            DebugStats::record_branch_admissibility_check();
            auto __dbg_bv_t0 = DebugStats::enabled() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            bool true_impossible = tree_bmc->is_impossible(node_idx, action.observe_predicate_id, true);
            bool false_impossible = tree_bmc->is_impossible(node_idx, action.observe_predicate_id, false);
            if (DebugStats::enabled()) {
                double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - __dbg_bv_t0).count();
                DebugStats::record_branch_verify_call(ms);
            }
            if (true_impossible && false_impossible) {
                // Both outcomes provably impossible -- current_state itself is
                // contradictory given history. Shouldn't normally happen; treat this
                // candidate conservatively as failing rather than asserting anything.
                continue;
            }

            expand_sensing_node(node_idx, action.id, action.observe_predicate_id, current_state);

            int t_idx = node_pool[node_idx].true_child_idx;
            int f_idx = node_pool[node_idx].false_child_idx;

            // BOTH branches must yield a valid plan -- except a provably impossible one,
            // which is marked solved (matching exactly how a genuine goal-node is left:
            // is_solved=true, chosen_action_id/single_child_idx at their -1 defaults)
            // instead of being recursed into, since it can never actually be reached.
            int t_lowlink = NO_CYCLE;
            bool t_solved;
            if (true_impossible) {
                node_pool[t_idx].is_solved = true;
                t_solved = true;
                // This "solved" conclusion is vacuous -- it holds only because
                // tree_bmc->is_impossible proved it given node_idx's own full
                // ANCESTOR HISTORY, not from current_state's flat bits alone (see
                // this whole branch's own doc comment above). solved_cache/
                // failed_cache are keyed on flat state only: a DIFFERENT node_idx
                // reaching this identical flat state via a different history
                // could get a different true_impossible verdict from tree_bmc, so
                // any success or failure that passes through this vacuous branch
                // must never be cached under that key. Reuse the depth-cap's own
                // -1 sentinel (see NO_CYCLE's doc comment): "never self-contained,
                // permanently blocks caching along the whole current path" is
                // exactly the semantics needed here too. Confirmed as the root
                // cause of a false UNSOLVABLE_PROVEN on localize5-tamer once the
                // heuristic fix made the search efficient enough to actually reach
                // this code path: a node whose AND-node candidate failed only
                // because of a sibling's genuine failure (not this vacuous branch)
                // was getting permanently failed_cache'd, even though the vacuous
                // conclusion it depended on doesn't hold for every node sharing
                // that flat state.
                t_lowlink = -1;
            } else {
                t_solved = solve_from_node(t_idx, current_path_indices, depth + 1, &t_lowlink);
            }
            if (std::getenv("CPOR_DEBUG_RESCUE")) {
                std::cerr << "[DBG-ATTEMPT] node_idx=" << node_idx << " depth=" << depth
                          << " AND(sense) action=" << action.id << " h=" << candidate.h_score
                          << " t_idx=" << t_idx << " t_solved=" << t_solved
                          << " true_impossible=" << true_impossible << std::endl;
            }
            if (!t_solved) {
                node_lowlink = std::min(node_lowlink, t_lowlink);
                continue; // Early short-circuit
            }

            int f_lowlink = NO_CYCLE;
            bool f_solved;
            if (false_impossible) {
                node_pool[f_idx].is_solved = true;
                f_solved = true;
                f_lowlink = -1; // vacuous -- see t_lowlink's doc comment above
            } else {
                f_solved = solve_from_node(f_idx, current_path_indices, depth + 1, &f_lowlink);
            }
            if (std::getenv("CPOR_DEBUG_RESCUE")) {
                std::cerr << "[DBG-ATTEMPT] node_idx=" << node_idx << " depth=" << depth
                          << " AND(sense) action=" << action.id << " h=" << candidate.h_score
                          << " f_idx=" << f_idx << " f_solved=" << f_solved
                          << " false_impossible=" << false_impossible << std::endl;
            }

            if (t_solved && f_solved) {
                DebugStats::record_and_node_success();
                node_pool[node_idx].is_solved = true;
                node_pool[node_idx].chosen_action_id = action.id;
                // Both branches' own low-links (real or vacuous) decide whether
                // THIS success is safe to cache -- computed fresh from just this
                // candidate's t_lowlink/f_lowlink, not the shared node_lowlink
                // aggregate (which may carry unrelated taint from previously
                // tried, failed candidates at this same node and would be overly
                // conservative here).
                int candidate_lowlink = std::min(t_lowlink, f_lowlink);
                if (candidate_lowlink >= depth) {
                    solved_cache[current_state] = node_idx;
                }
                current_path_indices.pop_back();
                if (out_lowlink) *out_lowlink = candidate_lowlink;
                return true;
            }
            node_lowlink = std::min({node_lowlink, t_lowlink, f_lowlink});
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

const std::vector<bool>& CPORSolver::get_predicate_can_become_unknown(const ProblemDef& problem) {
    if (!predicate_can_become_unknown_.has_value()) {
        std::vector<bool> can_become_unknown(problem.total_predicates, false);
        for (const auto& action : problem.actions) {
            for (int p : action.non_deterministic_effects) {
                if (p >= 0 && p < problem.total_predicates) can_become_unknown[p] = true;
            }
            // ActionApplier's OTHER path to UNKNOWN, distinct from
            // non_deterministic_effects: a conditional effect whose
            // condition evaluates to VAL_UNKNOWN (not definitively true or
            // false) at apply time degrades every one of ITS effect facts to
            // unknown too (see ActionApplier.hpp's "4. Apply Non-Deterministic
            // Degradation" section and the VAL_UNKNOWN branch just above it).
            // Missing this here was a real bug, not just a missed
            // optimization: wumpus05-tamer (percept-driven conditional
            // effects) regressed to a 30s+ timeout when this prune first
            // shipped without it, because a fact that genuinely CAN still
            // become unknown via such an effect was wrongly declared
            // permanently resolved, short-circuiting
            // find_resolving_sensing_action_via_prefix's BFS away from a
            // resolver that actually existed.
            for (const auto& ce : action.conditional_effects) {
                for (const auto& eff : ce.effects) {
                    if (eff.first >= 0 && eff.first < problem.total_predicates) can_become_unknown[eff.first] = true;
                }
            }
        }
        predicate_can_become_unknown_ = std::move(can_become_unknown);
    }
    return *predicate_can_become_unknown_;
}

int CPORSolver::find_resolving_sensing_action_via_prefix(const std::vector<int>& blocked_precondition_rpn,
                                                           const PartiallySpecifiedState& belief,
                                                           std::vector<int>& out_prefix_actions) {
    // Depth 0: the cheap, common case (doors/wumpus/ebtcs-shaped domains,
    // where the resolving sensing action is already applicable right now).
    int direct = find_resolving_sensing_action(blocked_precondition_rpn, belief);
    if (direct != -1) {
        DebugStats::record_prefix_search_found_direct();
        return direct;
    }

    const ProblemDef& problem = get_global_problem();

    // Upfront prune: chaining through MORE classical actions can only ever
    // turn up a resolver for a token that's currently known if some action
    // could make that token unknown again (see
    // get_predicate_can_become_unknown's doc comment). If every token in
    // blocked_precondition_rpn is already known AND can never become
    // unknown, the full BFS below is guaranteed to explore its entire
    // MAX_SENSING_PREFIX_DEPTH x MAX_VISITED_STATES budget and still find
    // nothing -- confirmed on doors15-tamer, where a permanently-resolved
    // `opened(p)` (never any action's effect at all) sent this search
    // through its whole budget on every single call, turning solve_cpor_loop's
    // "replan instead of falling back" fix (see its own comment) into a
    // 300s+ timeout even after fallback_calls itself dropped to 0.
    bool any_token_still_reachable = false;
    for (int token : blocked_precondition_rpn) {
        if (token < 0) continue;
        if (belief.is_unknown(token)) { any_token_still_reachable = true; break; }
        if (get_predicate_can_become_unknown(problem)[token]) { any_token_still_reachable = true; break; }
    }
    if (!any_token_still_reachable) {
        DebugStats::record_prefix_search_exhausted();
        return -1;
    }

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
                if (visited.size() >= MAX_VISITED_STATES) {
                    DebugStats::record_prefix_search_cut_off_budget();
                    return -1;
                }
                visited.insert(next_belief);

                std::vector<int> next_prefix = node.prefix;
                next_prefix.push_back(action.id);

                int resolver = find_resolving_sensing_action(blocked_precondition_rpn, next_belief);
                if (resolver != -1) {
                    DebugStats::record_prefix_search_found_chained();
                    out_prefix_actions = std::move(next_prefix);
                    return resolver;
                }
                next_frontier.push_back({std::move(next_belief), std::move(next_prefix)});
            }
        }
        frontier = std::move(next_frontier);
    }
    // Distinguishes WHY the search gave up: frontier.empty() means it was
    // genuinely exhausted (no more reachable classical-action states to try --
    // no resolver exists within this search's own model at all), while a
    // still-non-empty frontier means MAX_SENSING_PREFIX_DEPTH cut it off with
    // more left to explore -- a resolver might exist just beyond the bound.
    if (frontier.empty()) {
        DebugStats::record_prefix_search_exhausted();
    } else {
        DebugStats::record_prefix_search_cut_off_depth();
    }
    return -1;
}

bool CPORSolver::fallback_to_exhaustive_search(int node_idx) {
    ++fallback_invocation_count;
    DebugStats::record_fallback_call();
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

    // Safety cap for the "replan instead of falling back" retry below (see
    // the ran_off_end_goal_not_met handler): bounds how many extra
    // compute_linear_plan hops a single node_idx pop may take before giving
    // up and reaching for fallback_to_exhaustive_search after all.
    constexpr int MAX_REPLAN_HOPS = 5;

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
        //
        // A regression-fallback goal check here too (build_belief_state_for_node
        // + verify_condition_safely on the goal formula itself) was tried and
        // reverted: it's a pure add-on cost with no measured benefit --
        // localize5-tamer's stuck nodes turned out to already be genuinely
        // dead (compute_heuristic finds zero valid Z3 models for every
        // candidate action there, true independent of this fallback) -- a
        // different, deeper problem a goal-check fallback was never going to
        // reach. See solve_from_node's own near-identical revert (same root
        // cause) for the fuller rationale. Keep this check cheap.
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
        // action. Threads this node's own inherited guide witness in/out --
        // copied into locals first since compute_linear_plan doesn't touch
        // node_pool itself, but the emplace_back calls later in this loop
        // iteration (creating child nodes) can reallocate it, so nothing
        // below may hold a reference into node_pool[node_idx] across those.
        //
        // wrapped_belief is built with full ancestor history (not just the
        // bare node_belief) so that compute_linear_plan's own internal
        // derive_learned_constraints-informed witness sampling actually has
        // something to work with. This is NOT free -- build_belief_state_for_node
        // walks node_idx's full ancestor chain, an O(depth) cost -- but unlike
        // the goal-check fallback above, dropping it was tried and regressed
        // localize5-tamer's behavior: without real learned-constraint context
        // here, compute_linear_plan's witness sampling loses the very
        // information that keeps the offline tree's dead-end pruning
        // effective, and the runaway node-count explosion this wiring was
        // originally introduced to fix comes back. Keeping it costs real time
        // on other domains (e.g. blocks7-tamer) but is load bearing for
        // offline convergence, so it stays -- paid exactly ONCE per node_idx
        // pop below, then threaded through every replan-retry hop via
        // apply_forward_action (O(1) amortized) rather than rebuilt from
        // scratch each hop. An earlier version of the hop loop below simply
        // requeued cursor_idx through open_stack on each retry, which paid
        // this O(depth) cost again on EVERY hop -- confirmed via
        // CPOR_DEBUG_STATS on doors15-tamer to turn what should have been a
        // cheap retry into an O(depth^2) 300s+ timeout of its own, even after
        // fallback_calls itself had dropped to 0.
        BeliefState wrapped_belief = build_belief_state_for_node(node_idx);
        bool cursor_has_witness = node_pool[node_idx].has_guide_witness;
        PartiallySpecifiedState cursor_witness = node_pool[node_idx].guide_witness;
        int cursor_idx = node_idx;
        PartiallySpecifiedState cursor_belief = node_belief;

        // Bounded replan-retry loop: normally resolves (goal reached, sensing
        // branch queued, or a genuine failure) within its first pass. Loops
        // again, up to MAX_REPLAN_HOPS times, only for the specific case
        // where a partial plan ran to completion without reaching the goal
        // AND sensing genuinely can't help (see the "Sensing genuinely can't
        // help here" comment below) -- mirroring C#'s CPORPlanner, which
        // simply re-invokes classical planning from wherever its stack
        // currently sits instead of reaching for an expensive AND/OR
        // fallback whenever nothing is left to observe.
        for (int hop = 0; ; ++hop) {
        int blocking_action_id = -1;
        std::vector<int> blocking_fact_tokens;
        std::vector<int> partial_plan = SDRPlanner::compute_linear_plan(wrapped_belief, problem, CPOR_LOOP_WITNESS_SAMPLE_COUNT, &blocking_action_id, &blocking_fact_tokens, &cursor_has_witness, &cursor_witness);

        if (partial_plan.empty()) {
            // compute_linear_plan may have truncated to nothing because its
            // very first candidate action was blocked on a fact this belief
            // hasn't resolved yet (blocking_action_id != -1), or because
            // FFSolver::search itself found nothing for the guide witness
            // (blocking_action_id == -1, blocking_fact_tokens covers this
            // case instead) -- not because no plan exists at all. Try
            // sensing before assuming the online loop is stuck.
            if (!try_resolve_via_sensing_branch(cursor_idx, blocking_action_id, cursor_belief, problem, open_stack, &blocking_fact_tokens)) {
                if (std::getenv("CPOR_DEBUG_FALLBACK")) {
                    int d = 0; for (int c = cursor_idx; node_pool[c].parent_idx != -1; c = node_pool[c].parent_idx) ++d;
                    trace_fallback_trigger("empty_plan", cursor_idx, d, cursor_belief, problem, blocking_action_id, &blocking_fact_tokens);
                }
                fallback_to_exhaustive_search(node_idx);
            }
            break;
        }

        // 3. Execute the partial plan action by action, re-validating each
        // action's precondition against the ACTUAL current belief before
        // committing (CPOR Algorithm 2 line 15's soundness check -- here a
        // direct Evaluator call against the fully-materialized belief rather
        // than the C# original's regress-to-root, since this engine already
        // keeps a full PartiallySpecifiedState at every node).
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
                    if (std::getenv("CPOR_DEBUG_FALLBACK")) {
                        int d = 0; for (int c = cursor_idx; node_pool[c].parent_idx != -1; c = node_pool[c].parent_idx) ++d;
                        trace_fallback_trigger("aborted_precondition_failure", cursor_idx, d, cursor_belief, problem, action_id, nullptr);
                    }
                }
                break;
            }

            if (action.observe_predicate_id != -1 && cursor_belief.is_unknown(action.observe_predicate_id)) {
                // Genuine sensing branch: stop executing this partial plan here
                // -- the two children get their own independent OnlinePlan
                // calls on later stack pops.
                expand_sensing_node(cursor_idx, action.id, action.observe_predicate_id, cursor_belief);
                int t_idx = node_pool[cursor_idx].true_child_idx;
                int f_idx = node_pool[cursor_idx].false_child_idx;
                // Only the branch that agrees with what the inherited witness
                // itself already observes may inherit it -- exactly the same
                // guard SDRPlanner::apply_observation uses when a real
                // observation contradicts guide_witness_. The other branch
                // starts fresh (has_guide_witness left false), since the
                // witness's hidden-state hypothesis is now known wrong there.
                if (cursor_has_witness) {
                    if (cursor_witness.is_true(action.observe_predicate_id)) {
                        node_pool[t_idx].has_guide_witness = true;
                        node_pool[t_idx].guide_witness = cursor_witness;
                    } else {
                        node_pool[f_idx].has_guide_witness = true;
                        node_pool[f_idx].guide_witness = cursor_witness;
                    }
                }
                node_pool[cursor_idx].chosen_action_id = action.id;
                open_stack.push_back(f_idx);
                open_stack.push_back(t_idx);
                branched = true;
                break;
            }

            // Classical action (or a sensing action whose value we already
            // know, which behaves like one -- same optimization solve_from_node
            // applies).
            if (cursor_has_witness) {
                ActionApplier::apply_action(action, cursor_witness);
            }
            ActionApplier::apply_action(action, cursor_belief);
            // Keeps wrapped_belief's own history in lockstep with cursor_belief
            // (O(1) amortized) so a later hop in THIS SAME loop never needs to
            // rebuild it via build_belief_state_for_node -- see this loop's
            // header comment.
            wrapped_belief.apply_forward_action(action.id, cursor_belief);
            int child_idx = static_cast<int>(node_pool.size());
            node_pool.emplace_back(cursor_belief, action.id);
            node_pool[child_idx].parent_idx = cursor_idx;
            if (cursor_has_witness) {
                node_pool[child_idx].has_guide_witness = true;
                node_pool[child_idx].guide_witness = cursor_witness;
            }
            node_pool[cursor_idx].single_child_idx = child_idx;
            node_pool[cursor_idx].chosen_action_id = action.id;
            cursor_idx = child_idx;
        }

        if (aborted) {
            fallback_to_exhaustive_search(node_idx);
            break;
        }

        if (branched) {
            // The two new children (or a sensing-resolution branch queued by
            // try_resolve_via_sensing_branch above) are already on open_stack
            // and will eventually resolve node_idx via close_node_and_propagate.
            break;
        }

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
            break;
        }

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
        if (try_resolve_via_sensing_branch(cursor_idx, blocking_action_id, cursor_belief, problem, open_stack, tokens_to_try)) {
            break; // sensing branch queued
        }

        // Sensing genuinely can't help here (tokens_to_try was empty, or
        // nothing observes what it named) -- confirmed via CPOR_DEBUG_FALLBACK
        // tracing on doors15-tamer: every captured trigger of this reason hit
        // the exact same wall, belief.unknown == 0 (all 450 facts already
        // resolved), blocked on move's `opened(?j)` precondition, which is
        // never any action's effect in this domain (see d.pddl -- opened is
        // observe-only) so once known it can never change. That does NOT
        // mean cursor_idx is stuck: a fully-resolved belief with goal still
        // false is a plain classical replanning opportunity (e.g. route
        // around the now-known-closed door), exactly what C#'s stack-based
        // CPORPlanner.Plan()/OfflinePlanning already does for free by just
        // re-invoking classical search at whatever state the stack currently
        // holds -- it never needs an equivalent expensive AND/OR fallback for
        // a state with nothing left to observe. Mirror that here instead of
        // reaching straight for fallback_to_exhaustive_search: loop back
        // (hop) and call compute_linear_plan again fresh against
        // wrapped_belief/cursor_belief, giving it another, better-informed
        // shot at routing around whatever blocked this attempt.
        //
        // Bounded via MAX_REPLAN_HOPS rather than left to bounce forever: a
        // hop that finds a non-empty partial_plan always advances
        // cursor_belief by >=1 real action first (see the
        // `for (int action_id : partial_plan)` loop above), so it can't
        // re-enter this exact branch at an unchanged belief; a hop that
        // finds nothing at all lands in the sibling partial_plan.empty()
        // branch, which resolves (sense or fallback) immediately. Genuine
        // infinite bouncing shouldn't be reachable by either path, but the
        // cap is kept as a cheap guard against corner cases this reasoning
        // missed.
        if (hop < MAX_REPLAN_HOPS) {
            continue;
        }
        if (std::getenv("CPOR_DEBUG_FALLBACK")) {
            int d = 0; for (int c = cursor_idx; node_pool[c].parent_idx != -1; c = node_pool[c].parent_idx) ++d;
            trace_fallback_trigger("ran_off_end_goal_not_met", cursor_idx, d, cursor_belief, problem, blocking_action_id, tokens_to_try);
        }
        fallback_to_exhaustive_search(node_idx);
        break;
        } // hop loop
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