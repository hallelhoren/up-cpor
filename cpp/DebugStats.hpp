#pragma once
#include <cstdlib>
#include <iostream>

// Lightweight, opt-in (CPOR_DEBUG_STATS env var) call counters and Z3 timing,
// added specifically to get hard numbers on where localize5-tamer's wall-clock
// time actually goes -- branch-admissibility checks, rescue-path calls, and the
// underlying Z3 solver calls both feed -- instead of continuing to guess from
// plausible-sounding hypotheses. Near-zero overhead when disabled (one bool
// check per call site); not meant to be a permanent feature.
namespace CPOR {
namespace DebugStats {

inline bool enabled() {
    static bool e = std::getenv("CPOR_DEBUG_STATS") != nullptr;
    return e;
}

inline long branch_admissibility_checks = 0;
inline long rescue_calls = 0;
inline long z3_solver_calls = 0;
inline double z3_solver_total_ms = 0.0;
// get_node_learned_constraints: the incremental per-node cache build (its OWN
// build_belief_state_for_node walk + the single-step derive_learned_constraints
// call), NOT counting time spent in any nested get_node_learned_constraints call for
// an ancestor -- i.e. this isolates each node's own incremental contribution.
inline long node_constraints_builds = 0;
inline double node_constraints_total_ms = 0.0;
// The full per-candidate rescue/branch-check block, wall-clock, INCLUDING any Z3
// calls and regression work inside it -- the number to compare against
// z3_solver_total_ms to see how much of the cost is Z3 itself vs. the regression/
// bookkeeping work around it.
inline long candidate_block_calls = 0;
inline double candidate_block_total_ms = 0.0;
// compute_heuristic's own FF (relaxed-planning-graph) / bounded-BFS search call --
// separate from Z3, since it's a classical search over ONE determinized concrete
// world, not a Z3 solve.
inline long ff_calls = 0;
inline double ff_total_ms = 0.0;
// The per-candidate one-step-delta derive_learned_constraints() call in the
// rescue path (distinct from get_node_learned_constraints's cached per-node
// build) -- fresh work every time, never cached, one candidate query.
inline long new_step_calls = 0;
inline double new_step_total_ms = 0.0;
// The full verify_condition_safely_with_constraints() calls in the branch-
// admissibility check, INCLUDING their own regress_rpn-over-history loop
// (not just the Z3 call nested inside, already counted separately).
inline long branch_verify_calls = 0;
inline double branch_verify_total_ms = 0.0;
// sample_concrete_states' own extra_constraints -> Z3 conversion loop (rpn_to_z3 over
// whatever get_node_learned_constraints handed it) -- distinct from the raw
// solver.check() time already counted in z3_solver_total_ms.
inline long extra_constraints_calls = 0;
inline double extra_constraints_total_ms = 0.0;
// Z3BMCManager::make_solver()'s own re-assert loop (walking assertions_by_node_
// and calling solver.add() for every cumulative assertion) vs. the actual
// solver.check() call -- isolates whether the incremental-caching regression is
// in the O(depth) re-assertion cost or in Z3's own solving cost.
inline long bmc_readd_calls = 0;
inline double bmc_readd_total_ms = 0.0;
inline long bmc_readd_assertions_total = 0;
inline long bmc_check_calls = 0;
inline double bmc_check_total_ms = 0.0;
// Z3BMCManager::evict() call count -- how many closed nodes' cached vars/
// assertions have been freed. Compared against solve_from_node_calls, gives a
// rough sense of how much of the search tree's Z3-side memory is actually
// being reclaimed rather than held forever.
inline long bmc_evictions = 0;
inline void record_bmc_eviction() { if (enabled()) bmc_evictions++; }
// solve_cpor_loop's cheap online-planner iterations vs. how often it gives up
// and falls back to solve_from_node's full exhaustive AND/OR search -- printed
// only at the very end of a solve otherwise (native_bridge.cpp), too late to
// see on a run that gets killed mid-solve. Mirrored here so it shows up in
// every periodic [STATS] flush instead.
inline long fallback_calls = 0;
inline void record_fallback_call() { if (enabled()) fallback_calls++; }
// find_resolving_sensing_action_via_prefix's own outcome breakdown: does its
// bounded "navigate-then-sense" search find a resolver directly (depth 0),
// find one after chaining a few classical actions, or fail -- and when it
// fails, was the search space genuinely exhausted (no more reachable classical
// actions), or was it CUT OFF by MAX_SENSING_PREFIX_DEPTH/MAX_VISITED_STATES
// while there was still more to explore? The latter is the decisive signal:
// it means a resolver might exist just beyond the current bound, and every
// failure here forces the caller into the much more expensive exhaustive
// fallback.
inline long prefix_search_found_direct = 0;   // depth 0, the cheap common case
inline long prefix_search_found_chained = 0;  // found after >=1 classical action
inline long prefix_search_exhausted = 0;      // frontier genuinely emptied, nothing found
inline long prefix_search_cut_off_depth = 0;  // hit MAX_SENSING_PREFIX_DEPTH with frontier still non-empty
inline long prefix_search_cut_off_budget = 0; // hit MAX_VISITED_STATES
inline void record_prefix_search_found_direct() { if (enabled()) prefix_search_found_direct++; }
inline void record_prefix_search_found_chained() { if (enabled()) prefix_search_found_chained++; }
inline void record_prefix_search_exhausted() { if (enabled()) prefix_search_exhausted++; }
inline void record_prefix_search_cut_off_depth() { if (enabled()) prefix_search_cut_off_depth++; }
inline void record_prefix_search_cut_off_budget() { if (enabled()) prefix_search_cut_off_budget++; }
// solve_from_node's own outcome breakdown -- added to diagnose WHY the search
// visits so many nodes before OOM (13.9GB RSS on localize5-tamer even with the
// BMC throughput fix in place) instead of continuing to guess. depth_cap_hits
// in particular matters a lot: per solve_from_node's own comment, a depth-cap
// return is explicitly "inconclusive, not a real failure" and is NEVER cached
// -- so if the true solution needs more than the depth cap allows, or if the
// search keeps re-deriving the same doomed prefix without benefit of caching,
// this count (and how it compares to total solve_from_node calls) should make
// that visible.
inline long solve_from_node_calls = 0;
inline long depth_cap_hits = 0;
// Quality diagnostic for depth-cap hits specifically: is the search genuinely
// closing in on the goal when it runs out of room (h shrinking toward 0), or
// wandering through low-value action chains that never really progress (h
// staying high/flat regardless of depth)? Bucketed by the cheap heuristic's
// own h value AT the depth-cap node itself (score_concrete_sample's scale:
// 0 = goal, 999999 = the cheap FF/BFS search's own bounded horizon sees no
// path at all -- not proof of infeasibility, just "can't see one yet").
inline long depth_cap_h_sum = 0;   // excludes the 999999 bucket, which would swamp any average
inline long depth_cap_h_count = 0; // ditto
inline long depth_cap_h_near_goal = 0;  // h < 10
inline long depth_cap_h_moderate = 0;   // 10 <= h < 100
inline long depth_cap_h_far = 0;        // 100 <= h < 999999
inline long depth_cap_h_no_path_seen = 0; // h == 999999
inline long solved_cache_hits = 0;
inline long failed_cache_hits = 0;
inline long cycle_hits = 0;
inline int max_depth_seen = 0;

inline void record_solve_from_node_call(int depth) {
    if (enabled()) {
        solve_from_node_calls++;
        if (depth > max_depth_seen) max_depth_seen = depth;
    }
}
inline void record_depth_cap_hit() { if (enabled()) depth_cap_hits++; }
inline void record_depth_cap_h(int h) {
    if (!enabled()) return;
    if (h >= 999999) {
        depth_cap_h_no_path_seen++;
        return;
    }
    depth_cap_h_sum += h;
    depth_cap_h_count++;
    if (h < 10) depth_cap_h_near_goal++;
    else if (h < 100) depth_cap_h_moderate++;
    else depth_cap_h_far++;
}
inline void record_solved_cache_hit() { if (enabled()) solved_cache_hits++; }
inline void record_failed_cache_hit() { if (enabled()) failed_cache_hits++; }
inline void record_cycle_hit() { if (enabled()) cycle_hits++; }

// Genuine successes -- distinct from solved_cache_hits (which only counts
// REUSE of an already-cached solution). Added specifically to answer: does
// ANY node ever actually succeed during a run, or is the search failing to
// find success at all (as opposed to succeeding locally but the soundness
// fix denying it a cache entry)?
inline long goal_reached_successes = 0;
inline long or_node_successes = 0;
inline long and_node_successes = 0;
inline void record_goal_reached_success() { if (enabled()) goal_reached_successes++; }
inline void record_or_node_success() { if (enabled()) or_node_successes++; }
inline void record_and_node_success() { if (enabled()) and_node_successes++; }

inline void record_branch_admissibility_check() {
    if (enabled()) branch_admissibility_checks++;
}

inline void record_rescue_call() {
    if (enabled()) rescue_calls++;
}

inline void print_summary(); // fwd decl -- record_z3_call flushes periodically, defined below.

inline void record_z3_call(double ms) {
    if (enabled()) {
        z3_solver_calls++;
        z3_solver_total_ms += ms;
        // Flush every 500 calls so a run killed by an external timeout (the whole
        // reason this instrumentation exists -- localize5-tamer wasn't finishing
        // within 5 minutes even to report a verdict) still leaves real numbers
        // behind instead of nothing at all.
        if (z3_solver_calls % 500 == 0) print_summary();
    }
}

inline void record_node_constraints_build(double ms) {
    if (enabled()) {
        node_constraints_builds++;
        node_constraints_total_ms += ms;
    }
}

inline void record_candidate_block(double ms) {
    if (enabled()) {
        candidate_block_calls++;
        candidate_block_total_ms += ms;
        if (candidate_block_calls % 200 == 0) print_summary();
    }
}

inline void record_ff_call(double ms) {
    if (enabled()) {
        ff_calls++;
        ff_total_ms += ms;
    }
}

inline void record_new_step_call(double ms) {
    if (enabled()) {
        new_step_calls++;
        new_step_total_ms += ms;
    }
}

inline void record_branch_verify_call(double ms) {
    if (enabled()) {
        branch_verify_calls++;
        branch_verify_total_ms += ms;
    }
}

inline void record_extra_constraints_conversion(double ms) {
    if (enabled()) {
        extra_constraints_calls++;
        extra_constraints_total_ms += ms;
    }
}

inline void record_bmc_readd(double ms, long num_assertions) {
    if (enabled()) {
        bmc_readd_calls++;
        bmc_readd_total_ms += ms;
        bmc_readd_assertions_total += num_assertions;
    }
}

inline void record_bmc_check(double ms) {
    if (enabled()) {
        bmc_check_calls++;
        bmc_check_total_ms += ms;
    }
}

inline void print_summary() {
    if (!enabled()) return;
    std::cerr << "[STATS] branch_admissibility_checks=" << branch_admissibility_checks
              << " rescue_calls=" << rescue_calls
              << " z3_solver_calls=" << z3_solver_calls
              << " z3_solver_total_ms=" << z3_solver_total_ms
              << " z3_solver_avg_ms=" << (z3_solver_calls ? z3_solver_total_ms / z3_solver_calls : 0.0)
              << " | node_constraints_builds=" << node_constraints_builds
              << " node_constraints_total_ms=" << node_constraints_total_ms
              << " | candidate_block_calls=" << candidate_block_calls
              << " candidate_block_total_ms=" << candidate_block_total_ms
              << " | ff_calls=" << ff_calls
              << " ff_total_ms=" << ff_total_ms
              << " | new_step_calls=" << new_step_calls
              << " new_step_total_ms=" << new_step_total_ms
              << " | branch_verify_calls=" << branch_verify_calls
              << " branch_verify_total_ms=" << branch_verify_total_ms
              << " | extra_constraints_calls=" << extra_constraints_calls
              << " extra_constraints_total_ms=" << extra_constraints_total_ms
              << " | bmc_readd_calls=" << bmc_readd_calls
              << " bmc_readd_total_ms=" << bmc_readd_total_ms
              << " bmc_readd_assertions_total=" << bmc_readd_assertions_total
              << " bmc_readd_avg_assertions=" << (bmc_readd_calls ? (double)bmc_readd_assertions_total / bmc_readd_calls : 0.0)
              << " | bmc_check_calls=" << bmc_check_calls
              << " bmc_check_total_ms=" << bmc_check_total_ms
              << " | solve_from_node_calls=" << solve_from_node_calls
              << " max_depth_seen=" << max_depth_seen
              << " depth_cap_hits=" << depth_cap_hits
              << " solved_cache_hits=" << solved_cache_hits
              << " failed_cache_hits=" << failed_cache_hits
              << " cycle_hits=" << cycle_hits
              << " goal_reached_successes=" << goal_reached_successes
              << " or_node_successes=" << or_node_successes
              << " and_node_successes=" << and_node_successes
              << " bmc_evictions=" << bmc_evictions
              << " | depth_cap_h_avg=" << (depth_cap_h_count ? (double)depth_cap_h_sum / depth_cap_h_count : 0.0)
              << " depth_cap_h_near_goal=" << depth_cap_h_near_goal
              << " depth_cap_h_moderate=" << depth_cap_h_moderate
              << " depth_cap_h_far=" << depth_cap_h_far
              << " depth_cap_h_no_path_seen=" << depth_cap_h_no_path_seen
              << " | fallback_calls=" << fallback_calls
              << " | prefix_search_found_direct=" << prefix_search_found_direct
              << " prefix_search_found_chained=" << prefix_search_found_chained
              << " prefix_search_exhausted=" << prefix_search_exhausted
              << " prefix_search_cut_off_depth=" << prefix_search_cut_off_depth
              << " prefix_search_cut_off_budget=" << prefix_search_cut_off_budget
              << std::endl;
}

} // namespace DebugStats
} // namespace CPOR
