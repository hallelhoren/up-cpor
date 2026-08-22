#pragma once
#include <vector>
#include "BeliefState.hpp"
#include <unordered_set>
#include <unordered_map>
#include "ProblemData.hpp"

namespace CPOR {

class SDRPlanner {
private:
    // Core state and environment references
    BeliefState belief;
    const ProblemDef& global_problem;
    
    // The Execution Cache (Replaces C# FutureActions and NextActionIndex)
    std::vector<int> plan_queue;
    size_t next_action_index;
    
    // Safety Locks (Replaces C# ExpectingObservation)
    bool expecting_observation;
    int pending_sensing_action_id;

    // Tracks states actually visited during execution to prevent physical
    // loops. Maps each visited bitset to how many facts derive_learned_constraints()
    // could derive about the initial state at the time of that visit (see
    // get_next_action's cycle-detection step): the flat bitset alone is
    // blind to purely-regression-derivable knowledge (e.g. a hidden position
    // narrowed, but never resolved to a single value, by a localization
    // domain's per-room conditions), so two visits with an IDENTICAL bitset
    // can still represent genuine epistemic progress the bitset can't show.
    // Comparing this count lets the cycle check tell "truly stuck" apart
    // from "same bitset, but we now know strictly more than we did last
    // time we were here."
    std::unordered_map<PartiallySpecifiedState, size_t, StateHasher> visited_physical_states;

    // Witness continuity for the classical sub-planner (see get_plan_from_solver's
    // own comment for the bug this fixes): a single fully-concrete guess at the
    // hidden state, sampled once and then progressed forward via ActionApplier
    // in lockstep with every action actually dispatched, rather than being
    // re-sampled fresh from the belief on every get_plan_from_solver call.
    bool has_guide_witness_{false};
    PartiallySpecifiedState guide_witness_;

    int find_loop_breaking_sensing_action(const PartiallySpecifiedState& current_state);
    bool check_goal(const PartiallySpecifiedState& state) const;
    bool verify_contingent_deadends(const PartiallySpecifiedState& state) const;
    std::vector<int> get_plan_from_solver(const PartiallySpecifiedState& state);

    bool handle_contingent_deadends(const PartiallySpecifiedState& state);
    int plan_to_observe_deadend(const PartiallySpecifiedState& current_state, const std::vector<int>& target_fluents);

public:
    // Constructor
    SDRPlanner(const PartiallySpecifiedState& initial_state, const ProblemDef& problem);

    // OnlinePlan subroutine (Maliah, Komarnitski & Shani, "Computing Contingent
    // Plan Graphs using Online Planning", TAAS 2022, Section 4): samples
    // witness states from `current_belief` via Z3, asks FFSolver for a
    // candidate classical plan from the primary witness, then truncates it
    // before the first action whose precondition or observation isn't
    // consistent across every sampled witness. Returns a partial plan ending
    // at the goal or at a safe, informative sensing action -- or empty if no
    // plan could be proposed. Stateless (touches only its parameters), so it's
    // usable directly by CPORSolver's offline tree-building loop without
    // constructing a full SDRPlanner instance.
    //
    // out_blocking_action_id (if non-null) is set to the id of the action
    // that caused truncation -- i.e. one whose precondition (step 4A) or
    // post-condition dead-end check (step 4C) failed for at least one sampled
    // witness -- or left at -1 if no truncation happened for that reason
    // (empty witness set, FF found nothing at all, or the plan ended cleanly
    // at the goal or a kept, divergence-causing sensing action). Callers can
    // use this to attempt a sensing resolution instead of assuming no plan
    // was possible: an action truncated here isn't necessarily inapplicable,
    // it may just be blocked on a fact this belief hasn't resolved yet.
    //
    // out_blocking_fact_tokens (if non-null) covers the gap out_blocking_action_id
    // can't: when FFSolver::search itself returns nothing for the primary
    // guide witness (step 3), there's no specific action to blame, so
    // out_blocking_action_id stays at -1 -- previously leaving the caller
    // with no signal at all and forcing a full exhaustive-search fallback
    // even when the real issue is a single still-unresolved fact a sensing
    // action could clear up (e.g. a domain where establishing a package's
    // own location is a prerequisite for reasoning about it at all, but
    // isn't itself part of the goal formula, so it wouldn't be found by
    // scanning the goal alone). Filled with candidate predicate-id tokens to
    // try resolving via sensing, cleared/left empty if nothing looks
    // promising: first the tokens of the goal formula that are still unknown
    // in the real belief (the cheap, common case -- mirrors the reasoning
    // that already existed for 4A/4C's blocking_action_id), then, only if
    // that comes up empty, every predicate id the belief hasn't resolved yet
    // at all. The guide witness concretized all of them via sampling, so FF
    // had no *reason* to fail on that fully-determined copy of the problem
    // unless it's genuinely unreachable there -- an unresolved fact
    // elsewhere in the real belief is the next most likely explanation, and
    // the only one this planner has enough information to act on.
    //
    // inout_has_guide_witness / inout_guide_witness (both optional, both
    // null by default): the same cross-call witness-CONTINUITY fix already
    // applied to the instance path's get_plan_from_solver (see that method's
    // comment for the exact bug this closes -- independently re-sampling a
    // witness on every call lets position-dependent facts a domain never
    // records in its belief, like localize5's free-*, drift out of sync with
    // what an earlier witness already committed to, producing a witness
    // that's a genuine dead end even though the real history it's supposed
    // to model is solvable). CPORSolver::solve_cpor_loop is the caller that
    // actually needs this: it invokes compute_linear_plan once per search
    // node rather than once per instance, so without a way to thread a
    // witness parent-to-child across those calls, every node re-samples
    // independently, exactly reproducing the online bug at tree-building
    // scale (and blowing up into the exhaustive fallback or hanging outright
    // on domains like localize5). When *inout_has_guide_witness is true on
    // entry, *inout_guide_witness is used as the fixed primary guide state
    // instead of sampling a fresh one (a supplementary diverse batch is
    // still sampled for 4A/4C/4E's cross-witness validation, so this doesn't
    // weaken that check). If FF finds nothing at all from that witness -- a
    // genuine dead end, not a validation truncation -- it's discarded
    // (*inout_has_guide_witness set false) and a fresh diverse batch is
    // tried instead, mirroring get_plan_from_solver's own recovery. On any
    // successful return, *inout_guide_witness/*inout_has_guide_witness are
    // updated to the (possibly newly-adopted) primary guide state, already
    // progressed through every action in the returned plan via
    // ActionApplier -- ready for the caller to hand to whichever child
    // node/call comes next in that same history. Left entirely alone
    // (including on failure) when either pointer is null, so callers that
    // don't pass them see the previous, no-continuity behavior unchanged.
    static std::vector<int> compute_linear_plan(const BeliefState& current_belief, const ProblemDef& problem, int sample_size,
                                                  int* out_blocking_action_id = nullptr,
                                                  std::vector<int>* out_blocking_fact_tokens = nullptr,
                                                  bool* inout_has_guide_witness = nullptr,
                                                  PartiallySpecifiedState* inout_guide_witness = nullptr);

    // Phase 1: Deliberation and Execution Dispatch
    int get_next_action();

    // Phase 2: Environment Ingestion and Epistemic Collapse
    bool apply_observation(bool observation_value);

    bool execute_full_simulation();

    // Test hook: forces the internal plan queue for whitebox testing
    // (see tests/test_sdr_full_lifecycle.cpp).
    void set_plan_queue(const std::vector<int>& forced_plan) {
        plan_queue = forced_plan;
        next_action_index = 0; 
    }

    // Allows the execution environment to update the agent's history
    BeliefState& get_mutable_belief() {
        return belief;
    }
};

} // namespace CPOR