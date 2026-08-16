#pragma once
#include <vector>
#include "BeliefState.hpp"
#include <unordered_set>
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

    // Tracks states actually visited during execution to prevent physical loops
    std::unordered_set<PartiallySpecifiedState, StateHasher> visited_physical_states;

    int find_loop_breaking_sensing_action(const PartiallySpecifiedState& current_state); 
    bool check_goal(const PartiallySpecifiedState& state) const;
    bool verify_contingent_deadends(const PartiallySpecifiedState& state) const;
    std::vector<int> get_plan_from_solver(const PartiallySpecifiedState& state) const;

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
    static std::vector<int> compute_linear_plan(const BeliefState& current_belief, const ProblemDef& problem, int sample_size,
                                                  int* out_blocking_action_id = nullptr,
                                                  std::vector<int>* out_blocking_fact_tokens = nullptr);

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