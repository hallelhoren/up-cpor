#pragma once
#include <vector>
#include "State.hpp"
#include "ProblemData.hpp"

namespace CPOR {

/**
 * @class FFBridge
 * @brief Translates our grounded DOD problem representation (RPN preconditions,
 * bitset effects) into FF-v2.3's native connectivity graph (gop_conn/gef_conn/
 * gft_conn), and drives ff_search() as a fast, informed heuristic/plan-search
 * in place of a blind BFS.
 *
 * FF's classical STRIPS model only supports conjunctions of positive fact
 * literals as preconditions/effect-conditions -- no disjunction, no negation
 * as a first-class construct. Our RPN formulas are arbitrary AND/OR/NOT/EQUALS
 * trees (quantifier expansion in particular produces wide ANDs, and IFF
 * compiles to EQUALS). build() translates the representable fragment
 * (conjunctions of possibly-negated literals, with negation compiled into
 * dedicated "not-P" shadow facts kept consistent by mirroring every effect
 * that touches P).
 *
 * Genuine disjunction (e.g. a forall-generated conjunction of OR clauses,
 * such as navigate-to's `(forall (?c) (or (not (inside ?o ?c)) (open ?c)))`)
 * is DELIBERATELY treated as an over-approximation rather than a build
 * failure: an unrepresentable conjunct is dropped (treated as vacuously
 * true) instead of poisoning its whole conjunction, and a precondition that
 * ends up entirely unrepresentable becomes an empty (always-true)
 * precondition rather than aborting the ENTIRE problem's translation. An
 * unrepresentable conditional-effect condition is instead treated as never
 * firing, matching the existing Constant(false) case -- see FFBridge.cpp's
 * combine_and and the per-action/per-conditional-effect handling in build()
 * for exactly where each of these decisions is made.
 *
 * This relaxation is sound for how this engine actually uses FFBridge's
 * output: nowhere is it trusted as ground truth. CPORSolver::compute_heuristic
 * only uses it to score candidates that were already independently verified
 * applicable via the real (three-valued, OR-supporting) Evaluator;
 * SDRPlanner::compute_linear_plan treats it as an OnlinePlan proposal that
 * gets witness-validated and then re-checked against the true belief by
 * CPORSolver::solve_cpor_loop before any action is committed to -- exactly
 * the "online planner may be unsound, the outer loop must validate" pattern
 * SDR/CPOR's own literature (e.g. JAIR 45's Section 4.2) already assumes.
 * The cost of this relaxation is a potential drop in suggestion/heuristic
 * quality (FF may occasionally propose something the real belief rejects,
 * one more ordinary case for the existing validation/fallback machinery to
 * catch) -- never a soundness gap. Callers still must check is_available()
 * (only false now for pathological cases: an entirely-false goal, or a
 * malformed RPN) and fall back to the bounded BFS heuristic when it's false.
 */
class FFBridge {
public:
    // Attempts to build FF's connectivity graph for `problem`. Call once per
    // problem load (not once per search -- the graph only depends on problem
    // structure, not on any particular state). Always fully rebuilds,
    // discarding any previous graph, so it's safe to call again for a new
    // problem. Returns is_available()'s new value.
    static bool build(const ProblemDef& problem);

    // Whether the most recent build() succeeded (the whole problem was
    // representable in FF's STRIPS model). search() must not be called
    // otherwise.
    static bool is_available();

    // Runs FF's search (enforced hill-climbing, falling back to best-first)
    // from a fully determinized witness state. Returns the action id sequence
    // to the goal, or empty if FF found none.
    static std::vector<int> search(const PartiallySpecifiedState& concrete_state);

private:
    static bool s_available;
    static int s_total_predicates;
    // Indexed by original predicate id; -1 if that predicate is never negated
    // anywhere in the problem (no shadow fact needed), else the id of its
    // "not-P" shadow fact (>= s_total_predicates).
    static std::vector<int> s_shadow_fact_of;
    static int s_num_ff_facts;
};

} // namespace CPOR
