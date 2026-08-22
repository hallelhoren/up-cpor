#include "FFSolver.hpp"
#include "FFBridge.hpp"
#include "BFSSolver.hpp"

namespace CPOR {

// Thin pass-through to FFBridge, which owns the negation ("not-P" shadow
// fact) bookkeeping needed to correctly build the determinized-state bitset
// FF's ff_search() expects. FFBridge::build(problem) must already have been
// called for `global_problem` (native_bridge.cpp's solve_native() does this
// once per problem load, before any search begins).
//
// Falls back to BFSSolver -- the same complete, sound, deduplicated search
// compute_heuristic() already trusts as its own fallback when FF is
// unavailable -- whenever FF itself is unavailable OR returns no plan.
// FF-v2.3 (this project's embedded 1999-2003-era adaptation of Hoffmann's
// planner) was found empirically to report "unsolvable" via both its
// enforced-hill-climbing and best-first fallback on at least one domain
// (the sliding-doors/localize5 example from the CPOR/SDR papers, whose
// `checking` action has ~76 conditional effects on one operator) where a
// valid plan demonstrably exists -- confirmed by directly executing a
// hand-built plan via ActionApplier from the exact same concrete witness FF
// was given. Root-causing that further would mean auditing FF-v2.3's own
// relaxed-planning-graph/EHC/best-first C internals, decades-old code this
// project didn't write; a sound, already-trusted fallback that only ever
// activates when FF has already failed is a safer and more directly
// impactful fix than debugging that C code, and can only ever turn a
// previously-empty result into a found plan, never regress an
// FF-successful case (BFS never runs unless FF's own result was already
// empty).
std::vector<int> FFSolver::search(const PartiallySpecifiedState& concrete_state, const ProblemDef& global_problem) {
    if (FFBridge::is_available()) {
        std::vector<int> plan = FFBridge::search(concrete_state);
        if (!plan.empty()) return plan;
        // FF found nothing (or the witness IS the goal, which FFBridge
        // reports as an empty plan too) -- BFSSolver's own start-state
        // check below correctly turns the latter into an empty plan as
        // well, so falling through here is safe either way.
    }
    return BFSSolver::solve(concrete_state, global_problem, /*max_expansions=*/100000, /*verbose=*/false);
}

} // namespace CPOR