#include "FFSolver.hpp"
#include "FFBridge.hpp"

namespace CPOR {

// Thin pass-through to FFBridge, which owns the negation ("not-P" shadow
// fact) bookkeeping needed to correctly build the determinized-state bitset
// FF's ff_search() expects. FFBridge::build(problem) must already have been
// called for `global_problem` (native_bridge.cpp's solve_native() does this
// once per problem load, before any search begins) -- if it wasn't, or the
// problem wasn't fully representable in FF's STRIPS model, this returns
// empty and the caller should fall back to a sound alternative.
std::vector<int> FFSolver::search(const PartiallySpecifiedState& concrete_state, const ProblemDef& global_problem) {
    (void)global_problem;
    if (!FFBridge::is_available()) return {};
    return FFBridge::search(concrete_state);
}

} // namespace CPOR