#pragma once
#include <cstdint>
#include <vector>
#include "State.hpp"
#include "ProblemData.hpp"

// Real signatures, matching FF-v2.3/ff_api.h exactly. (A prior version of this
// file declared a different, incompatible ff_search signature -- int*/int/int*
// instead of the real const uint64_t*/int* -- which the linker happily bound
// anyway since C has no signature-based overload resolution, silently
// corrupting every call across the C-ABI boundary.)
extern "C" {
    void ff_reset_search_state();
    void ff_clear_hash_table();
    int* ff_search(const uint64_t* determinized_state_bitset, int* out_plan_length);
}

namespace CPOR {

/**
 * @class FFSolver
 * @brief High-performance C++ wrapper for the native C Fast-Forward (FF) Planner.
 * Converts Data-Oriented bitsets into classical state arrays for the FF engine.
 */
class FFSolver {
public:
    /**
     * @brief Executes the FF Heuristic Search.
     * @param concrete_state A fully determinized witness state (no UNKNOWN bits).
     * @param global_problem The problem structure containing domain definitions.
     * @return A vector of Action IDs representing the path to the goal. Empty if failed.
     */
    static std::vector<int> search(const PartiallySpecifiedState& concrete_state, const ProblemDef& global_problem);
};

} // namespace CPOR