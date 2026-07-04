#pragma once
#include <vector>
#include "State.hpp"
#include "ProblemData.hpp"

// הצהרה על פונקציות ה-C מספריית FF (מניח שקיימות ב-ff_api.h)
extern "C" {
    void ff_reset_search_state();
    void ff_clear_hash_table();
    // החתימה המעודכנת: מקבלת מערך עובדות אמת, את מספרן, ופוינטר לאורך התוכנית
    int* ff_search(int* true_facts, int num_true_facts, int* out_plan_length);
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