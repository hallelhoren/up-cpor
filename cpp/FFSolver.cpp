#include "FFSolver.hpp"
#include <iostream>
#include <cstdlib>


namespace CPOR {

std::vector<int> FFSolver::search(const PartiallySpecifiedState& concrete_state, const ProblemDef& global_problem) {
    ff_reset_search_state();
    ff_clear_hash_table();

    std::vector<int> true_facts;
    true_facts.reserve(global_problem.total_predicates);

    for (int i = 0; i < global_problem.total_predicates; ++i) {
        if (concrete_state.is_true(i)) {
            true_facts.push_back(i);
        }
    }

    int plan_length = 0;
    
    // Note: Added static_cast<int> to avoid compiler warnings since size() returns size_t
    int* raw_c_plan = ff_search(true_facts.data(), static_cast<int>(true_facts.size()), &plan_length);

    if (raw_c_plan == nullptr || plan_length < 0) {
        return {}; 
    }

    std::vector<int> candidate_plan;
    candidate_plan.reserve(static_cast<size_t>(plan_length));
    
    for (int i = 0; i < plan_length; ++i) {
        candidate_plan.push_back(raw_c_plan[i]);
    }

    std::free(raw_c_plan);

    return candidate_plan;
}

} // namespace CPOR