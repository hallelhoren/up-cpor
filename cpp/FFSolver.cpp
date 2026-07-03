#include "FFSolver.hpp"
#include <iostream>
#include <cstdlib>


namespace CPOR {

std::vector<int> FFSolver::search(const PartiallySpecifiedState& concrete_state, const ProblemDef& global_problem) {
    // 1. Memory Safety: Reset FF Global State
    // The original FF planner relies heavily on static memory arenas. 
    // We must clear them so the previous replanning turn does not bleed into this one.
    ff_reset_search_state();
    ff_clear_hash_table();

    // 2. Data-Oriented Memory Mapping
    // Extract the raw C-array pointer from the Dual-Mask bitset. 
    // Because the SDRSampler has already completely resolved the 'known_mask', 
    // the 'value_mask' strictly represents a fully observable 2-valued concrete state.
    const uint64_t* raw_state_ptr = concrete_state.value_mask.data();

    // 3. Execute Native C Heuristic Search
    int plan_length = 0;
    int* raw_c_plan = ff_search(raw_state_ptr, &plan_length);

    // 4. Handle Unsolvable States / Dead-Ends
    if (raw_c_plan == nullptr || plan_length < 0) {
        std::cerr << "[FFSolver] Native FF search hit a dead-end. No valid plan found." << std::endl;
        return {}; // Return empty vector to trigger the caller's failure logic
    }

    // 5. C to C++ Paradigm Translation
    std::vector<int> candidate_plan;
    candidate_plan.reserve(static_cast<size_t>(plan_length));
    
    for (int i = 0; i < plan_length; ++i) {
        candidate_plan.push_back(raw_c_plan[i]);
    }

    // 6. Memory Deallocation
    // Since FF is written in C, it allocated the returned array using malloc().
    // We must free() it here to prevent continuous memory leaks during execution.
    std::free(raw_c_plan);

    return candidate_plan;
}

} // namespace CPOR