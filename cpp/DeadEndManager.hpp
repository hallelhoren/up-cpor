#pragma once

#include <unordered_set>
#include "State.hpp"
#include "ProblemData.hpp"
#include "Evaluator.hpp"

namespace CPOR {

enum class DeadEndStatus {
    SAFE,       // Dead end is definitively FALSE
    FATAL,      // Dead end is definitively TRUE (Trigger Backpropagation)
    MAYBE       // Dead end is UNKNOWN (Requires branching to resolve)
};

class DeadEndManager {
private:
    // Caching system to prevent re-evaluation of known fatal states
    static inline std::unordered_set<PartiallySpecifiedState, StateHasher> dead_end_cache;

public:
    // dead_end_cache is keyed purely on bitmask content (PartiallySpecifiedState's
    // operator==/StateHasher carry no problem identity), so a state cached as
    // FATAL for one problem can collide with an equal-bitmask state from a
    // later, unrelated problem loaded into the same process and be wrongly
    // pruned. Must be called whenever a new problem is loaded.
    static void reset_for_new_problem() {
        dead_end_cache.clear();
    }

    static DeadEndStatus check_dead_ends(const PartiallySpecifiedState& state) {
        // 1. O(1) Cache Lookup
        if (dead_end_cache.find(state) != dead_end_cache.end()) {
            return DeadEndStatus::FATAL;
        }

        bool maybe_dead_end = false;

        // 2. Evaluate all known dead-end expressions
        for (const auto& rpn : get_global_problem().deadend_rpns) {
            uint8_t result = Evaluator::evaluate_rpn_raw(rpn, state);
            
            if (result == VAL_TRUE) {
                dead_end_cache.insert(state); // Cache the fatal result
                return DeadEndStatus::FATAL;
            } else if (result == VAL_UNKNOWN) {
                maybe_dead_end = true; // We can't guarantee safety yet
            }
        }
        
        return maybe_dead_end ? DeadEndStatus::MAYBE : DeadEndStatus::SAFE;
    }
};

} // namespace CPOR