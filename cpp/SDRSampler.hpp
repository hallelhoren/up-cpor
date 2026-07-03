#pragma once
#include <vector>
#include "State.hpp"
#include "ProblemData.hpp"

namespace CPOR {

/**
 * @class SDRSampler
 * @brief Handles the 'Sample' phase of SDR using the Z3 SMT Solver.
 * Maps partial dual-mask bitsets and structural constraints into a SAT problem.
 */
class SDRSampler {
public:
    /**
     * @brief Samples mathematically valid, fully concrete world states from a partial belief mask.
     * @param current_belief The Dual-Mask state containing UNKNOWN bits.
     * @param global_problem The problem definition containing OneOf invariants.
     * @param target_sample_count The maximum number of diverse concrete states to generate.
     * @return A vector of PartiallySpecifiedStates where ALL bits are fully resolved (known_mask is completely filled).
     */
    static std::vector<PartiallySpecifiedState> sample_concrete_states(
        const PartiallySpecifiedState& current_belief,
        const ProblemDef& global_problem,
        int target_sample_count
    );
};

} // namespace CPOR