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
     * @param extra_constraints Optional additional RPN formulas asserted alongside
     * `current_belief`'s own known facts and the problem's oneof invariants --
     * see BeliefState::derive_learned_constraints. Lets a caller with a full
     * history (unlike this flat belief snapshot) make sampled witnesses
     * respect everything that history's own observations already imply,
     * even for facts the belief's bitset never resolved directly (e.g. a
     * hidden position only ever observed through its side effects).
     * Empty by default, so every existing caller's behavior is unchanged.
     * @return A vector of PartiallySpecifiedStates where ALL bits are fully resolved (known_mask is completely filled).
     */
    static std::vector<PartiallySpecifiedState> sample_concrete_states(
        const PartiallySpecifiedState& current_belief,
        const ProblemDef& global_problem,
        int target_sample_count,
        const std::vector<std::vector<int>>& extra_constraints = {}
    );

    /**
     * @brief Invalidates this thread's Z3 state (fluent variables and the
     * permanently-asserted dead-end constraints; oneof constraints are asserted
     * per-query -- see Z3Manager::assert_relevant_oneofs). Must be called whenever
     * a new problem is loaded, before any sample_concrete_states() call for
     * that problem -- otherwise a thread that solves more than one problem
     * (e.g. sequential pytest cases without process isolation) silently
     * reuses the first problem's stale Z3 state for every subsequent one.
     */
    static void reset_z3_state();
};

} // namespace CPOR