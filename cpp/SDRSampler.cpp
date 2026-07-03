#include "SDRSampler.hpp"
#include <z3++.h>
#include <string>
#include <stdexcept>

namespace CPOR {

std::vector<PartiallySpecifiedState> SDRSampler::sample_concrete_states(
    const PartiallySpecifiedState& current_belief,
    const ProblemDef& global_problem,
    int target_sample_count) 
{
    std::vector<PartiallySpecifiedState> sampled_states;
    if (target_sample_count <= 0) return sampled_states;

    // 1. Initialize Z3 Context and Solver
    z3::context ctx;
    z3::solver solver(ctx);

    // 2. Map all total_predicates to Z3 boolean variables
    std::vector<z3::expr> fluent_vars;
    fluent_vars.reserve(global_problem.total_predicates);

    for (int i = 0; i < global_problem.total_predicates; ++i) {
        std::string var_name = "f_" + std::to_string(i);
        fluent_vars.push_back(ctx.bool_const(var_name.c_str()));
    }

    // 3. Assert current Known Facts from the Dual-Mask State
    for (int i = 0; i < global_problem.total_predicates; ++i) {
        if (!current_belief.is_unknown(i)) {
            if (current_belief.is_true(i)) {
                solver.add(fluent_vars[i]);        // Must be TRUE
            } else {
                solver.add(!fluent_vars[i]);       // Must be FALSE
            }
        }
        // If UNKNOWN, we leave the variable unconstrained for Z3 to sample
    }

    // 4. Assert Structural Invariants (OneOf constraints passed from Grounder)
    for (const auto& oneof_group : global_problem.oneofs) {
        if (oneof_group.empty()) continue;

        z3::expr_vector group_vars(ctx);
        for (int fluent_id : oneof_group) {
            group_vars.push_back(fluent_vars[fluent_id]);
        }

        // Rule A: At least one must be true (OR over all)
        solver.add(z3::mk_or(group_vars));

        // Rule B: Mutually exclusive (Pairwise NOT AND)
        // For larger groups, Z3's PbEq/AtMostOne is preferred, but pairwise is fast for small sets.
        for (size_t i = 0; i < oneof_group.size(); ++i) {
            for (size_t j = i + 1; j < oneof_group.size(); ++j) {
                solver.add(!(fluent_vars[oneof_group[i]] && fluent_vars[oneof_group[j]]));
            }
        }
    }

    // 5. Sample & Block Loop (Generating Diverse Concrete States)
    for (int k = 0; k < target_sample_count; ++k) {
        if (solver.check() != z3::sat) {
            break; // No more valid models exist in this belief space
        }

        z3::model model = solver.get_model();
        PartiallySpecifiedState concrete_state(global_problem.total_predicates);
        z3::expr_vector blocking_clause(ctx);

        // Extract the model into a concrete state
        for (int i = 0; i < global_problem.total_predicates; ++i) {
            z3::expr var = fluent_vars[i];
            z3::expr eval = model.eval(var, true); 
            bool is_true = eval.is_true();

            // Set the concrete bit in the determinized state
            concrete_state.set_known_value(i, is_true);

            // Build blocking clause: flip at least one currently unknown bit to force diversity
            if (current_belief.is_unknown(i)) {
                if (is_true) {
                    blocking_clause.push_back(!var);
                } else {
                    blocking_clause.push_back(var);
                }
            }
        }

        sampled_states.push_back(concrete_state);

        // Add blocking clause to force the solver to find a different path for the next sample
        if (!blocking_clause.empty()) {
            solver.add(z3::mk_or(blocking_clause));
        } else {
            // If there were no unknown bits to flip, the state is strictly deterministic
            break; 
        }
    }

    return sampled_states;
}

} // namespace CPOR