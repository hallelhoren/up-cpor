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

    // 1. Initialize Z3 Context
    z3::context ctx;
    z3::solver solver(ctx);

    std::vector<z3::expr> fluent_vars;
    fluent_vars.reserve(global_problem.total_predicates);

    // Create Z3 Boolean Variables
    for (int i = 0; i < global_problem.total_predicates; ++i) {
        std::string var_name = "f_" + std::to_string(i);
        fluent_vars.push_back(ctx.bool_const(var_name.c_str()));
    }

    // 2. Assert Known Facts
    for (int i = 0; i < global_problem.total_predicates; ++i) {
        if (!current_belief.is_unknown(i)) {
            if (current_belief.is_true(i)) solver.add(fluent_vars[i]);
            else solver.add(!fluent_vars[i]);
        }
    }

    // 3. Assert OneOf Invariants
    for (const auto& oneof_group : global_problem.oneofs) {
        if (oneof_group.empty()) continue;
        z3::expr_vector group_vars(ctx);
        for (int fluent_id : oneof_group) {
            group_vars.push_back(fluent_vars[fluent_id]);
        }
        
        solver.add(z3::mk_or(group_vars)); // At least one true
        for (size_t i = 0; i < oneof_group.size(); ++i) {
            for (size_t j = i + 1; j < oneof_group.size(); ++j) {
                solver.add(!(fluent_vars[oneof_group[i]] && fluent_vars[oneof_group[j]])); // Mutually exclusive
            }
        }
    }

    // 4. Sample and Block (Diverse Generation)
    for (int k = 0; k < target_sample_count; ++k) {
        if (solver.check() != z3::sat) {
            break; // No more valid models
        }

        z3::model model = solver.get_model();
        PartiallySpecifiedState concrete_state(global_problem.total_predicates);
        z3::expr_vector blocking_clause(ctx);

        for (int i = 0; i < global_problem.total_predicates; ++i) {
            z3::expr var = fluent_vars[i];
            bool is_true = model.eval(var, true).is_true();
            concrete_state.set_known_value(i, is_true);

            // Force solver to pick a different assignment for unknown bits next time
            if (current_belief.is_unknown(i)) {
                if (is_true) blocking_clause.push_back(!var);
                else blocking_clause.push_back(var);
            }
        }

        sampled_states.push_back(concrete_state);

        if (!blocking_clause.empty()) {
            solver.add(z3::mk_or(blocking_clause));
        } else {
            // Strictly deterministic space, cannot generate diverse states
            break; 
        }
    }

    return sampled_states;
}

} // namespace CPOR