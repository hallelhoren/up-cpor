#include "SDRSampler.hpp"
#include "Z3Manager.hpp"
#include <z3++.h>
#include <string>

namespace CPOR {

// Thread-local storage ensures lock-free performance in a multi-threaded heuristic search
thread_local Z3Manager g_z3_manager;

std::vector<PartiallySpecifiedState> SDRSampler::sample_concrete_states(
    const PartiallySpecifiedState& current_belief,
    const ProblemDef& global_problem,
    int target_sample_count) 
{
    std::vector<PartiallySpecifiedState> sampled_states;
    if (target_sample_count <= 0) return sampled_states;

    // 1. Initialize Z3 Manager (Fast exit if already initialized)
    g_z3_manager.initialize(global_problem);

    z3::context& ctx = g_z3_manager.ctx;
    z3::solver& solver = g_z3_manager.solver;
    const std::vector<z3::expr>& fluent_vars = g_z3_manager.fluent_vars;

    // 2. Scope the Solver State
    // Pushes a new frame onto the solver stack. Everything added after this 
    // will be erased when pop() is called.
    solver.push();

    // 3. Assert Known Epistemic Facts for THIS specific evaluation
    for (int i = 0; i < global_problem.total_predicates; ++i) {
        if (!current_belief.is_unknown(i)) {
            if (current_belief.is_true(i)) {
                solver.add(fluent_vars[i]);
            } else {
                solver.add(!fluent_vars[i]);
            }
        }
    }

    // 4. Sample and Block (Diverse Generation)
    for (int k = 0; k < target_sample_count; ++k) {
        if (solver.check() != z3::sat) {
            break; // No more valid models exist under current constraints
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

        // Add the blocking clause to the current scope so we don't sample this reality again
        if (!blocking_clause.empty()) {
            solver.add(z3::mk_or(blocking_clause));
        } else {
            // Space is strictly deterministic; no diverse states can be generated
            break; 
        }
    }

    // 5. Clean up the Solver Stack
    // This instantly wipes the belief assertions and blocking clauses from Z3's memory,
    // leaving the OneOf invariants completely intact at depth 0.
    solver.pop();

    return sampled_states;
}

} // namespace CPOR