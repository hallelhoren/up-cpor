#pragma once
#include <z3++.h>
#include <vector>
#include <string>
#include "ProblemData.hpp"

namespace CPOR {

class Z3Manager {
public:
    z3::context ctx;
    z3::solver solver;
    std::vector<z3::expr> fluent_vars;
    bool is_initialized;

    Z3Manager() : solver(ctx), is_initialized(false) {}

    // Initializes the context once per thread.
    void initialize(const ProblemDef& problem) {
        if (is_initialized) return;

        fluent_vars.reserve(problem.total_predicates);

        // 1. Create persistent Z3 Boolean Variables
        for (int i = 0; i < problem.total_predicates; ++i) {
            std::string var_name = "f_" + std::to_string(i);
            fluent_vars.push_back(ctx.bool_const(var_name.c_str()));
        }

        // 2. Assert Global Invariants (OneOfs) permanently at depth 0
        for (const auto& oneof_group : problem.oneofs) {
            if (oneof_group.empty()) continue;
            
            z3::expr_vector group_vars(ctx);
            for (int fluent_id : oneof_group) {
                group_vars.push_back(fluent_vars[fluent_id]);
            }
            
            // At least one is true
            solver.add(z3::mk_or(group_vars)); 
            
            // Mutually exclusive (at most one is true)
            for (size_t i = 0; i < oneof_group.size(); ++i) {
                for (size_t j = i + 1; j < oneof_group.size(); ++j) {
                    solver.add(!(fluent_vars[oneof_group[i]] && fluent_vars[oneof_group[j]]));
                }
            }
        }
        
        is_initialized = true;
    }
};

} // namespace CPOR