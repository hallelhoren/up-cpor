#include "ProblemData.hpp"
#include "State.hpp"
#include "Evaluator.hpp"
#include "CPORSolver.hpp" 
#include <vector>
#include <iostream>
#include <unordered_set>

ProblemDef global_problem;

extern "C" {

    void init_problem(int total_predicates, int total_functions = 0) {
        global_problem = ProblemDef(); 
        global_problem.total_predicates = total_predicates;
        global_problem.total_functions = total_functions;

        // Initialize the comparable_mask with all 1s (all variables comparable by default)
        // Python can later zero-out the bits for constants/options via a new API call if needed.
        int blocks = (total_predicates / 64) + 1;
        global_problem.comparable_mask.assign(blocks, ~0ULL);
    }

    void add_initial_fact(int fact_id) {
        global_problem.initial_true_facts.push_back(fact_id);
    }

    void set_goal_rpn(int* rpn_array, int length) {
        global_problem.goal_rpn.assign(rpn_array, rpn_array + length);
    }

    // Renamed legacy function to populate guaranteed effects
    void add_action(int id, int cost, int* pre_rpn, int pre_len, int* eff_ids, bool* eff_vals, int eff_len, int obs_id) {
        GroundedAction act;
        act.id = id;
        act.cost = cost; 
        
        // Allocate the bitset mask to exactly match the state dimension size
        int blocks = (global_problem.total_predicates / 64) + 1;
        act.fast_precondition_mask.assign(blocks, 0ULL);
        bool is_complex = false;

        if (pre_len > 0) {
            act.precondition_rpn.assign(pre_rpn, pre_rpn + pre_len);
            
            // PRE-COMPILATION: Analyze the RPN string. 
            // If it only contains positive facts and OP_AND (-1), it is a pure STRIPS action.
            for (int i = 0; i < pre_len; ++i) {
                int token = pre_rpn[i];
                if (token >= 0) {
                    act.fast_precondition_mask[token / 64] |= (1ULL << (token % 64));
                } else if (token != -1) { // -1 is OP_AND in Evaluator.hpp
                    is_complex = true; // Contains OR, NOT, ONEOF, etc.
                }
            }
        }
        
        act.has_complex_precondition = is_complex;

        for(int i = 0; i < eff_len; ++i) {
            act.guaranteed_effects.push_back({eff_ids[i], eff_vals[i]});
        }
        
        act.observe_predicate_id = obs_id;
        global_problem.actions.push_back(std::move(act));
    }

    //Attach a conditional effect to an existing action
    void add_conditional_effect(int action_id, int* cond_rpn, int cond_len, int* eff_ids, bool* eff_vals, int eff_len) {
        for (auto& act : global_problem.actions) {
            if (act.id == action_id) {
                ConditionalEffect ce;
                ce.condition_rpn.assign(cond_rpn, cond_rpn + cond_len);
                for(int i = 0; i < eff_len; ++i) {
                    ce.effects.push_back({eff_ids[i], eff_vals[i]});
                }
                act.conditional_effects.push_back(std::move(ce));
                return;
            }
        }
    }

    //Attach a non-deterministic target to an existing action
    void add_non_deterministic_effect(int action_id, int fact_id) {
        for (auto& act : global_problem.actions) {
            if (act.id == action_id) {
                act.non_deterministic_effects.push_back(fact_id);
                return;
            }
        }
    }

    void add_oneof_constraint(int* ids_array, int length) {
        std::vector<int> oneof_vec(ids_array, ids_array + length);
        global_problem.oneofs.push_back(oneof_vec);
    }

    //API to load a dead-end formula into the problem definition
    void add_deadend_rpn(int* rpn_array, int length) {
        if (length > 0) {
            std::vector<int> rpn(rpn_array, rpn_array + length);
            global_problem.deadend_rpns.push_back(std::move(rpn));
        }
    }

    void add_initial_unknown_fact(int fact_id) {
        global_problem.initial_unknown_facts.push_back(fact_id);
    }

    void print_problem_stats() {
        std::cout << "=== C++ Native Memory Verification ===" << std::endl;
        std::cout << "Total Predicates: " << global_problem.total_predicates << std::endl;
        std::cout << "Initial Facts: " << global_problem.initial_true_facts.size() << std::endl;
        std::cout << "Actions Loaded: " << global_problem.actions.size() << std::endl;
        std::cout << "======================================" << std::endl;
    }


// =====================================================================
    // GLOBAL SOLVER (Persists in memory so Python can read the plan later)
    // =====================================================================
    CPORSolver* global_solver = nullptr;

    bool solve_native() {
        std::cout << "\n=== Starting Native CPOR AND/OR Search ===" << std::endl;
        
        if (global_solver) delete global_solver;
        global_solver = new CPORSolver();
        
        PartiallySpecifiedState initial_state(global_problem.total_predicates, global_problem.total_functions);
        
        // 1. CLOSED WORLD FALLBACK: Set everything to False initially
        for (int i = 0; i < global_problem.total_predicates; i++) {
            initial_state.set_known_value(i, false);
        }

        // 2. EXPLICITLY TRUE FACTS
        for (int fact_id : global_problem.initial_true_facts) {
            initial_state.set_known_value(fact_id, true);
        }

        // 3. ISOLATED UNKNOWN FACTS (NEW: Open World Assumption override)
        for (int fact_id : global_problem.initial_unknown_facts) {
            initial_state.set_unknown(fact_id);
        }

        // 4. ONEOF CONSTRAINTS (Also sets facts to unknown, pending deductions)
        for (const auto& group : global_problem.oneofs) {
            for (int fact_id : group) {
                initial_state.set_unknown(fact_id);
            }
        }
        
        //  Observable facts are UNKNOWN.
        // If an action senses a fact, that fact is inherently a hidden variable.
        for (const auto& action : global_problem.actions) {
            if (action.observe_predicate_id != -1) {
                int obs = action.observe_predicate_id;
                initial_state.known_mask[obs / 64] &= ~(1ULL << (obs % 64)); // Force to Unknown
            }
        }

        int root_idx = global_solver->create_root_node(initial_state);
        std::unordered_set<PartiallySpecifiedState, StateHasher> visited;

        bool success = global_solver->solve_from_node(root_idx, visited);

        if (success) {
            std::cout << ">>> SUCCESS: Contingent Plan Found! <<<" << std::endl;
            std::cout << "Total Universes Explored (Node Count): " << global_solver->get_node_count() << std::endl;
        }
        return success;
    }
    // =====================================================================
    // PLAN EXTRACTION GETTERS FOR PYTHON
    // =====================================================================
    int get_chosen_action(int node_idx) { 
        return global_solver->get_node(node_idx).chosen_action_id; 
    }
    
    int get_single_child(int node_idx) { 
        return global_solver->get_node(node_idx).single_child_idx; 
    }
    
    int get_true_child(int node_idx) { 
        return global_solver->get_node(node_idx).true_child_idx; 
    }
    
    int get_false_child(int node_idx) { 
        return global_solver->get_node(node_idx).false_child_idx; 
    }
} // End of extern "C"