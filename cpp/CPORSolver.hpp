#pragma once
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <iostream>
#include "State.hpp"
#include "Evaluator.hpp"
#include "ProblemData.hpp"
#include "SDRSampler.hpp"
#include "FFSolver.hpp"

extern ProblemDef global_problem;

struct PlanNode {
    int action_id{-1}; 
    PartiallySpecifiedState state;
    
    int single_child_idx{-1};
    int true_child_idx{-1};
    int false_child_idx{-1};

    int parent_idx{-1};
    int generating_action_id{-1};

    bool is_solved{false};
    bool is_failed{false};
    int chosen_action_id{-1};

    PlanNode(const PartiallySpecifiedState& s, int act_id) : action_id(act_id), state(s) {}
};

class CPORSolver {
private:
    std::vector<PlanNode> node_pool;
    std::unordered_map<PartiallySpecifiedState, int, StateHasher> solved_cache; 
    std::unordered_set<PartiallySpecifiedState, StateHasher> failed_cache; 
    
    struct ActionCandidate {
        int action_idx;
        int h_score;
    };

    // Old function, currently in use: compute_heuristic_FF instead of compute_heuristic
    int compute_heuristic(const PartiallySpecifiedState& state, std::vector<int>& out_helpful_actions) {
        PartiallySpecifiedState relaxed_state = state;
        int total_cost = 0;
        
        std::vector<int> fact_achiever(global_problem.total_predicates, -1);
        std::vector<int> action_layer(global_problem.actions.size(), -1);
        
        if (Evaluator::evaluate_rpn_raw(global_problem.goal_rpn, relaxed_state) == VAL_TRUE) return 0;

        int current_layer = 0;
        while (true) {
            PartiallySpecifiedState next_layer = relaxed_state;
            bool state_changed = false;
            int min_cost_this_layer = 999999;

            for (size_t a_idx = 0; a_idx < global_problem.actions.size(); ++a_idx) {
                const auto& action = global_problem.actions[a_idx];
                
                if (action_layer[a_idx] != -1) continue; 

                bool is_applicable = (Evaluator::evaluate_rpn_raw(action.precondition_rpn, relaxed_state) == VAL_TRUE);

                if (is_applicable) {
                    bool applied_effect = false;
                    action_layer[a_idx] = current_layer;

                    for (const auto& eff : action.guaranteed_effects) {
                        if (eff.second == true && !next_layer.is_true(eff.first)) {
                            next_layer.set_known_value(eff.first, true);
                            fact_achiever[eff.first] = a_idx;
                            applied_effect = true;
                        }
                    }

                    for (const auto& cond_eff : action.conditional_effects) {
                        if (Evaluator::evaluate_rpn_raw(cond_eff.condition_rpn, relaxed_state) == VAL_TRUE) {
                            for (const auto& eff : cond_eff.effects) {
                                if (eff.second == true && !next_layer.is_true(eff.first)) {
                                    next_layer.set_known_value(eff.first, true);
                                    if (fact_achiever[eff.first] == -1) fact_achiever[eff.first] = a_idx;
                                    applied_effect = true;
                                }
                            }
                        }
                    }

                    if (action.observe_predicate_id != -1 && !next_layer.is_true(action.observe_predicate_id)) {
                        next_layer.set_known_value(action.observe_predicate_id, true);
                        if (fact_achiever[action.observe_predicate_id] == -1) fact_achiever[action.observe_predicate_id] = a_idx;
                        applied_effect = true;
                    }

                    if (applied_effect) {
                        state_changed = true;
                        min_cost_this_layer = std::min(min_cost_this_layer, action.cost);
                    }
                }
            }
            
            if (min_cost_this_layer != 999999) total_cost += min_cost_this_layer;
            else total_cost += 1;

            if (Evaluator::evaluate_rpn_raw(global_problem.goal_rpn, next_layer) == VAL_TRUE) {
                std::vector<int> goals_to_achieve;
                for (int token : global_problem.goal_rpn) {
                    if (token >= 0 && !state.is_true(token)) goals_to_achieve.push_back(token);
                }

                std::unordered_set<int> processed_goals;
                while (!goals_to_achieve.empty()) {
                    int g = goals_to_achieve.back();
                    goals_to_achieve.pop_back();

                    if (processed_goals.count(g)) continue;
                    processed_goals.insert(g);

                    int achiever = fact_achiever[g];
                    if (achiever != -1) {
                        if (action_layer[achiever] == 0) {
                            out_helpful_actions.push_back(achiever);
                        } else {
                            const auto& act = global_problem.actions[achiever];
                            for (int pre : act.precondition_rpn) {
                                if (pre >= 0 && !state.is_true(pre)) goals_to_achieve.push_back(pre);
                            }
                        }
                    }
                }
                return total_cost;
            }
            
            if (!state_changed) return 999999; 
            
            relaxed_state = next_layer;
            current_layer++;
        }
    }

    bool apply_deductions(PartiallySpecifiedState& state) {
        bool state_changed = true;
        while (state_changed) {
            state_changed = false;
            for (const auto& group : global_problem.oneofs) {
                int true_count = 0;
                int unknown_count = 0;
                int last_unknown_id = -1;

                for (int fact_id : group) {
                    if (state.is_true(fact_id)) {
                        true_count++;
                    } else if (state.is_unknown(fact_id)) {
                        unknown_count++;
                        last_unknown_id = fact_id;
                    }
                }

                if (true_count > 1 || (true_count == 0 && unknown_count == 0)) {
                    return false; 
                }

                if (true_count == 1 && unknown_count > 0) {
                    for (int fact_id : group) {
                        if (state.is_unknown(fact_id)) {
                            state.set_known_value(fact_id, false);
                            state_changed = true;
                        }
                    }
                }
                else if (true_count == 0 && unknown_count == 1) {
                    state.set_known_value(last_unknown_id, true);
                    state_changed = true;
                }
            }
        }
        return true; 
    }

    bool apply_effects(const PartiallySpecifiedState& current, const GroundedAction& action, PartiallySpecifiedState& out_next_state) {
        out_next_state = current;
        for (const auto& eff : action.guaranteed_effects) {
            out_next_state.set_known_value(eff.first, eff.second);
        }
        for (const auto& cond_eff : action.conditional_effects) {
            uint8_t eval = Evaluator::evaluate_rpn_raw(cond_eff.condition_rpn, current);
            if (eval == VAL_TRUE) {
                for (const auto& eff : cond_eff.effects) {
                    out_next_state.set_known_value(eff.first, eff.second);
                }
            } 
            else if (eval == VAL_UNKNOWN) {
                for (const auto& eff : cond_eff.effects) {
                    out_next_state.set_unknown(eff.first);
                }
            }
        }
        for (int nd_fact_id : action.non_deterministic_effects) {
            out_next_state.set_unknown(nd_fact_id);
        }
        return apply_deductions(out_next_state); 
    }

    bool check_consistency_with_regression(const std::vector<int>& negated_rpn, int start_node_idx) {
        std::vector<int> current_rpn = negated_rpn;
        int curr_idx = start_node_idx;

        while (curr_idx != -1) {
            const PlanNode& curr_node = node_pool[curr_idx];
            uint8_t eval = Evaluator::evaluate_rpn_raw(current_rpn, curr_node.state);
            if (eval == VAL_TRUE) return true;
            if (eval == VAL_FALSE) return false;

            if (curr_node.parent_idx != -1 && curr_node.generating_action_id != -1) {
                const GroundedAction& gen_action = global_problem.actions[curr_node.generating_action_id];
                current_rpn = regress_rpn(current_rpn, gen_action);
            }
            curr_idx = curr_node.parent_idx;
        }
        return true; 
    }

    void apply_implicit_deduction(const std::vector<int>& rpn, PartiallySpecifiedState& state) {
        bool is_pure_conjunction = true;
        std::vector<int> facts_to_deduce;

        for (int token : rpn) {
            if (token >= 0) {
                facts_to_deduce.push_back(token);
            } else if (token != OP_AND) {
                is_pure_conjunction = false;
                break;
            }
        }

        if (is_pure_conjunction) {
            for (int fact_id : facts_to_deduce) {
                state.set_known_value(fact_id, true);
            }
            apply_deductions(state);
        }
    }

    std::vector<int> negate_rpn(const std::vector<int>& rpn) const {
        if (rpn.empty()) return {OP_FALSE};
        std::vector<int> negated;
        negated.reserve(rpn.size() + 1);
        negated.insert(negated.end(), rpn.begin(), rpn.end());
        negated.push_back(OP_NOT);
        return negated;
    }
    
    bool is_action_applicable(const GroundedAction& action, PartiallySpecifiedState& current_state, int current_node_idx) {
        if (action.precondition_rpn.empty()) return true;

        uint8_t eval_result = Evaluator::evaluate_rpn_raw(action.precondition_rpn, current_state);

        if (eval_result == VAL_TRUE) return true;
        if (eval_result == VAL_FALSE) return false;

        bool is_negation_possible = check_consistency_with_regression(negate_rpn(action.precondition_rpn), current_node_idx);
        if (!is_negation_possible) {
            apply_implicit_deduction(action.precondition_rpn, current_state);
            return true;
        }
        return false;
    }

    // Cycle detection for the current branch
    bool is_logical_cycle(const PartiallySpecifiedState& state, const std::vector<int>& current_path) {
        for (int idx : current_path) {
            if (node_pool[idx].state == state) return true;
        }
        return false;
    }

    int compute_heuristic_FF(const PartiallySpecifiedState& state) {
        // Sample a single concrete witness state (Guide State)
        auto samples = CPOR::SDRSampler::sample_concrete_states(state, global_problem, 1);
        if (samples.empty()) {
            return 999999; // Contradictory state (Dead End)
        }

        // Run the C-based FF Solver on the determinized guide state
        std::vector<int> plan = CPOR::FFSolver::search(samples[0], global_problem);
        if (plan.empty()) {
            return 999999; // Dead End detected by FF
        }

        return plan.size(); // Length of the relaxed plan
    }

    
public:
    CPORSolver() {
        node_pool.reserve(1000000); 
    }

    int get_node_count() const { return node_pool.size(); }
    const PlanNode& get_node(int idx) const { return node_pool[idx]; }

    int create_root_node(PartiallySpecifiedState initial_state) {
        apply_deductions(initial_state); 
        node_pool.emplace_back(initial_state, -1);
        int root_idx = node_pool.size() - 1;
        node_pool[root_idx].parent_idx = -1;
        node_pool[root_idx].generating_action_id = -1;
        return root_idx;
    }

    int expand_classical_node(int parent_idx, int action_id, const PartiallySpecifiedState& new_state) {
        node_pool.emplace_back(new_state, -1);
        int child_idx = node_pool.size() - 1;
        node_pool[parent_idx].single_child_idx = child_idx;
        node_pool[child_idx].parent_idx = parent_idx;
        node_pool[child_idx].generating_action_id = action_id;
        return child_idx;
    }

    void expand_sensing_node(int parent_idx, int action_id, int obs_pred_id, const PartiallySpecifiedState& current_state) {
        PartiallySpecifiedState true_state = current_state;
        true_state.set_known_value(obs_pred_id, true);
        bool t_valid = apply_deductions(true_state); 
        
        node_pool.emplace_back(true_state, -1);
        int t_idx = node_pool.size() - 1;
        node_pool[t_idx].parent_idx = parent_idx;
        node_pool[t_idx].generating_action_id = action_id;
        if (!t_valid) node_pool[t_idx].is_solved = true; 

        PartiallySpecifiedState false_state = current_state;
        false_state.set_known_value(obs_pred_id, false);
        bool f_valid = apply_deductions(false_state); 
        
        node_pool.emplace_back(false_state, -1);
        int f_idx = node_pool.size() - 1;
        node_pool[f_idx].parent_idx = parent_idx;
        node_pool[f_idx].generating_action_id = action_id;
        if (!f_valid) node_pool[f_idx].is_solved = true; 

        node_pool[parent_idx].true_child_idx = t_idx;
        node_pool[parent_idx].false_child_idx = f_idx;
    }

    std::vector<int> regress_rpn(const std::vector<int>& rpn, const GroundedAction& action) const {
        std::vector<int> regressed_rpn;
        regressed_rpn.reserve(rpn.size());
        for (int token : rpn) {
            if (token >= 0) {
                bool modified_by_action = false;
                for (const auto& eff : action.guaranteed_effects) {
                    if (eff.first == token) {
                        modified_by_action = true;
                        regressed_rpn.push_back(eff.second ? OP_TRUE : OP_FALSE);
                        break;
                    }
                }
                if (!modified_by_action) {
                    regressed_rpn.push_back(token);
                }
            } else {
                regressed_rpn.push_back(token);
            }
        }
        return regressed_rpn;
    }

    bool solve_from_node(int node_idx, std::vector<int>& current_path_indices, int depth = 0) {
        PartiallySpecifiedState& current_state = node_pool[node_idx].state;
        std::string ind(depth * 2, ' ');

        if (solved_cache.find(current_state) != solved_cache.end()) {
            int cached_idx = solved_cache[current_state];
            // Extract the solution from the cache to the current node so Python reconstructs it as a DAG tree
            node_pool[node_idx].chosen_action_id = node_pool[cached_idx].chosen_action_id;
            node_pool[node_idx].single_child_idx = node_pool[cached_idx].single_child_idx;
            node_pool[node_idx].true_child_idx = node_pool[cached_idx].true_child_idx;
            node_pool[node_idx].false_child_idx = node_pool[cached_idx].false_child_idx;
            node_pool[node_idx].is_solved = true;
            return true;
        }

        if (failed_cache.find(current_state) != failed_cache.end()) {
            node_pool[node_idx].is_failed = true;
            return false;
        }

        if (Evaluator::evaluate_rpn_raw(global_problem.goal_rpn, current_state) == VAL_TRUE) {
            std::cout << ind << "[CPOR] Node " << node_idx << " -> GOAL REACHED!" << std::endl;
            node_pool[node_idx].is_solved = true;
            solved_cache[current_state] = node_idx;
            return true;
        }

        current_path_indices.push_back(node_idx);

        std::vector<ActionCandidate> candidates;
        candidates.reserve(global_problem.actions.size());

        // Call the C-API determinized FF solver 
        int base_h = compute_heuristic_FF(current_state);
        
        if (base_h == 999999) {
            std::cout << ind << "[CPOR] Node " << node_idx << " -> DEAD END (Heuristic=999999)" << std::endl;
            current_path_indices.pop_back();
            failed_cache.insert(current_state);
            node_pool[node_idx].is_failed = true; // explicitly backpropagate failure
            return false; 
        }

        for (size_t i = 0; i < global_problem.actions.size(); ++i) {
            const auto& action = global_problem.actions[i];
            if (!is_action_applicable(action, current_state, node_idx)) continue;

            int h_val = 0;
            if (action.observe_predicate_id == -1) {
                PartiallySpecifiedState next_state;
                if (!apply_effects(current_state, action, next_state)) continue;
                
                h_val = compute_heuristic_FF(next_state);
            } else {
                if (!current_state.is_unknown(action.observe_predicate_id)) continue;

                PartiallySpecifiedState t_state = current_state;
                t_state.set_known_value(action.observe_predicate_id, true);
                bool t_valid = apply_deductions(t_state);

                PartiallySpecifiedState f_state = current_state;
                f_state.set_known_value(action.observe_predicate_id, false);
                bool f_valid = apply_deductions(f_state);

                if (!t_valid && !f_valid) continue;

                int h_t = t_valid ? compute_heuristic_FF(t_state) : 0; 
                int h_f = f_valid ? compute_heuristic_FF(f_state) : 0; 
                h_val = std::max(h_t, h_f); 
            }

            if (h_val < 999999) candidates.push_back({(int)i, h_val});
        }

        std::sort(candidates.begin(), candidates.end(), [](const ActionCandidate& a, const ActionCandidate& b) {
            return a.h_score < b.h_score;
        });

        std::cout << ind << "[CPOR] Node " << node_idx << " Expanding " << candidates.size() << " candidates..." << std::endl;

        bool failed_due_to_loop = false;

        for (const auto& candidate : candidates) {
            const auto& action = global_problem.actions[candidate.action_idx];

            if (action.observe_predicate_id == -1) {
                PartiallySpecifiedState next_state;
                if (!apply_effects(current_state, action, next_state)) {
                    failed_cache.insert(next_state); 
                    continue;
                }
                
                if (is_logical_cycle(next_state, current_path_indices)) {
                    failed_due_to_loop = true;
                    continue; 
                }

                std::cout << ind << " |- Trying classical action " << action.id << " (h=" << candidate.h_score << ")" << std::endl;
                int child_idx = expand_classical_node(node_idx, action.id, next_state);
                
                if (solve_from_node(child_idx, current_path_indices, depth + 1)) {
                    node_pool[node_idx].is_solved = true;
                    node_pool[node_idx].chosen_action_id = action.id;
                    solved_cache[current_state] = node_idx; 
                    current_path_indices.pop_back(); 
                    return true; 
                }
            } else {
                std::cout << ind << " |- Trying sensing action " << action.id << " (h=" << candidate.h_score << ")" << std::endl;
                PartiallySpecifiedState t_state = current_state;
                t_state.set_known_value(action.observe_predicate_id, true);
                bool t_valid = apply_deductions(t_state);

                PartiallySpecifiedState f_state = current_state;
                f_state.set_known_value(action.observe_predicate_id, false);
                bool f_valid = apply_deductions(f_state);

                if ((t_valid && is_logical_cycle(t_state, current_path_indices)) || (f_valid && is_logical_cycle(f_state, current_path_indices))) {
                    failed_due_to_loop = true;
                    continue;
                }

                expand_sensing_node(node_idx, action.id, action.observe_predicate_id, current_state);
                
                int t_idx = node_pool[node_idx].true_child_idx;
                int f_idx = node_pool[node_idx].false_child_idx;

                bool t_solved = !t_valid || solve_from_node(t_idx, current_path_indices, depth + 1);
                bool f_solved = !f_valid || solve_from_node(f_idx, current_path_indices, depth + 1);

                if (t_solved && f_solved) {
                    node_pool[node_idx].is_solved = true;
                    node_pool[node_idx].chosen_action_id = action.id;
                    solved_cache[current_state] = node_idx;
                    current_path_indices.pop_back();
                    return true;
                } else {
                    std::cout << ind << " |-> Sensing action " << action.id << " failed (one or both branches dead)" << std::endl;
                }
            }
        }

        current_path_indices.pop_back();
        
        // Final fallback: if all candidates failed (and not just skipped due to a cycle)
        if (!failed_due_to_loop) {
            failed_cache.insert(current_state); 
            node_pool[node_idx].is_failed = true; // Safely backpropagate the death of this node to its parent
        }
        
        return false; 
    }
};