#pragma once
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include "State.hpp"
#include "Evaluator.hpp"
#include "ProblemData.hpp"

extern ProblemDef global_problem;

struct PlanNode {
    int action_id{-1}; 
    PartiallySpecifiedState state;
    
    int single_child_idx{-1};
    int true_child_idx{-1};
    int false_child_idx{-1};

    // Required for backward regression walking
    int parent_idx{-1};
    int generating_action_id{-1};

    bool is_solved{false};
    int chosen_action_id{-1};

    // Explicit constructor to prevent aggregate initialization mapping errors
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

    // Computes heuristic and extracts Helpful Actions via goal backtracing
    int compute_heuristic(const PartiallySpecifiedState& state, std::vector<int>& out_helpful_actions) {
        PartiallySpecifiedState relaxed_state = state;
        int total_cost = 0;
        
        // Trackers for backtracing
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
                
                // Skip if already applied in previous layers
                if (action_layer[a_idx] != -1) continue; 

                // ----------------------------------------------------------------
                // L1 CACHE OPTIMIZATION: Bitwise Precondition Evaluation
                // ----------------------------------------------------------------
                bool is_applicable = false;
                // if (!action.has_complex_precondition) {
                //   is_applicable = true;
                    // ULTRA-FAST BITWISE SUBSET CHECK: (State & Precondition) == Precondition
                //    for (size_t k = 0; k < action.fast_precondition_mask.size(); ++k) {
                //        // A fact is explicitly True if both known and value bits are 1
                //        uint64_t actual_true_bits = relaxed_state.known_mask[k] & relaxed_state.value_mask[k];
                        
                        // Check if the state contains all bits required by the action's precondition
                //        if ((actual_true_bits & action.fast_precondition_mask[k]) != action.fast_precondition_mask[k]) {
                //            is_applicable = false;
                //            break;
                //        }
                //    }
                //} else {
                    // Fallback to heavy stack evaluator only for complex conditional logic
                is_applicable = (Evaluator::evaluate_rpn_raw(action.precondition_rpn, relaxed_state) == VAL_TRUE);
                //}

                if (is_applicable) {
                    bool applied_effect = false;
                    action_layer[a_idx] = current_layer;

                    // 1. Guaranteed effects
                    for (const auto& eff : action.guaranteed_effects) {
                        if (eff.second == true && !next_layer.is_true(eff.first)) {
                            next_layer.set_known_value(eff.first, true);
                            fact_achiever[eff.first] = a_idx;
                            applied_effect = true;
                        }
                    }

                    // 2. Conditional effects
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

                    // 3. Sensing targets
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
                
                // Goal reached. Initiate Fast Forward backtrace
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
                            // Action is applicable immediately
                            out_helpful_actions.push_back(achiever);
                        } else {
                            // Propagate dependencies backward
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

    // Returns FALSE if a mathematical contradiction is found (an impossible universe).
    bool apply_deductions(PartiallySpecifiedState& state) {
        bool state_changed = true;
        
        // Cascade deductions until equilibrium is reached
        while (state_changed) {
            state_changed = false;
            
            for (const auto& group : global_problem.oneofs) {
                int true_count = 0;
                int unknown_count = 0;
                int last_unknown_id = -1;

                // 1. Tally the current state of the mutually exclusive group
                for (int fact_id : group) {
                    if (state.is_true(fact_id)) {
                        true_count++;
                    } else if (state.is_unknown(fact_id)) {
                        unknown_count++;
                        last_unknown_id = fact_id;
                    }
                }

                // 2. Contradiction Detection
                // - More than one fact is True.
                // - Zero facts are True and there are no unknowns left (All False).
                if (true_count > 1 || (true_count == 0 && unknown_count == 0)) {
                    return false; 
                }

                // 3. Rule of Exclusivity: If ONE is true, all remaining unknowns MUST be false
                if (true_count == 1 && unknown_count > 0) {
                    for (int fact_id : group) {
                        if (state.is_unknown(fact_id)) {
                            state.set_known_value(fact_id, false);
                            state_changed = true;
                        }
                    }
                }
                // 4. Rule of Inevitability: If ALL BUT ONE are false, the last MUST be true
                else if (true_count == 0 && unknown_count == 1) {
                    state.set_known_value(last_unknown_id, true);
                    state_changed = true;
                }
            }
        }
        return true; // Equilibrium reached, state is mathematically valid
    }

    bool apply_effects(const PartiallySpecifiedState& current, const GroundedAction& action, PartiallySpecifiedState& out_next_state) {
        out_next_state = current;
        
        // 1. Apply Guaranteed Effects
        for (const auto& eff : action.guaranteed_effects) {
            out_next_state.set_known_value(eff.first, eff.second);
        }

        // 2. Evaluate and Apply Conditional Effects (Knowledge Loss)
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

        // 3. Apply Non-Deterministic Effects
        for (int nd_fact_id : action.non_deterministic_effects) {
            out_next_state.set_unknown(nd_fact_id);
        }

        // Attempt to reach logical equilibrium. 
        // If it returns false, the effects caused a paradox.
        return apply_deductions(out_next_state); 
    }

    bool check_consistency_with_regression(const std::vector<int>& negated_rpn, int start_node_idx) {
        std::vector<int> current_rpn = negated_rpn;
        int curr_idx = start_node_idx;

        while (curr_idx != -1) {
            const PlanNode& curr_node = node_pool[curr_idx];
            
            // 1. Evaluate the formula in the current historical state
            uint8_t eval = Evaluator::evaluate_rpn_raw(current_rpn, curr_node.state);
            
            // If we find explicit truth or falsehood, the regression concludes
            if (eval == VAL_TRUE) return true;
            if (eval == VAL_FALSE) return false;

            // 2. If unknown and we have a parent, regress through the generating action
            if (curr_node.parent_idx != -1 && curr_node.generating_action_id != -1) {
                const GroundedAction& gen_action = global_problem.actions[curr_node.generating_action_id];
                current_rpn = regress_rpn(current_rpn, gen_action);
            }
            
            curr_idx = curr_node.parent_idx;
        }

        // If we reach the root and it is still unknown, we assume it is possible 
        // to maintain Open World logic rules.
        return true; 
    }

    void apply_implicit_deduction(const std::vector<int>& rpn, PartiallySpecifiedState& state) {
        // The C# code AddObserved(Formula f) only extracts certainties from AND blocks and naked facts.
        // If an OR or NOT is present, it does not attempt complex SAT solving to guess the deduction.
        
        bool is_pure_conjunction = true;
        std::vector<int> facts_to_deduce;

        for (int token : rpn) {
            if (token >= 0) {
                facts_to_deduce.push_back(token);
            } else if (token != OP_AND) {
                // If the formula contains OR, NOT, ONEOF, or EQUALS we cannot simply
                // deduce that every contained fact is True.
                is_pure_conjunction = false;
                break;
            }
        }

        if (is_pure_conjunction) {
            for (int fact_id : facts_to_deduce) {
                // Force the dual mask bitset to explicit TRUE
                state.set_known_value(fact_id, true);
            }
            // Trigger intra state deductions to resolve oneofs using the newly learned facts
            apply_deductions(state);
        }
    }

    std::vector<int> negate_rpn(const std::vector<int>& rpn) const {
    if (rpn.empty()) {
        // An empty precondition evaluates to explicitly true
        // Therefore its mathematical negation is false
        return {OP_FALSE};
    }

    std::vector<int> negated;
    // Preallocate exact capacity to prevent internal reallocation
    negated.reserve(rpn.size() + 1);
    
    // Copy the original formula instruction sequence
    negated.insert(negated.end(), rpn.begin(), rpn.end());
    
    // Append the inversion operator at the absolute end of the stream
    negated.push_back(OP_NOT);
    
    return negated;
    }
    
    // Helper to abstract the C# 'IsApplicable' logic
    bool is_action_applicable(const GroundedAction& action, PartiallySpecifiedState& current_state, int current_node_idx) {
        if (action.precondition_rpn.empty()) return true;

        uint8_t eval_result = Evaluator::evaluate_rpn_raw(action.precondition_rpn, current_state);

        if (eval_result == VAL_TRUE) return true;
        if (eval_result == VAL_FALSE) return false;

        // VAL_UNKNOWN case: 
        // We lack current observations, but can we prove the negation is impossible via history?
        
        // C# Equivalent: bool possible = ConsistentWith(fNegatePreconditions, true);
        bool is_negation_possible = check_consistency_with_regression(negate_rpn(action.precondition_rpn), current_node_idx);
        
        if (!is_negation_possible) {
            // Implicit deduction: The negation is impossible, so the precondition MUST be true.
            // C# Equivalent: AddObserved(a.Preconditions);
            apply_implicit_deduction(action.precondition_rpn, current_state);
            return true;
        }
        
        return false;
    }

    // Cycle detection fix comparing actual state bits instead of node indices
    bool is_logical_cycle(const PartiallySpecifiedState& current_state, const std::vector<int>& path_indices) {
        for (int idx : path_indices) {
            if (node_pool[idx].state == current_state) {
                return true; 
            }
        }
        return false;
    }

    // Dead end detection using a bitwise Relaxed Planning Graph
    bool is_dead_end(const PartiallySpecifiedState& base_state) {
        PartiallySpecifiedState relaxed_state = base_state;
        bool changed = true;

        while (changed) {
            changed = false;
            
            // Check if goal is satisfied in the relaxed graph
            if (Evaluator::evaluate_rpn_raw(global_problem.goal_rpn, relaxed_state) == VAL_TRUE) {
                return false;
            }

            // Apply all applicable actions accumulating positive facts
            for (const auto& action : global_problem.actions) {
                if (Evaluator::evaluate_rpn_raw(action.precondition_rpn, relaxed_state) == VAL_TRUE) {
                    
                    // Accumulate guaranteed effects
                    for (const auto& eff : action.guaranteed_effects) {
                        int fact_id = eff.first;
                        bool is_positive = eff.second;
                        
                        if (is_positive && !relaxed_state.is_true(fact_id)) {
                            relaxed_state.known_mask[fact_id / 64] |= (1ULL << (fact_id % 64));
                            relaxed_state.value_mask[fact_id / 64] |= (1ULL << (fact_id % 64));
                            changed = true;
                        }
                    }

                    // Accumulate conditional effects
                    for (const auto& cond_eff : action.conditional_effects) {
                        if (Evaluator::evaluate_rpn_raw(cond_eff.condition_rpn, relaxed_state) == VAL_TRUE) {
                            for (const auto& eff : cond_eff.effects) {
                                int fact_id = eff.first;
                                bool is_positive = eff.second;
                                
                                if (is_positive && !relaxed_state.is_true(fact_id)) {
                                    relaxed_state.known_mask[fact_id / 64] |= (1ULL << (fact_id % 64));
                                    relaxed_state.value_mask[fact_id / 64] |= (1ULL << (fact_id % 64));
                                    changed = true;
                                }
                            }
                        }
                    }
                }
            }
        }

        return true;
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

        // Root node has no historical predecessor
        node_pool[root_idx].parent_idx = -1;
        node_pool[root_idx].generating_action_id = -1;

        return root_idx;
    }

    int expand_classical_node(int parent_idx, int action_id, const PartiallySpecifiedState& new_state) {
        node_pool.emplace_back(new_state, -1);
        int child_idx = node_pool.size() - 1;

        // Forward link
        node_pool[parent_idx].single_child_idx = child_idx;

        // Backward link required for regression
        node_pool[child_idx].parent_idx = parent_idx;
        node_pool[child_idx].generating_action_id = action_id;

        return child_idx;
    }

    void expand_sensing_node(int parent_idx, int action_id, int obs_pred_id, const PartiallySpecifiedState& current_state) {
        // True branch expansion
        PartiallySpecifiedState true_state = current_state;
        true_state.set_known_value(obs_pred_id, true);
        bool t_valid = apply_deductions(true_state); 
        
        node_pool.emplace_back(true_state, -1);
        int t_idx = node_pool.size() - 1;
        node_pool[t_idx].parent_idx = parent_idx;
        node_pool[t_idx].generating_action_id = action_id;
        if (!t_valid) node_pool[t_idx].is_solved = true; // Vacuously solved (unreachable)

        // False branch expansion
        PartiallySpecifiedState false_state = current_state;
        false_state.set_known_value(obs_pred_id, false);
        bool f_valid = apply_deductions(false_state); 
        
        node_pool.emplace_back(false_state, -1);
        int f_idx = node_pool.size() - 1;
        node_pool[f_idx].parent_idx = parent_idx;
        node_pool[f_idx].generating_action_id = action_id;
        if (!f_valid) node_pool[f_idx].is_solved = true; // Vacuously solved (unreachable)

        // Forward links
        node_pool[parent_idx].true_child_idx = t_idx;
        node_pool[parent_idx].false_child_idx = f_idx;
    }

    std::vector<int> regress_rpn(const std::vector<int>& rpn, const GroundedAction& action) const {
        std::vector<int> regressed_rpn;
        regressed_rpn.reserve(rpn.size());

        for (int token : rpn) {
            if (token >= 0) {
                bool modified_by_action = false;
                
                // Check if the generating action forced this fact to true or false unconditionally
                for (const auto& eff : action.guaranteed_effects) {
                    if (eff.first == token) {
                        modified_by_action = true;
                        // If the action added it, regression is OP_TRUE
                        // If the action deleted it, regression is OP_FALSE
                        regressed_rpn.push_back(eff.second ? OP_TRUE : OP_FALSE);
                        break;
                    }
                }
                
                // If the action did not unconditionally touch this fact, it remains unchanged
                if (!modified_by_action) {
                    regressed_rpn.push_back(token);
                }
            } else {
                // Operators transfer directly
                regressed_rpn.push_back(token);
            }
        }
        return regressed_rpn;
    }

    bool solve_from_node(int node_idx, std::vector<int>& current_path_indices) {
        PartiallySpecifiedState& current_state = node_pool[node_idx].state;

        // Check cache to avoid duplicate work
        if (solved_cache.find(current_state) != solved_cache.end()) return true;
        if (failed_cache.find(current_state) != failed_cache.end()) return false;

        // Evaluate goal
        if (Evaluator::evaluate_rpn_raw(global_problem.goal_rpn, current_state) == VAL_TRUE) {
            node_pool[node_idx].is_solved = true;
            solved_cache[current_state] = node_idx;
            return true;
        }

        // Prune mathematical dead ends early
        if (is_dead_end(current_state)) {
            failed_cache.insert(current_state);
            return false;
        }

        current_path_indices.push_back(node_idx);

        std::vector<ActionCandidate> candidates;
        candidates.reserve(global_problem.actions.size());

        // ====================================================================
        // PHASE 1: Candidate Gathering via Helpful Actions Pruning
        // ====================================================================
        
        std::vector<int> helpful_actions;
        int base_h = compute_heuristic(current_state, helpful_actions);
        
        // Impossible universe
        if (base_h == 999999) return false; 
        
        std::unordered_set<int> helpful_set(helpful_actions.begin(), helpful_actions.end());

        for (size_t i = 0; i < global_problem.actions.size(); ++i) {
            
            // Branching Pruning
            if (!helpful_set.count(i)) continue;
            
            const auto& action = global_problem.actions[i];
            if (!is_action_applicable(action, current_state, node_idx)) continue;

            int h_val = 0;
            if (action.observe_predicate_id == -1) {
                PartiallySpecifiedState next_state;
                if (!apply_effects(current_state, action, next_state)) continue;
                
                std::vector<int> dummy;
                h_val = compute_heuristic(next_state, dummy);
            } else {
                if (!current_state.is_unknown(action.observe_predicate_id)) continue;

                PartiallySpecifiedState t_state = current_state;
                t_state.set_known_value(action.observe_predicate_id, true);
                bool t_valid = apply_deductions(t_state);

                PartiallySpecifiedState f_state = current_state;
                f_state.set_known_value(action.observe_predicate_id, false);
                bool f_valid = apply_deductions(f_state);

                if (!t_valid && !f_valid) continue;

                std::vector<int> dummy;
                int h_t = t_valid ? compute_heuristic(t_state, dummy) : 0; 
                int h_f = f_valid ? compute_heuristic(f_state, dummy) : 0; 
                h_val = std::max(h_t, h_f); 
            }

            if (h_val < 999999) candidates.push_back({(int)i, h_val});
        }

        std::sort(candidates.begin(), candidates.end(), [](const ActionCandidate& a, const ActionCandidate& b) {
            return a.h_score < b.h_score;
        });

        // Track path dependent failures to prevent global cache poisoning
        bool failed_due_to_loop = false;

        // ====================================================================
        // PHASE 2: Candidate Expansion & Recursion
        // ====================================================================
        for (const auto& candidate : candidates) {
            const auto& action = global_problem.actions[candidate.action_idx];

            if (action.observe_predicate_id == -1) {
                PartiallySpecifiedState next_state;
                
                //Re-verify physics in case the candidate list was generated in a weird order
                // or if we rely on the side-effects of apply_effects
                if (!apply_effects(current_state, action, next_state)) {
                    failed_cache.insert(next_state); // Poison the bad state
                    continue;
                }
                
                
                if (is_logical_cycle(next_state, current_path_indices)) {
                    failed_due_to_loop = true;
                    continue; 
                }

                int child_idx = expand_classical_node(node_idx, action.id, next_state);
                
                if (solve_from_node(child_idx, current_path_indices)) {
                    node_pool[node_idx].is_solved = true;
                    node_pool[node_idx].chosen_action_id = action.id;
                    solved_cache[current_state] = node_idx; 
                    current_path_indices.pop_back(); 
                    return true; 
                }
            } else {
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

                // If a branch is physically impossible (!valid), it is vacuously solved.
                // We only need to recurse on branches that actually exist in reality.
                bool t_solved = !t_valid || solve_from_node(t_idx, current_path_indices);
                bool f_solved = !f_valid || solve_from_node(f_idx, current_path_indices);

                if (t_solved && f_solved) {
                    node_pool[node_idx].is_solved = true;
                    node_pool[node_idx].chosen_action_id = action.id;
                    solved_cache[current_state] = node_idx;
                    current_path_indices.pop_back();
                    return true;
                }
            }
        }

        current_path_indices.pop_back();
        
        // Cache only if all branches fundamentally failed
        if (!failed_due_to_loop) {
            failed_cache.insert(current_state); 
        }
        
        return false; 
    }
};