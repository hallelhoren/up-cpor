#pragma once
#include <vector>
#include <queue>
#include <unordered_set>
#include <iostream>
#include "State.hpp"
#include "Evaluator.hpp"
#include "ActionApplier.hpp"
#include "ProblemData.hpp"

extern ProblemDef global_problem;

class BFSSolver {
public:
    std::vector<int> solve() {
        PartiallySpecifiedState initial_state;
        
        int blocks = (global_problem.total_predicates / 64) + 1;
        
        initial_state.known_mask.assign(blocks, ~0ULL);
        initial_state.value_mask.assign(blocks, 0);

        for (int id : global_problem.initial_true_facts) {
            initial_state.value_mask[id / 64] |= (1ULL << (id % 64));
        }
        
        for (int id : global_problem.initial_false_facts) {
            initial_state.value_mask[id / 64] &= ~(1ULL << (id % 64));
        }

        std::queue<std::pair<PartiallySpecifiedState, std::vector<int>>> q;
        std::unordered_set<PartiallySpecifiedState, StateHasher> visited;

        q.push({initial_state, {}});
        visited.insert(initial_state);
        
        int expanded = 0;

        while (!q.empty()) {
            auto current = q.front();
            q.pop();

            PartiallySpecifiedState curr_state = current.first;
            std::vector<int> path = current.second;

            if (Evaluator::evaluate_rpn_raw(global_problem.goal_rpn, curr_state) == VAL_TRUE) {
                std::cout << "[BFSSolver] Goal found! Expanded " << expanded << " states." << std::endl;
                return path;
            }

            for (const auto& action : global_problem.actions) {
                if (Evaluator::evaluate_rpn_raw(action.precondition_rpn, curr_state) == VAL_TRUE) {
                    
                    PartiallySpecifiedState next_state = curr_state;
                    ActionApplier::apply_action(action, next_state);

                    if (visited.find(next_state) == visited.end()) {
                        visited.insert(next_state);
                        std::vector<int> next_path = path;
                        next_path.push_back(action.id);
                        q.push({next_state, next_path});
                    }
                }
            }
            
            expanded++;
            if (expanded > 100000) {
                 std::cout << "[BFSSolver] Safety limit reached." << std::endl;
                 break;
            }
        }
        
        std::cout << "[BFSSolver] Search exhausted. Evaluated " << expanded << " states." << std::endl;
        return {}; 
    }
};