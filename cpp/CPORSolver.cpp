#include "CPORSolver.hpp"

namespace CPOR {

int CPORSolver::create_root_node(const PartiallySpecifiedState& initial_state) {
    node_pool.emplace_back(initial_state, -1);
    return 0;
}

// -----------------------------------------------------------------------------
// SDR Epistemic Deduction (Deductive Closure)
// -----------------------------------------------------------------------------
void CPORSolver::apply_oneof_deductions(PartiallySpecifiedState& state) {
    const ProblemDef& problem = get_global_problem();
    bool changed = true;
    
    // Loop until fixpoint is reached
    while (changed) {
        changed = false;
        for (const auto& group : problem.oneofs) {
            int true_count = 0;
            int unknown_count = 0;
            int last_unknown = -1;
            
            for (int p : group) {
                if (state.is_true(p)) true_count++;
                else if (state.is_unknown(p)) {
                    unknown_count++;
                    last_unknown = p;
                }
            }
            
            // Rule 1: If exactly one is true, all other unknowns MUST be false
            if (true_count > 0 && unknown_count > 0) {
                for (int p : group) {
                    if (state.is_unknown(p)) {
                        state.set_known_value(p, false);
                        changed = true;
                    }
                }
            } 
            // Rule 2: If all are false except one unknown, the unknown MUST be true
            else if (true_count == 0 && unknown_count == 1) {
                state.set_known_value(last_unknown, true);
                changed = true;
            }
        }
    }
}

// -----------------------------------------------------------------------------
// AND-Node Expansion (Epistemic Branching)
// -----------------------------------------------------------------------------
void CPORSolver::expand_sensing_node(int node_idx, int action_id, int observe_fluent, const PartiallySpecifiedState& current_state) {
    // 1. Create TRUE Universe
    PartiallySpecifiedState true_state = current_state;
    true_state.set_known_value(observe_fluent, true);
    apply_oneof_deductions(true_state);
    
    int t_idx = static_cast<int>(node_pool.size());
    node_pool.emplace_back(true_state, action_id);
    node_pool[t_idx].parent_idx = node_idx;

    // 2. Create FALSE Universe
    PartiallySpecifiedState false_state = current_state;
    false_state.set_known_value(observe_fluent, false);
    apply_oneof_deductions(false_state);
    
    int f_idx = static_cast<int>(node_pool.size());
    node_pool.emplace_back(false_state, action_id);
    node_pool[f_idx].parent_idx = node_idx;

    // Link back to AND node
    node_pool[node_idx].true_child_idx = t_idx;
    node_pool[node_idx].false_child_idx = f_idx;
}

bool CPORSolver::is_action_applicable(const GroundedAction& action, const PartiallySpecifiedState& state) {
    // PESSIMISTIC mode: If a precondition is UNKNOWN, we mathematically CANNOT apply the action.
    return Evaluator::evaluate(action.precondition_rpn, state, EvalMode::PESSIMISTIC);
}

int CPORSolver::compute_heuristic(const PartiallySpecifiedState& state) {
    const ProblemDef& problem = get_global_problem();
    
    // Extract exactly ONE determinized reality from the belief state to feed the classical solver
    auto samples = SDRSampler::sample_concrete_states(state, problem, 1);
    if (samples.empty()) return 999999; // Mathematically Dead State (Zero valid models)
    
    std::vector<int> path = BFSSolver::solve(samples[0], problem);
    
    if (path.empty()) {
        // If empty, check if it's already the goal. If not, it's a dead end.
        if (Evaluator::evaluate(problem.goal_rpn, samples[0], EvalMode::PESSIMISTIC)) return 0;
        return 999999;
    }
    return static_cast<int>(path.size());
}

// -----------------------------------------------------------------------------
// Core Contingent AND/OR Search Logic
// -----------------------------------------------------------------------------
bool CPORSolver::solve_from_node(int node_idx, std::vector<int>& current_path_indices, int depth) {
    const ProblemDef& problem = get_global_problem();
    PartiallySpecifiedState current_state = node_pool[node_idx].state;

    // Depth Protection
    if (depth > 50) return false;

    // Cache / Loop Verification
    if (solved_cache.find(current_state) != solved_cache.end()) {
        node_pool[node_idx].is_solved = true;
        return true;
    }
    if (failed_cache.find(current_state) != failed_cache.end()) {
        return false;
    }
    if (std::find(current_path_indices.begin(), current_path_indices.end(), node_idx) != current_path_indices.end()) {
        return false; // Loop detected
    }

    // Goal Check
    if (Evaluator::evaluate(problem.goal_rpn, current_state, EvalMode::PESSIMISTIC)) {
        node_pool[node_idx].is_solved = true;
        solved_cache[current_state] = node_idx;
        return true;
    }

    current_path_indices.push_back(node_idx);
    std::vector<ActionCandidate> candidates;

    // Generate Candidates
    for (size_t i = 0; i < problem.actions.size(); ++i) {
        const GroundedAction& action = problem.actions[i];
        if (is_action_applicable(action, current_state)) {
            
            // Lookahead determinization logic
            PartiallySpecifiedState projected_state = current_state;
            ActionApplier::apply_action(action, projected_state);
            
            int h = compute_heuristic(projected_state);
            if (h < 999999) {
                candidates.push_back({static_cast<int>(i), h});
            }
        }
    }

    // Sort by Best-First Heuristic
    std::sort(candidates.begin(), candidates.end());
    bool failed_due_to_loop = false;

    // Attempt Execution
    for (const auto& candidate : candidates) {
        const GroundedAction& action = problem.actions[candidate.action_idx];

        // OR Node (Classical Action)
        if (action.observe_predicate_id == -1) {
            PartiallySpecifiedState next_state = current_state;
            ActionApplier::apply_action(action, next_state);

            int child_idx = static_cast<int>(node_pool.size());
            node_pool.emplace_back(next_state, action.id);
            node_pool[child_idx].parent_idx = node_idx;
            node_pool[node_idx].single_child_idx = child_idx;

            if (solve_from_node(child_idx, current_path_indices, depth + 1)) {
                node_pool[node_idx].is_solved = true;
                node_pool[node_idx].chosen_action_id = action.id;
                solved_cache[current_state] = node_idx;
                current_path_indices.pop_back();
                return true;
            }
        } 
        // AND Node (Sensing Action)
        else {
            // Optimization: If we already know the observation, it's just a regular action.
            if (!current_state.is_unknown(action.observe_predicate_id)) continue;

            expand_sensing_node(node_idx, action.id, action.observe_predicate_id, current_state);
            
            int t_idx = node_pool[node_idx].true_child_idx;
            int f_idx = node_pool[node_idx].false_child_idx;

            // BOTH branches must yield a valid plan
            bool t_solved = solve_from_node(t_idx, current_path_indices, depth + 1);
            if (!t_solved) continue; // Early short-circuit

            bool f_solved = solve_from_node(f_idx, current_path_indices, depth + 1);
            
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
    
    if (!failed_due_to_loop) {
        failed_cache.insert(current_state);
    }
    
    return false;
}

void CPORSolver::print_conditional_plan(int node_idx, int indent) const {
    if (node_idx < 0 || node_idx >= node_pool.size()) return;
    const PlanNode& node = node_pool[node_idx];

    std::string padding(indent * 2, ' ');
    
    if (node.is_solved && node.chosen_action_id != -1) {
        const GroundedAction& action = get_global_problem().actions[node.chosen_action_id];
        
        if (action.observe_predicate_id == -1) {
            std::cout << padding << "ACTION: " << node.chosen_action_id << "\n";
            print_conditional_plan(node.single_child_idx, indent);
        } else {
            std::cout << padding << "SENSE: " << node.chosen_action_id << " (Fluent " << action.observe_predicate_id << ")\n";
            std::cout << padding << "├─ [TRUE Branch]:\n";
            print_conditional_plan(node.true_child_idx, indent + 2);
            std::cout << padding << "└─ [FALSE Branch]:\n";
            print_conditional_plan(node.false_child_idx, indent + 2);
        }
    } else if (node.is_solved) {
        std::cout << padding << "-> [GOAL REACHED]\n";
    } else {
        std::cout << padding << "-> [DEAD END]\n";
    }
}

} // namespace CPOR