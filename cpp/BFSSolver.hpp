#pragma once
#include <vector>
#include <queue>
#include <unordered_map>
#include <iostream>
#include <algorithm>
#include "State.hpp"
#include "Evaluator.hpp"
#include "ActionApplier.hpp"
#include "ProblemData.hpp"

namespace CPOR {

struct BFSNode {
    PartiallySpecifiedState state;
    int parent_index; // Index of the parent in the arena vector (-1 for root)
    int action_id;    // The action that generated this state (-1 for root)
};

class BFSSolver {
public:
    /**
     * @brief Executes an optimal Breadth-First Search from the current belief state.
     * @param current_state The epistemic state the agent is CURRENTLY in (do not use T0 initial facts).
     * @param problem The global problem definition.
     * @param max_expansions Safety cap on nodes expanded. Callers doing genuine plan
     *        extraction should keep the default; callers using this purely as a cheap
     *        heuristic signal (e.g. CPORSolver::compute_heuristic, called once per
     *        candidate action per search node) should pass a much smaller bound --
     *        a full blind search to 100k nodes is not a heuristic, it's a second solver.
     * @param verbose Emit progress/diagnostic logging. Disable for high-frequency
     *        heuristic calls where per-call iostream flushes become their own overhead.
     * @return A sequence of Action IDs representing the shortest path to the goal. Empty if unsolvable.
     */
    static std::vector<int> solve(const PartiallySpecifiedState& current_state, const ProblemDef& problem,
                                   int max_expansions = 100000, bool verbose = true) {

        // Check if we are already at the goal
        if (Evaluator::evaluate(problem.goal_rpn, current_state, EvalMode::PESSIMISTIC)) {
            if (verbose) std::cout << "[BFSSolver] Start state is already the goal." << std::endl;
            return {};
        }

        // Memory Arena: Preallocate to prevent vector reallocations and memory fragmentation
        std::vector<BFSNode> node_arena;
        node_arena.reserve(static_cast<size_t>(max_expansions) + 1); // Tune based on average problem size

        // Map state hash directly to its index in the node_arena
        std::unordered_map<PartiallySpecifiedState, int, StateHasher> visited;

        // Queue stores INDICES to the arena, NOT copies of states or paths
        std::queue<int> open_list;

        // Initialize Root
        node_arena.push_back({current_state, -1, -1});
        visited[current_state] = 0;
        open_list.push(0);

        int expanded = 0;
        const int MAX_EXPANSIONS = max_expansions;

        while (!open_list.empty()) {
            int current_index = open_list.front();
            open_list.pop();
            // Must be a copy, not a reference: the loop below calls node_arena.push_back()
            // repeatedly, and once node_arena grows past its initial reserve() capacity,
            // reallocation invalidates any reference into it -- corrupting curr_state
            // mid-iteration and causing a use-after-free read on later actions.
            const PartiallySpecifiedState curr_state = node_arena[current_index].state;

            // Goal evaluation is done ON GENERATION below. 
            // We just expand the current node here.
            
            for (const auto& action : problem.actions) {
                // Check applicability deterministically
                if (Evaluator::evaluate(action.precondition_rpn, curr_state, EvalMode::PESSIMISTIC)) {
                    
                    PartiallySpecifiedState next_state = curr_state;
                    ActionApplier::apply_action(action, next_state);

                    // If unvisited, generate node
                    if (visited.find(next_state) == visited.end()) {
                        
                        // Goal Evaluation on PUSH (Optimal for BFS uniform cost)
                        if (Evaluator::evaluate(problem.goal_rpn, next_state, EvalMode::PESSIMISTIC)) {
                            if (verbose) std::cout << "[BFSSolver] Goal found! Expanded " << expanded << " states." << std::endl;

                            // Reconstruct path
                            return reconstruct_path(current_index, action.id, node_arena);
                        }

                        // Register new node in arena
                        int next_index = static_cast<int>(node_arena.size());
                        node_arena.push_back({next_state, current_index, action.id});
                        
                        visited[next_state] = next_index;
                        open_list.push(next_index);
                    }
                }
            }
            
            expanded++;
            if (expanded >= MAX_EXPANSIONS) {
                 if (verbose) std::cerr << "[BFSSolver] Safety limit reached (" << MAX_EXPANSIONS << " expansions). Aborting." << std::endl;
                 break;
            }
        }

        if (verbose) std::cout << "[BFSSolver] Search exhausted. No plan found." << std::endl;
        return {};
    }

private:
    /**
     * @brief Reconstructs the action path by walking backward through the parent indices.
     * Logically identical to the C# GeneratePlan method, but uses contiguous memory.
     */
    static std::vector<int> reconstruct_path(int final_parent_index, int final_action_id, const std::vector<BFSNode>& arena) {
        std::vector<int> path;
        
        // Walk backwards from the parent of the goal state to the root
        int current = final_parent_index;
        path.push_back(final_action_id); // The action that actually achieved the goal

        while (current != -1 && arena[current].parent_index != -1) {
            path.push_back(arena[current].action_id);
            current = arena[current].parent_index;
        }

        // Reverse to get chronological order
        std::reverse(path.begin(), path.end());
        return path;
    }
};

} // namespace CPOR