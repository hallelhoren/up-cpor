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
#include "BFSSolver.hpp"
#include "ActionApplier.hpp"

namespace CPOR { 

struct PlanNode {
    int action_id{-1}; 
    PartiallySpecifiedState state;
    
    // For classical actions (OR nodes)
    int single_child_idx{-1};
    
    // For sensing actions (AND nodes)
    int true_child_idx{-1};
    int false_child_idx{-1};

    int parent_idx{-1};
    int generating_action_id{-1};

    bool is_solved{false};
    bool is_failed{false};
    int chosen_action_id{-1};

    PlanNode(const PartiallySpecifiedState& s, int act_id) : state(s), action_id(act_id) {}
};

class CPORSolver {
private:
    std::vector<PlanNode> node_pool;
    std::unordered_map<PartiallySpecifiedState, int, StateHasher> solved_cache; 
    std::unordered_set<PartiallySpecifiedState, StateHasher> failed_cache; 
    
    struct ActionCandidate {
        int action_idx;
        int h_score;
        
        bool operator<(const ActionCandidate& other) const {
            return h_score < other.h_score;
        }
    };

    // Internal Epistemic Logic
    void apply_oneof_deductions(PartiallySpecifiedState& state);
    void expand_sensing_node(int node_idx, int action_id, int observe_fluent, const PartiallySpecifiedState& current_state);
    
    bool is_action_applicable(const GroundedAction& action, const PartiallySpecifiedState& state);
    int compute_heuristic(const PartiallySpecifiedState& state);

public:
    CPORSolver() {
        node_pool.reserve(500000); // Pre-allocate Arena to avoid vector reallocation invalidating indices
    }

    const PlanNode& get_node(int idx) const {
        return node_pool[idx];
    }

    int create_root_node(const PartiallySpecifiedState& initial_state);
    bool solve_from_node(int node_idx, std::vector<int>& current_path_indices, int depth = 0);
    
    size_t get_node_count() const { return node_pool.size(); }
    void print_conditional_plan(int node_idx, int indent = 0) const;
};

} // namespace CPOR