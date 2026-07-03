#pragma once
#include <memory>
#include <vector>

namespace CPOR {

/**
 * @enum PlanNodeType
 * @brief Identifies the functional role of a PlanNode without relying on RTTI or virtual dispatch.
 */
enum class PlanNodeType {
    ACTION,
    OBSERVATION,
    GOAL
};

/**
 * @struct PlanNode
 * @brief A lightweight, unified node structure representing the Contingent Plan Graph.
 * Avoids virtual inheritance overhead to maximize L1/L2 cache locality during graph execution.
 */
struct PlanNode {
    PlanNodeType type;
    
    // ACTION data
    int action_id{-1};
    std::shared_ptr<PlanNode> next{nullptr};
    
    // OBSERVATION (Branching) data
    int observation_predicate_id{-1};
    std::shared_ptr<PlanNode> true_child{nullptr};
    std::shared_ptr<PlanNode> false_child{nullptr};

    // Constructors for safe, expressive instantiation
    static std::shared_ptr<PlanNode> make_action(int a_id) {
        auto node = std::make_shared<PlanNode>();
        node->type = PlanNodeType::ACTION;
        node->action_id = a_id;
        return node;
    }

    static std::shared_ptr<PlanNode> make_observation(int predicate_id) {
        auto node = std::make_shared<PlanNode>();
        node->type = PlanNodeType::OBSERVATION;
        node->observation_predicate_id = predicate_id;
        return node;
    }

    static std::shared_ptr<PlanNode> make_goal() {
        auto node = std::make_shared<PlanNode>();
        node->type = PlanNodeType::GOAL;
        return node;
    }
};

} // namespace CPOR