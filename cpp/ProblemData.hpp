#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include <utility>

/**
 * @file ProblemData.hpp
 * @brief Defines the Intermediate Grounded Representation (IGR) for the CPOR native backend.
 * * This file embraces Data-Oriented Design (DOD). By replacing the legacy C# architecture's 
 * heavily nested object trees (e.g., HashSet<Predicate>, virtual methods) with flat integer 
 * arrays and bit-friendly structures, we maximize CPU L1/L2 cache locality, prevent memory 
 * fragmentation, and completely avoid the latency overhead of recursive pointer-chasing 
 * during state evaluation.
 */

 // Defines arithmetic operations for numeric effects
enum class NumericOp { 
    ASSIGN = 0, 
    INCREASE = 1, 
    DECREASE = 2 
};

// Represents a modification to a numeric function
struct NumericEffect {
    int function_id;
    NumericOp op;
    double value;
};

 struct ConditionalEffect {
    std::vector<int> condition_rpn;
    std::vector<std::pair<int, bool>> effects;
};

/**
 * @struct GroundedAction
 * @brief Represents a fully instantiated action in the planning problem.
 * * Replaces the legacy `PlanningAction.cs` and `ParametrizedAction.cs`. All variables 
 * have been bound to constants, allowing us to evaluate preconditions and apply effects 
 * using lightning-fast integer array traversals rather than matching string parameters.
 */
struct GroundedAction {
    /** @brief Unique identifier for the action. */
    int id{0};
    int cost{1};

    /** * @brief Flattened Reverse Polish Notation (RPN) array of the logic formula.
     * * By flattening the abstract syntax tree into a contiguous array, we guarantee 
     * cache-friendly sequential memory access. A stack-based evaluator iterating over 
     * this vector is significantly faster than recursively traversing `CompoundFormula` objects.
     */
    std::vector<int> precondition_rpn{};

    /** * @brief The effects of the action, represented as a list of (predicate_id, new_truth_value).
     * * Using flat integer pairs eliminates the need for allocating `Predicate` objects on the heap.
     */
    // Split into guaranteed, conditional, and non-deterministic
    std::vector<std::pair<int, bool>> guaranteed_effects{};
    std::vector<ConditionalEffect> conditional_effects{};

    std::vector<NumericEffect> numeric_effects{};
    std::vector<int> non_deterministic_effects{}; // List of fact IDs that become UNKNOWN

    /** * @brief ID of the observed predicate for sensing actions. 
     * Defaults to -1 for standard, non-sensing actions.
     */
    int observe_predicate_id{-1};
};

/**
 * @struct ProblemDef
 * @brief The entirely flattened representation of the contingent planning problem.
 * * Replaces the legacy `Problem.cs` and `Domain.cs`. This acts as the payload contract 
 * between the Python administration layer and the C++ computation core.
 */
struct ProblemDef {
    /** * @brief Total number of unique grounded facts in the universe (N).
     * * This defines the size required for our state bitsets. Using bitsets bounded by 
     * this number allows O(1) state comparisons and ultra-low memory footprints.
     */
    int total_predicates{0};
    int total_functions{0};

    /** * @brief Integer IDs of facts that are explicitly true in the initial state.
     */
    std::vector<int> initial_true_facts{};

    std::vector<int> initial_false_facts{};

    std::vector<std::pair<int, double>> initial_function_values{};

    /** * @brief Flattened RPN array representing the goal formula. 
     */
    std::vector<int> goal_rpn{};

    /** * @brief The contiguous list of all grounded actions available in the domain.
     */
    std::vector<GroundedAction> actions{};

    // Each inner vector holds the IDs of predicates where EXACTLY ONE is true.
    std::vector<std::vector<int>> oneofs{};

    // Precomputed mask to zero out Option variables and AlwaysConstant variables
    // Computed once during Python bridge initialization
    std::vector<uint64_t> comparable_mask{};

    // Dead-End constraints. If any of these evaluate to True, the state is pruned.
    std::vector<std::vector<int>> deadend_rpns{};

    std::vector<int> auto_observable_predicates;

    // Mirrors the legacy C# engine's Domain.IsSimple: false the instant any
    // action anywhere in the problem has a conditional effect (see
    // up_cpor.problem_grounder's generate_native_problem, which computes this
    // exactly that way and is the only real setter -- see
    // native_bridge.cpp's set_problem_is_simple). Gates CPORSolver's Plan
    // Graph Compaction (K(n)/H(n) relevant-fluent belief-equivalence caching,
    // CPOR TAAS 2022 Algorithm 3/4): applying that mechanism outside simple
    // domains is a soundness hazard, not just a missed-optimization one --
    // see CPORSolver.cpp's compute_and_register_relevance for why. Defaults
    // to false (compaction off) so any caller that never explicitly sets
    // this -- in particular every existing cpp/tests/*.cpp, which construct
    // ProblemDef directly via the extern "C" API and never call
    // set_problem_is_simple -- keeps exactly its prior behavior.
    bool is_simple{false};

};

inline ProblemDef& get_global_problem() {
        static ProblemDef instance;
        return instance;
    }

    // Forces a total wipe of the domain memory between PDDL runs.
    inline void reset_global_problem() {
        get_global_problem() = ProblemDef(); 
    }