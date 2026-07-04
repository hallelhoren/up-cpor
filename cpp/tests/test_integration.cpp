#include <iostream>
#include <cassert>
#include <vector>
#include "../ProblemData.hpp"
#include "../State.hpp"
#include "../Evaluator.hpp"

// Explicitly link to the global problem memory stored in native_bridge.cpp
extern ProblemDef global_problem;

// Expose the C-API bridge functions so we can simulate Python's behavior
extern "C" {
    void init_problem(int total_predicates);
    void add_initial_fact(int fact_id);
    void add_action(int id, int* pre_rpn, int pre_len, int* eff_ids, bool* eff_vals, int eff_len, int obs_id);
}

void run_integration_tests() {
    std::cout << "Running Full Pipeline Integration Test..." << std::endl;

    // ==============================================================================
    // PHASE 1: THE BRIDGE (Simulate Python sending PDDL data)
    // ==============================================================================
    init_problem(20); // Create a universe with 20 possible predicates
    add_initial_fact(2); // Fact 2 is explicitly TRUE initially

    // Create a mock action: "Pick up Block"
    // Precondition: Fact 2 must be TRUE
    // Effect: Fact 5 becomes TRUE
    int pre_rpn[] = {2}; 
    int eff_ids[] = {5}; 
    bool eff_vals[] = {true}; 
    
    add_action(1, pre_rpn, 1, eff_ids, eff_vals, 1, -1);

    // Verify the Bridge successfully wrote to C++ memory
    assert(global_problem.total_predicates == 20);
    assert(global_problem.actions.size() == 1);


    // ==============================================================================
    // PHASE 2: THE STATE (Simulate the Solver starting up)
    // ==============================================================================
    PartiallySpecifiedState current_state(global_problem.total_predicates);
    
    // The solver reads the initial facts and sets them in the bitset
    for (int fact_id : global_problem.initial_true_facts) {
        current_state.set_known_value(fact_id, true);
    }

    // Verify the physics engine initialized correctly
    assert(current_state.is_true(2) == true);
    assert(current_state.is_unknown(5) == true); // Fact 5 hasn't happened yet


    // ==============================================================================
    // PHASE 3: THE EVALUATOR (Simulate checking if the action is legal)
    // ==============================================================================
    const GroundedAction& action = global_problem.actions[0];
    
    // Check if the action's RPN is satisfied by the current bitset state
    bool is_legal = Evaluator::evaluate_rpn(action.precondition_rpn, current_state);
    assert(is_legal == true); // It requires Fact 2 to be true, which it is!


    // ==============================================================================
    // PHASE 4: EXECUTION (Simulate applying the action's effects)
    // ==============================================================================
    for (const auto& effect : action.effects) {
        current_state.set_known_value(effect.first, effect.second);
    }

    // Verify the universe was successfully altered by the action!
    assert(current_state.is_true(5) == true);
    assert(current_state.is_unknown(5) == false);

    std::cout << "SUCCESS: Integration across Bridge -> State -> Evaluator is mathematically sound!" << std::endl;
}

int main() {
    run_integration_tests();
    return 0;
}