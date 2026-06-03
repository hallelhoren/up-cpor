#include <iostream>
#include <cassert>
#include "State.hpp"
#include "CPORSolver.hpp" // We will write this next!

void run_solver_tests() {
    std::cout << "Running CPORSolver.hpp Tests..." << std::endl;

    CPORSolver solver;

    // Initialize a blank state
    PartiallySpecifiedState initial_state(100);
    
    // TEST 1: Create Root Node
    int root_idx = solver.create_root_node(initial_state);
    assert(root_idx == 0);
    assert(solver.get_node_count() == 1);

    // TEST 2: Classical Action Expansion (e.g., picking up a block)
    // Action ID = 5. It should create exactly 1 child node.
    int child_idx = solver.expand_classical_node(root_idx, 5, initial_state); 
    assert(child_idx == 1);
    assert(solver.get_node_count() == 2);
    assert(solver.get_node(root_idx).single_child_idx == 1);

    // TEST 3: Sensing Action Expansion (SDR Split)
    // Action ID = 8, Observed Predicate ID = 42. It MUST create 2 children.
    PartiallySpecifiedState current_state = solver.get_node(child_idx).state;
    solver.expand_sensing_node(child_idx, 8, 42, current_state);

    assert(solver.get_node_count() == 4); // Root(1) + Child(1) + TrueBranch(1) + FalseBranch(1)
    
    // Check if the parent correctly points to its new split children
    assert(solver.get_node(child_idx).true_child_idx == 2);
    assert(solver.get_node(child_idx).false_child_idx == 3);

    // Verify the physics of the SDR Split
    assert(solver.get_node(2).state.is_true(42) == true);   // Child 2 is the "Heads" universe
    assert(solver.get_node(3).state.is_true(42) == false);  // Child 3 is the "Tails" universe
    assert(solver.get_node(3).state.is_unknown(42) == false); // We definitely know it's Tails

    std::cout << "SUCCESS: All CPORSolver.hpp tests passed!" << std::endl;
}

int main() {
    run_solver_tests();
    return 0;
}