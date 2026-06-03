#include <iostream>
#include <cassert>
#include <vector>
#include "State.hpp"
#include "Evaluator.hpp" // We will write this next!

void run_evaluator_tests() {
    std::cout << "Running Evaluator.hpp Tests..." << std::endl;

    // Initialize a state and set some facts
    PartiallySpecifiedState state(100);
    state.set_known_value(5, true);   // Fact 5 is TRUE
    state.set_known_value(12, false); // Fact 12 is FALSE
    // Fact 50 remains UNKNOWN

    // TEST 1: Empty RPN (implicitly true, standard for empty preconditions)
    assert(Evaluator::evaluate_rpn({}, state) == true);

    // TEST 2: Single Fact (True)
    assert(Evaluator::evaluate_rpn({5}, state) == true);

    // TEST 3: Single Fact (False)
    assert(Evaluator::evaluate_rpn({12}, state) == false);

    // TEST 4: Single Fact (Unknown)
    // If a precondition requires an unknown fact, the action is NOT applicable yet.
    assert(Evaluator::evaluate_rpn({50}, state) == false);

    // TEST 5: NOT Operation -> (NOT 12) should be True
    assert(Evaluator::evaluate_rpn({12, OP_NOT}, state) == true);

    // TEST 6: AND Operation -> (5 AND NOT 12) should be True
    // RPN: 5, 12, NOT, AND
    assert(Evaluator::evaluate_rpn({5, 12, OP_NOT, OP_AND}, state) == true);

    // TEST 7: OR Operation -> (12 OR 5) should be True
    assert(Evaluator::evaluate_rpn({12, 5, OP_OR}, state) == true);

    // TEST 8: Complex Formula -> (5 AND 12) OR (NOT 12)
    // RPN: 5, 12, AND, 12, NOT, OR
    // (True AND False) OR (True) -> False OR True -> True
    assert(Evaluator::evaluate_rpn({5, 12, OP_AND, 12, OP_NOT, OP_OR}, state) == true);

    std::cout << "SUCCESS: All Evaluator.hpp tests passed!" << std::endl;
}

int main() {
    run_evaluator_tests();
    return 0;
}