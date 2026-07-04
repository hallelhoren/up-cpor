#include <iostream>
#include <cassert>
#include <vector>
#include "../ProblemData.hpp"
#include "../State.hpp"
#include "../Evaluator.hpp"

// Mock global problem required by State.hpp
ProblemDef global_problem;

// Simple macro to emulate GTest/Catch2 assertions
#define TEST_CHECK(name, condition) \
    if (!(condition)) { \
        std::cerr << "[FAILED] " << name << std::endl; \
        exit(1); \
    } else { \
        std::cout << "[PASSED] " << name << std::endl; \
    }

void initialize_test_environment() {
    global_problem.total_predicates = 10;
    global_problem.total_functions = 0;
    int blocks = (10 / 64) + 1;
    global_problem.comparable_mask.assign(blocks, ~0ULL);
}

void test_three_valued_truth_tables() {
    std::cout << "\n--- Running Truth Table Tests ---\n";
    PartiallySpecifiedState state(10, 0);
    
    state.set_known_value(0, true);   // F0 is True
    state.set_known_value(1, false);  // F1 is False
    state.set_unknown(2);             // F2 is Unknown

    // AND TESTS
    std::vector<int> rpn_and_true_unk = {0, 2, OP_AND};
    TEST_CHECK("AND(True, Unknown) == Unknown", 
        Evaluator::evaluate_rpn_raw(rpn_and_true_unk, state) == VAL_UNKNOWN);

    std::vector<int> rpn_and_false_unk = {1, 2, OP_AND};
    TEST_CHECK("AND(False, Unknown) == False", 
        Evaluator::evaluate_rpn_raw(rpn_and_false_unk, state) == VAL_FALSE);

    // OR TESTS
    std::vector<int> rpn_or_true_unk = {0, 2, OP_OR};
    TEST_CHECK("OR(True, Unknown) == True", 
        Evaluator::evaluate_rpn_raw(rpn_or_true_unk, state) == VAL_TRUE);

    std::vector<int> rpn_or_false_unk = {1, 2, OP_OR};
    TEST_CHECK("OR(False, Unknown) == Unknown", 
        Evaluator::evaluate_rpn_raw(rpn_or_false_unk, state) == VAL_UNKNOWN);

    // NOT TESTS
    std::vector<int> rpn_not_unk = {2, OP_NOT};
    TEST_CHECK("NOT(Unknown) == Unknown", 
        Evaluator::evaluate_rpn_raw(rpn_not_unk, state) == VAL_UNKNOWN);
        
    std::vector<int> rpn_not_false = {1, OP_NOT};
    TEST_CHECK("NOT(False) == True", 
        Evaluator::evaluate_rpn_raw(rpn_not_false, state) == VAL_TRUE);
}

void test_evaluation_modes() {
    std::cout << "\n--- Running Evaluation Mode Tests ---\n";
    PartiallySpecifiedState state(10, 0);
    state.set_unknown(5); // F5 is Unknown

    std::vector<int> rpn_single_unk = {5};

    TEST_CHECK("PESSIMISTIC mode on Unknown returns False", 
        Evaluator::evaluate(rpn_single_unk, state, EvalMode::PESSIMISTIC) == false);

    TEST_CHECK("OPTIMISTIC mode on Unknown returns True", 
        Evaluator::evaluate(rpn_single_unk, state, EvalMode::OPTIMISTIC) == true);
}

void test_complex_state_integration() {
    std::cout << "\n--- Running State Integration Tests ---\n";
    PartiallySpecifiedState state(10, 0);
    
    state.set_known_value(3, true);   
    state.set_known_value(4, true);   
    state.set_unknown(5);             

    // Formula: (F3 AND F4) AND F5 => (True AND True) AND Unknown => Unknown
    std::vector<int> complex_rpn = {3, 4, OP_AND, 5, OP_AND};
    
    TEST_CHECK("Complex RPN resolves to Unknown", 
        Evaluator::evaluate_rpn_raw(complex_rpn, state) == VAL_UNKNOWN);
        
    // Mode application on complex formula
    TEST_CHECK("Complex RPN (Pessimistic) is False", 
        Evaluator::evaluate(complex_rpn, state, EvalMode::PESSIMISTIC) == false);
}

int main() {
    initialize_test_environment();
    
    test_three_valued_truth_tables();
    test_evaluation_modes();
    test_complex_state_integration();
    
    std::cout << "\nALL EVALUATOR TESTS PASSED SUCCESSFULLY.\n";
    return 0;
}