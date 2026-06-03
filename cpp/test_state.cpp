#include <iostream>
#include <cassert>
#include "State.hpp"

void run_state_tests() {
    std::cout << "Running State.hpp Tests..." << std::endl;

    // 1. Initialize a state with 100 possible predicates
    PartiallySpecifiedState state(100);

    // TEST 1: Everything should start as Unknown
    assert(state.is_unknown(5) == true);
    assert(state.is_unknown(99) == true);

    // TEST 2: Setting a value to TRUE
    state.set_known_value(5, true);
    assert(state.is_unknown(5) == false);
    assert(state.is_true(5) == true);

    // TEST 3: Setting a value to FALSE
    state.set_known_value(12, false);
    assert(state.is_unknown(12) == false);
    assert(state.is_true(12) == false); // It is known, but it is False

    // TEST 4: Bitwise separation across 64-bit boundaries
    // Predicate 70 is in the second 64-bit block (70 > 64)
    state.set_known_value(70, true);
    assert(state.is_true(70) == true);
    assert(state.is_unknown(70) == false);

    // TEST 5: State Equality (Comparing two universes)
    PartiallySpecifiedState state2(100);
    state2.set_known_value(5, true);
    state2.set_known_value(12, false);
    state2.set_known_value(70, true);
    
    assert(state == state2); // They should be identical

    std::cout << "SUCCESS: All State.hpp tests passed!" << std::endl;
}

int main() {
    run_state_tests();
    return 0;
}