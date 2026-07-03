#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Global Search State Reset
 * Must be called before EVERY search turn to prevent memory leaks and corruption.
 */
void ff_reset_search_state();

/*
 * Native entry point for the FF planner.
 * Bypasses PDDL and reads the state directly from the C++ bitset.
 */
int* ff_search(const uint64_t* determinized_state_bitset, int* out_plan_length);

#ifdef __cplusplus
}
#endif