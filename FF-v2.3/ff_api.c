/*********************************************************************
 * FF Planner C-ABI Integration Layer
 * Embedded version of Fast-Forward (FF) for high-performance C++20 engines.
 *********************************************************************/
#include "ff_api.h"
#include "ff.h"
#include "search.h"
#include "memory.h"
#include "output.h"
#include "parse.h"
#include "inst_pre.h"
#include "inst_easy.h"
#include "inst_hard.h"
#include "inst_final.h"
#include "relax.h"
#include "search.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Memory Reclamation Routine
 * Safely frees all dynamically allocated memory within the FF state hash table.
 * Essential for preventing memory leaks in an online planning loop.
 */


extern int gevaluated_states;

/*
 * Global Search State Reset
 * Must be called before EVERY search turn in your C++ planner.
 */
void ff_reset_search_state() {
    // 1. Perform deep memory cleanup of the hash table
    ff_clear_hash_table();

    // 2. Free dynamically allocated facts arrays inside the plan sequence!
    // We must free the inner arrays before we reset the counter.
    for (int i = 0; i < gnum_plan_ops; i++) {
        if (gplan_states[i].F != NULL) {
            free(gplan_states[i].F);
            gplan_states[i].F = NULL;
        }
    }

    // 3. Reset plan sequence tracker
    gnum_plan_ops = 0;

    // 4. Reset RPG connectivity flags for Facts
    for (int i = 0; i < gnum_ft_conn; i++) {
        gft_conn[i].is_true = 0;
    }

    // 5. Reset RPG connectivity flags for Actions/Operators
    for (int i = 0; i < gnum_op_conn; i++) {
        gop_conn[i].is_in_A = 0;
        gop_conn[i].is_in_H = 0;
        gop_conn[i].is_used = 0;
    }

    // 6. Reset search counters
    gevaluated_states = 0;

}

/*
 * Native entry point for the C++ engine to trigger a planning turn.
 */
int* ff_search(const uint64_t* determinized_state_bitset, int* out_plan_length) {
    if (!determinized_state_bitset || !out_plan_length) {
        return NULL;
    }

    // 1. Initialize FF Initial State from C++ Bitset
    ginitial_state.num_F = 0;
    for (int i = 0; i < gnum_ft_conn; ++i) {
        if (determinized_state_bitset[i / 64] & (1ULL << (i % 64))) {
            ginitial_state.F[ginitial_state.num_F++] = i;
        }
    }

    // 2. Prepare search space
    make_search_space();

    // 3. Perform Planning
    Bool plan_found = do_enforced_hill_climbing(&ginitial_state, &gplan_states);
    if (!plan_found) {
        plan_found = do_best_first_search();
    }

    // 4. Handle Failure
    if (!plan_found) {
        *out_plan_length = -1;
        return NULL;
    }

    // 5. Success: Goal reached in initial state
    if (gnum_plan_ops == 0) {
        *out_plan_length = 0;
        return NULL;
    }

    // 6. Return plan to C++ Engine
    int* plan_array = (int*)malloc(gnum_plan_ops * sizeof(int));
    if (!plan_array) {
        *out_plan_length = -1;
        return NULL;
    }

    for (int i = 0; i < gnum_plan_ops; ++i) {
        plan_array[i] = gplan_ops[i];
    }

    *out_plan_length = gnum_plan_ops;
    return plan_array;
}

#ifdef __cplusplus
}
#endif