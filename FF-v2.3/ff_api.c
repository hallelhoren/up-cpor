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

    // 2. Reset plan sequence tracker. gplan_states[i].F buffers are allocated
    // once per problem load (in ff_load_problem) and reused across searches --
    // do NOT free them here. extract_plan_fragment/extract_plan (search.c)
    // write fresh F[]/num_F contents into gplan_states[gnum_plan_ops+1] on
    // every successful search via source_to_dest(), which assumes the
    // destination's F buffer is already allocated and does not check for
    // NULL; freeing it here without reallocating would leave a dangling
    // pointer that crashes the very next search.
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
 * Guards whether ff_load_problem() has successfully built the connectivity
 * graph. ff_search() must not run against a graph that was never loaded (or
 * whose load failed) -- gnum_ft_conn/gnum_op_conn/gnum_ef_conn would still be
 * zero-initialized, meaning every search would trivially "succeed" against an
 * empty problem instead of failing loudly.
 */
static int gff_problem_loaded = 0;

static void ff_free_previous_problem(void) {
    int i;
    if (!gff_problem_loaded) {
        return;
    }
    for (i = 0; i < gnum_op_conn; i++) {
        if (gop_conn[i].E) free(gop_conn[i].E);
    }
    free(gop_conn);
    for (i = 0; i < gnum_ef_conn; i++) {
        if (gef_conn[i].PC) free(gef_conn[i].PC);
        if (gef_conn[i].A) free(gef_conn[i].A);
        if (gef_conn[i].D) free(gef_conn[i].D);
    }
    free(gef_conn);
    for (i = 0; i < gnum_ft_conn; i++) {
        if (gft_conn[i].PC) free(gft_conn[i].PC);
        if (gft_conn[i].A) free(gft_conn[i].A);
        if (gft_conn[i].D) free(gft_conn[i].D);
        if (gft_conn[i].False) free(gft_conn[i].False);
    }
    free(gft_conn);
    if (ginitial_state.F) free(ginitial_state.F);
    if (ggoal_state.F) free(ggoal_state.F);
    /* gplan_states[]: see the allocation comment in ff_load_problem for why
     * this must exist at all -- FF's own plan-extraction code
     * (extract_plan_fragment/extract_plan in search.c) was written assuming
     * something upstream already allocated these buffers (true in FF's
     * original standalone-executable form, where a one-time startup
     * allocated them for the process's single problem) and never checks for
     * NULL before writing into them. */
    for (i = 0; i <= MAX_PLAN_LENGTH; i++) {
        if (gplan_states[i].F) {
            free(gplan_states[i].F);
            gplan_states[i].F = NULL;
        }
    }
    gff_problem_loaded = 0;
}

int ff_load_problem(
    int num_facts,
    int num_ops,
    int num_effects,
    const int* effect_op,
    const int* effect_pc_flat, const int* effect_pc_offsets, const int* effect_pc_counts,
    const int* effect_add_flat, const int* effect_add_offsets, const int* effect_add_counts,
    const int* effect_del_flat, const int* effect_del_offsets, const int* effect_del_counts,
    int num_goal_facts, const int* goal_facts
) {
    int i, j;

    ff_free_previous_problem();
    /* relax.c's build_fixpoint/extract_1P/collect_A_info/initialize_goals/
     * collect_H_info each lazily allocate their own working buffers on their
     * first-ever call in this thread, sized to whatever problem happened to
     * be loaded at that moment -- reused as-is (never resized) for any later,
     * differently-sized problem loaded into the same thread. Must be torn
     * down and re-armed on every new problem load, exactly like the
     * connectivity graph above. */
    relax_reset_for_new_problem();
    /* Same lazily-sized-on-first-call issue as relax.c above, but in
     * search.c's EHC/BFS driver functions and result_to_dest(). */
    search_reset_for_new_problem();

    /* --- Facts --- */
    gnum_ft_conn = num_facts;
    gft_conn = (FtConn*)calloc(num_facts > 0 ? (size_t)num_facts : 1, sizeof(FtConn));
    if (!gft_conn) return 0;
    for (i = 0; i < num_facts; i++) {
        gft_conn[i].rand = rand();
    }

    /* --- Effects (each already carries its owning op's precondition unioned
     * into PC by the caller) --- */
    gnum_ef_conn = num_effects;
    gef_conn = (EfConn*)calloc(num_effects > 0 ? (size_t)num_effects : 1, sizeof(EfConn));
    if (!gef_conn) return 0;

    for (i = 0; i < num_effects; i++) {
        int pc_off = effect_pc_offsets[i], pc_cnt = effect_pc_counts[i];
        int add_off = effect_add_offsets[i], add_cnt = effect_add_counts[i];
        int del_off = effect_del_offsets[i], del_cnt = effect_del_counts[i];

        gef_conn[i].op = effect_op[i];

        gef_conn[i].num_PC = pc_cnt;
        gef_conn[i].PC = pc_cnt > 0 ? (int*)malloc((size_t)pc_cnt * sizeof(int)) : NULL;
        for (j = 0; j < pc_cnt; j++) gef_conn[i].PC[j] = effect_pc_flat[pc_off + j];

        gef_conn[i].num_A = add_cnt;
        gef_conn[i].A = add_cnt > 0 ? (int*)malloc((size_t)add_cnt * sizeof(int)) : NULL;
        for (j = 0; j < add_cnt; j++) gef_conn[i].A[j] = effect_add_flat[add_off + j];

        gef_conn[i].num_D = del_cnt;
        gef_conn[i].D = del_cnt > 0 ? (int*)malloc((size_t)del_cnt * sizeof(int)) : NULL;
        for (j = 0; j < del_cnt; j++) gef_conn[i].D[j] = effect_del_flat[del_off + j];

        /* No implied-effects support (only used by an optional helper-action
         * refinement we don't rely on); level/in_E/num_active_PCs/ch are
         * relaxed-fixpoint working state, reset by relax.c's own first-call
         * initialization in build_fixpoint(). */
        gef_conn[i].I = NULL;
        gef_conn[i].num_I = 0;
    }

    /* --- Operators: derive OpConn.E (this op's effect indices) from effect_op --- */
    gnum_op_conn = num_ops;
    gop_conn = (OpConn*)calloc(num_ops > 0 ? (size_t)num_ops : 1, sizeof(OpConn));
    if (!gop_conn) return 0;
    {
        int *op_e_count = (int*)calloc(num_ops > 0 ? (size_t)num_ops : 1, sizeof(int));
        int *fill = (int*)calloc(num_ops > 0 ? (size_t)num_ops : 1, sizeof(int));
        for (i = 0; i < num_effects; i++) op_e_count[effect_op[i]]++;
        for (i = 0; i < num_ops; i++) {
            gop_conn[i].num_E = op_e_count[i];
            gop_conn[i].E = op_e_count[i] > 0 ? (int*)malloc((size_t)op_e_count[i] * sizeof(int)) : NULL;
        }
        for (i = 0; i < num_effects; i++) {
            int op = effect_op[i];
            gop_conn[op].E[fill[op]++] = i;
        }
        free(op_e_count);
        free(fill);
    }

    /* --- Reverse indices on facts: which effects gate on / add / delete each fact --- */
    {
        int *pc_count = (int*)calloc(num_facts > 0 ? (size_t)num_facts : 1, sizeof(int));
        int *a_count = (int*)calloc(num_facts > 0 ? (size_t)num_facts : 1, sizeof(int));
        int *d_count = (int*)calloc(num_facts > 0 ? (size_t)num_facts : 1, sizeof(int));
        int *pc_fill = (int*)calloc(num_facts > 0 ? (size_t)num_facts : 1, sizeof(int));
        int *a_fill = (int*)calloc(num_facts > 0 ? (size_t)num_facts : 1, sizeof(int));
        int *d_fill = (int*)calloc(num_facts > 0 ? (size_t)num_facts : 1, sizeof(int));

        for (i = 0; i < num_effects; i++) {
            for (j = 0; j < gef_conn[i].num_PC; j++) pc_count[gef_conn[i].PC[j]]++;
            for (j = 0; j < gef_conn[i].num_A; j++) a_count[gef_conn[i].A[j]]++;
            for (j = 0; j < gef_conn[i].num_D; j++) d_count[gef_conn[i].D[j]]++;
        }
        for (i = 0; i < num_facts; i++) {
            gft_conn[i].num_PC = pc_count[i];
            gft_conn[i].PC = pc_count[i] > 0 ? (int*)malloc((size_t)pc_count[i] * sizeof(int)) : NULL;
            gft_conn[i].num_A = a_count[i];
            gft_conn[i].A = a_count[i] > 0 ? (int*)malloc((size_t)a_count[i] * sizeof(int)) : NULL;
            gft_conn[i].num_D = d_count[i];
            gft_conn[i].D = d_count[i] > 0 ? (int*)malloc((size_t)d_count[i] * sizeof(int)) : NULL;
        }
        for (i = 0; i < num_effects; i++) {
            for (j = 0; j < gef_conn[i].num_PC; j++) {
                int ft = gef_conn[i].PC[j];
                gft_conn[ft].PC[pc_fill[ft]++] = i;
            }
            for (j = 0; j < gef_conn[i].num_A; j++) {
                int ft = gef_conn[i].A[j];
                gft_conn[ft].A[a_fill[ft]++] = i;
            }
            for (j = 0; j < gef_conn[i].num_D; j++) {
                int ft = gef_conn[i].D[j];
                gft_conn[ft].D[d_fill[ft]++] = i;
            }
        }
        free(pc_count); free(a_count); free(d_count);
        free(pc_fill); free(a_fill); free(d_fill);
    }

    /* --- Goal --- */
    make_state(&ggoal_state, num_facts > 0 ? num_facts : 1);
    for (i = 0; i < num_goal_facts; i++) {
        ggoal_state.F[i] = goal_facts[i];
        gft_conn[goal_facts[i]].is_global_goal = TRUE;
    }
    ggoal_state.num_F = num_goal_facts;
    ggoal_state.max_F = num_facts;

    /* --- Initial-state scratch buffer, filled fresh on every ff_search() call --- */
    make_state(&ginitial_state, num_facts > 0 ? num_facts : 1);
    ginitial_state.max_F = num_facts;

    /* --- Plan-extraction scratch buffers. FF's own extract_plan_fragment/
     * extract_plan (search.c) write into gplan_states[gnum_plan_ops+1] via
     * source_to_dest(), which unconditionally does dest->F[i] = source->F[i]
     * with no NULL check -- these must be allocated before the first search,
     * not left as zero-initialized static storage. */
    for (i = 0; i <= MAX_PLAN_LENGTH; i++) {
        make_state(&gplan_states[i], num_facts > 0 ? num_facts : 1);
        gplan_states[i].max_F = num_facts;
    }

    gff_problem_loaded = 1;
    return 1;
}

/*
 * Native entry point for the C++ engine to trigger a planning turn.
 * Requires ff_load_problem() to have already built the connectivity graph --
 * make_search_space() used to be responsible for this and was an empty stub,
 * silently leaving FF searching an all-zero (nonexistent) problem.
 */
int* ff_search(const uint64_t* determinized_state_bitset, int* out_plan_length) {
    if (!determinized_state_bitset || !out_plan_length || !gff_problem_loaded) {
        return NULL;
    }

    // 1. Initialize FF Initial State from C++ Bitset
    ginitial_state.num_F = 0;
    for (int i = 0; i < gnum_ft_conn; ++i) {
        if (determinized_state_bitset[i / 64] & (1ULL << (i % 64))) {
            ginitial_state.F[ginitial_state.num_F++] = i;
        }
    }

    // 2. Perform Planning. The goal is ggoal_state (populated once by
    // ff_load_problem), not gplan_states -- gplan_states is do_enforced_hill_climbing's
    // *output* plan-state-sequence scratch buffer, not a goal description; passing
    // it as the goal argument (the previous code here) made every search see an
    // empty goal and report instant, spurious success.
    Bool plan_found = do_enforced_hill_climbing(&ginitial_state, &ggoal_state);
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