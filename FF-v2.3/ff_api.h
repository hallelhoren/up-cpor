#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

extern void make_search_space(void);

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

/*
 * Loads a fully-grounded, STRIPS-normal-form problem into FF's connectivity
 * graph (gop_conn/gef_conn/gft_conn), bypassing FF's own PDDL parsing and
 * instantiation pipeline entirely -- the caller has already grounded the
 * problem (our C++ engine's RPN-based representation, translated down to plain
 * conjunctions of positive fact ids by the caller; negative literals must
 * already be compiled into dedicated "not-P" shadow facts by the caller).
 *
 * Each effect's PC (precondition) list must already be the union of its
 * owning operator's base precondition and the effect's own condition (empty
 * beyond the base precondition for an unconditional effect) -- that union is
 * what FF's relaxed fixpoint and result_to_dest actually gate on, per-effect,
 * not per-operator. Every operator must contribute at least one effect (with
 * empty add/delete lists if it has no guaranteed effects) so its base
 * precondition alone is still registered and the operator can be recognized
 * as applicable.
 *
 * Called once per problem, not once per search -- the connectivity graph
 * itself doesn't depend on the current state, only on the action/effect/goal
 * structure. Safe to call again for a new problem: frees any previously
 * loaded graph first.
 *
 * All flat_* arrays use the standard offset/count-per-item flattening: item i's
 * slice is flat[offsets[i] .. offsets[i]+counts[i]).
 *
 * Returns 0 if allocation fails, 1 on success.
 */
int ff_load_problem(
    int num_facts,
    int num_ops,
    int num_effects,
    const int* effect_op,
    const int* effect_pc_flat, const int* effect_pc_offsets, const int* effect_pc_counts,
    const int* effect_add_flat, const int* effect_add_offsets, const int* effect_add_counts,
    const int* effect_del_flat, const int* effect_del_offsets, const int* effect_del_counts,
    int num_goal_facts, const int* goal_facts
);

/*
 * Cheap heuristic-only entry point: computes h(state) via a SINGLE relaxed-
 * planning-graph fixpoint build (relax.c's get_1P_and_H), the same primitive
 * FF's own enforced-hill-climbing driver calls internally at every node it
 * visits -- without running do_enforced_hill_climbing/do_best_first_search's
 * full real search out to the goal the way ff_search() does. Also returns
 * FF's own "helpful actions" set: the applicable, zero-cost-in-the-relaxation
 * operators achieving a level-1 subgoal, exactly what EHC itself uses to
 * avoid ever expanding every applicable action from a state.
 *
 * Safe to call repeatedly with no ff_reset_search_state() in between --
 * get_1P_and_H is self-contained (build_fixpoint/extract_1P/reset_fixpoint
 * all run within this one call; collect_H_info clears its own previous
 * is_in_H flags at its own start) -- exactly how FF's own EHC driver already
 * calls it back-to-back in a tight loop with no reset between calls.
 *
 * out_helpful_actions must have room for at least max_helpful_actions ints;
 * *out_num_helpful is set to how many were actually written (<= gnum_op_conn,
 * so sizing the buffer to the problem's own action count is always enough).
 * Returns h(state), or -1 (FF's own INFINITY sentinel) if the goal is
 * unreachable from state in the delete-relaxed problem.
 */
int ff_estimate_heuristic(const uint64_t* determinized_state_bitset,
                           int* out_helpful_actions, int max_helpful_actions,
                           int* out_num_helpful);

#ifdef __cplusplus
}
#endif