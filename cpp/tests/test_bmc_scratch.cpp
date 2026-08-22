#include <iostream>
#include <cassert>
#include "../ProblemData.hpp"
#include "../State.hpp"
#include "../ActionApplier.hpp"
#include "../Z3BMCManager.hpp"

using namespace CPOR;

// Mirrors test_regression_scratch.cpp's toy domain EXACTLY (3 positions, a `checking`
// action whose conditional effects on shared facts FREE_A/FREE_B are gated on
// (not ok) AND at(X) for each position, and a sense action observing FREE_A) -- so the
// same expected outcomes can be checked, this time via the tree-indexed,
// incrementally-cached Z3BMCManager instead of RegressionEngine::regress_rpn +
// verify_condition_safely. If both give the same answers here, the BMC encoding (and
// its incremental node-to-node caching) is a faithful, validated alternative for this
// shape of query, before touching the real (and far larger) localize5 domain.
constexpr int P_OK = 0;
constexpr int P_AT1 = 1;
constexpr int P_AT2 = 2;
constexpr int P_AT3 = 3;
constexpr int P_FREE_A = 4;
constexpr int P_FREE_B = 5;

int main() {
    reset_global_problem();
    get_global_problem().total_predicates = 6;
    int blocks = (get_global_problem().total_predicates / 64) + 1;
    get_global_problem().comparable_mask.assign(blocks, ~0ULL);
    get_global_problem().oneofs.push_back({P_AT1, P_AT2, P_AT3});

    GroundedAction checking;
    checking.id = 0;
    checking.observe_predicate_id = -1;
    checking.guaranteed_effects.push_back({P_OK, true});
    {
        ConditionalEffect ce;
        ce.condition_rpn = {P_OK, OP_NOT, P_AT1, OP_AND};
        ce.effects.push_back({P_FREE_A, true});
        ce.effects.push_back({P_FREE_B, false});
        checking.conditional_effects.push_back(ce);
    }
    {
        ConditionalEffect ce;
        ce.condition_rpn = {P_OK, OP_NOT, P_AT2, OP_AND};
        ce.effects.push_back({P_FREE_A, true});
        ce.effects.push_back({P_FREE_B, false});
        checking.conditional_effects.push_back(ce);
    }
    {
        ConditionalEffect ce;
        ce.condition_rpn = {P_OK, OP_NOT, P_AT3, OP_AND};
        ce.effects.push_back({P_FREE_A, false});
        ce.effects.push_back({P_FREE_B, true});
        checking.conditional_effects.push_back(ce);
    }

    GroundedAction sense_a;
    sense_a.id = 1;
    sense_a.observe_predicate_id = P_FREE_A;

    get_global_problem().actions = {checking, sense_a};

    // Fully unknown initial state (position never known), EXCEPT ok, which starts
    // known-false -- matches the real localize5 domain's actual initial state.
    PartiallySpecifiedState s0(get_global_problem().total_predicates);
    s0.set_known_value(P_OK, false);

    // node 0 = root, node 1 = after checking, node 2 = after sense_a observing TRUE.
    PartiallySpecifiedState s1 = s0;
    ActionApplier::apply_action(checking, s1);
    assert(s1.is_true(P_OK));
    assert(s1.is_unknown(P_FREE_A)); // ambiguous across P1/P2/P3 -- ActionApplier itself can't resolve this

    PartiallySpecifiedState s2 = s1;
    s2.set_known_value(P_FREE_A, true); // genuine sensing observation

    Z3BMCManager bmc(get_global_problem().total_predicates);
    bmc.build_root(0, s0, get_global_problem());
    bmc.build_child(1, 0, checking, s1, get_global_problem());
    bmc.build_child(2, 1, sense_a, s2, get_global_problem());

    // is_impossible/extend_and_sample now require the caller to first sync the
    // persistent path solver to the node being queried via enter_node/exit_node
    // (mirrors solve_from_node's own RAII-guarded usage in CPORSolver.cpp) --
    // parent_idx only affects whether enter_node takes its fast or slow path,
    // never correctness, so it's safe to switch between arbitrary nodes here.

    // The core claim: BMC should PROVE at(P3) is false at node 2 (P3 is the only
    // position whose signature predicts FREE_A=false, contradicting the observation),
    // even though nothing ever directly wrote to P_AT3's bits.
    bmc.enter_node(2, 1);
    bool at3_impossible = bmc.is_impossible(2, P_AT3, true);
    std::cout << "is_impossible(at(P3)=true) = " << (at3_impossible ? "TRUE" : "false") << std::endl;
    assert(at3_impossible && "Expected BMC to prove position is NOT P3");

    // And it should NOT claim to know P1 specifically (still genuinely ambiguous
    // between P1 and P2).
    bool at1_true_impossible = bmc.is_impossible(2, P_AT1, true);
    bool at1_false_impossible = bmc.is_impossible(2, P_AT1, false);
    std::cout << "is_impossible(at(P1)=true) = " << (at1_true_impossible ? "TRUE" : "false") << std::endl;
    std::cout << "is_impossible(at(P1)=false) = " << (at1_false_impossible ? "TRUE" : "false") << std::endl;
    assert(!at1_true_impossible && !at1_false_impossible && "P1 vs P2 should remain genuinely ambiguous");

    // Sanity: FREE_A's own observed value must of course be recoverable.
    bool free_a_false_impossible = bmc.is_impossible(2, P_FREE_A, false);
    std::cout << "is_impossible(FREE_A=false) = " << (free_a_false_impossible ? "TRUE" : "false") << std::endl;
    assert(free_a_false_impossible && "Directly observed fact must be recoverable");
    bmc.exit_node();

    // Node 1 (BEFORE the sense_a observation) must NOT have this knowledge -- confirms
    // each node's cached assertions reflect only ITS OWN history, not a later
    // sibling/descendant's, and that node 1's cache wasn't retroactively mutated by
    // building node 2 afterward (append_transition/append_oneofs on node 2 must only
    // ever have appended to a COPY of node 1's list, never node 1's own stored copy).
    bmc.enter_node(1, 0);
    bool at3_impossible_at_node1 = bmc.is_impossible(1, P_AT3, true);
    std::cout << "is_impossible(at(P3)=true) at node 1 (pre-observation) = "
              << (at3_impossible_at_node1 ? "TRUE (unexpected)" : "false (expected)") << std::endl;
    assert(!at3_impossible_at_node1 && "Node 1 must not see node 2's later observation");
    bmc.exit_node();

    std::cout << "[PASS] Z3BMCManager correctly narrows a multi-conditional-effect "
                 "disjunction via a oneof invariant, without over-claiming on the still-ambiguous "
                 "residue, matching test_regression_scratch's own expected outcomes, and keeps "
                 "each node's own cached history independent of its descendants'."
              << std::endl;

    // extend_and_sample: the rescue path's mechanism. A hypothetical action that
    // unconditionally forces at(P3)=true should be recognized as IMPOSSIBLE (nullopt)
    // given node 2's history (which already ruled P3 out via the FREE_A observation).
    GroundedAction confirm_p3;
    confirm_p3.id = 2;
    confirm_p3.observe_predicate_id = -1;
    confirm_p3.guaranteed_effects.push_back({P_AT3, true});
    PartiallySpecifiedState s3_impossible = s2;
    s3_impossible.set_known_value(P_AT3, true);
    bmc.enter_node(2, 1);
    auto dead_sample = bmc.extend_and_sample(2, confirm_p3, s3_impossible, get_global_problem());
    std::cout << "extend_and_sample(node 2, confirm_p3) = " << (dead_sample.has_value() ? "SAT (unexpected)" : "UNSAT (nullopt)") << std::endl;
    assert(!dead_sample.has_value() && "Forcing at(P3)=true should be impossible given the FREE_A observation");

    // A hypothetical action that unconditionally forces at(P1)=true (consistent with
    // history, since P1 is still a live candidate) should succeed with a fully
    // determinized, history-consistent sample.
    GroundedAction confirm_p1;
    confirm_p1.id = 3;
    confirm_p1.observe_predicate_id = -1;
    confirm_p1.guaranteed_effects.push_back({P_AT1, true});
    PartiallySpecifiedState s3_possible = s2;
    s3_possible.set_known_value(P_AT1, true);
    auto live_sample = bmc.extend_and_sample(2, confirm_p1, s3_possible, get_global_problem());
    std::cout << "extend_and_sample(node 2, confirm_p1) = " << (live_sample.has_value() ? "SAT" : "UNSAT (unexpected)") << std::endl;
    assert(live_sample.has_value() && "Forcing at(P1)=true should be consistent with history");
    assert(live_sample->is_true(P_AT1) && live_sample->is_false(P_AT2) && live_sample->is_false(P_AT3));
    assert(live_sample->is_true(P_FREE_A) && "Sample must respect the FREE_A=true observation too");

    // Node 2's own cached encoding must be untouched by both extend_and_sample calls
    // above (both purely transient, scoped to their own throwaway solver) -- re-check
    // the original claim still holds.
    assert(bmc.is_impossible(2, P_AT3, true) && "node 2's own cached encoding must survive extend_and_sample calls unmodified");
    bmc.exit_node();

    std::cout << "[PASS] Z3BMCManager::extend_and_sample correctly extends a cached node's "
                 "history by one action, recognizing an impossible resulting state as such "
                 "(nullopt) and returning a fully determinized, history-consistent sample for "
                 "a live one, without mutating the node's own cached encoding."
              << std::endl;

    // Incremental caching sanity: a THIRD child of node 1 (a sibling of node 2, taking
    // a DIFFERENT branch -- e.g. observing FREE_A=false instead) must independently
    // reflect ITS OWN observation, proving P1/P2 are the impossible ones instead (only
    // P3 predicts FREE_A=false) -- and this works by extending node 1's cached
    // assertions again, completely independently of node 2's own extension, exactly
    // the tree-branching case linear time-indexing couldn't handle at all.
    PartiallySpecifiedState s2b = s1;
    s2b.set_known_value(P_FREE_A, false);
    bmc.build_child(3, 1, sense_a, s2b, get_global_problem());
    bmc.enter_node(3, 1);
    bool p1_impossible_at_node3 = bmc.is_impossible(3, P_AT1, true);
    bool p2_impossible_at_node3 = bmc.is_impossible(3, P_AT2, true);
    bool p3_impossible_at_node3 = bmc.is_impossible(3, P_AT3, true);
    std::cout << "node 3 (sibling, FREE_A=false): is_impossible(at(P1))=" << p1_impossible_at_node3
              << " is_impossible(at(P2))=" << p2_impossible_at_node3
              << " is_impossible(at(P3))=" << p3_impossible_at_node3 << std::endl;
    assert(p1_impossible_at_node3 && p2_impossible_at_node3 && !p3_impossible_at_node3
           && "Node 3's sibling branch should independently prove position IS P3");
    bmc.exit_node();
    // And node 2 (the other sibling) must still show its own, opposite conclusion.
    bmc.enter_node(2, 1);
    assert(bmc.is_impossible(2, P_AT3, true) && "Node 2's own conclusion must be unaffected by node 3's later, independent extension");
    bmc.exit_node();

    std::cout << "[PASS] Sibling nodes extend their shared parent's cached history "
                 "completely independently, each reflecting only its own branch's "
                 "observation -- validates the incremental TREE caching, not just a "
                 "single linear chain."
              << std::endl;
    return 0;
}
