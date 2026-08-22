#include <iostream>
#include <cassert>
#include "../ProblemData.hpp"
#include "../State.hpp"
#include "../ActionApplier.hpp"
#include "../Regression.hpp"
#include "../BeliefState.hpp"
#include "../Evaluator.hpp"

using namespace CPOR;

// Mirrors localize5's structure at toy scale: 3 positions (P1, P2, P3) in a
// oneof, a `checking` action whose conditional effects on shared facts
// FREE_A/FREE_B are gated on (not ok) AND at(X) for each position, and a
// sense action observing FREE_A.
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
    // P1 and P2 share the SAME effect signature on FREE_A/FREE_B (ambiguous
    // if both remain candidates); P3 differs, so observing FREE_A=true
    // should let us rule OUT P3 specifically, narrowing the oneof.
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

    // Fully unknown initial state (position never known -- mirrors localize5),
    // EXCEPT ok, which starts known-false: localize5's own p.pddl never
    // mentions `ok` in :init (and it isn't part of any oneof/uncertainty
    // declaration), matching the closed-world default for an un-mentioned
    // plain fluent. Without this, `ok`'s pre-first-checking value is
    // genuinely free in Z3's model, and "ok happened to already be true"
    // makes every one of checking's (not ok)-gated conditional effects
    // vacuously not-fire -- a logically available escape hatch that isn't
    // a flaw in the regression/entailment machinery itself.
    PartiallySpecifiedState s0(get_global_problem().total_predicates);
    s0.set_known_value(P_OK, false);
    BeliefState belief(s0);

    // Apply `checking`.
    PartiallySpecifiedState s1 = s0;
    ActionApplier::apply_action(checking, s1);
    assert(s1.is_true(P_OK));
    assert(s1.is_unknown(P_FREE_A)); // ambiguous across P1/P2/P3 -- correctly dropped provenance
    belief.apply_forward_action(0, s1);

    // Observe FREE_A = true (a genuine sensing observation).
    PartiallySpecifiedState s2 = s1;
    s2.set_known_value(P_FREE_A, true);
    belief.apply_forward_action(1, s2);

    // The core claim: regression + Z3 entailment should now PROVE at(P3) is
    // false (P3 is the only position whose signature predicts FREE_A=false,
    // contradicting the observation), even though nothing ever directly
    // wrote to P_AT3's bits, and even though P1 vs P2 remains genuinely
    // ambiguous (both predict FREE_A=true).
    std::vector<int> not_at3 = RegressionEngine::negate_rpn({P_AT3});
    bool proven_not_at3 = belief.verify_condition_safely(not_at3, get_global_problem());
    std::cout << "verify_condition_safely(NOT at(P3)) = " << (proven_not_at3 ? "TRUE" : "false") << std::endl;
    assert(proven_not_at3 && "Expected regression+Z3 to prove position is NOT P3");

    // And it should NOT claim to know P1 specifically (still genuinely
    // ambiguous between P1 and P2) -- neither at(P1) nor its negation should
    // be provable.
    bool proven_at1 = belief.verify_condition_safely({P_AT1}, get_global_problem());
    bool proven_not_at1 = belief.verify_condition_safely(RegressionEngine::negate_rpn({P_AT1}), get_global_problem());
    std::cout << "verify_condition_safely(at(P1)) = " << (proven_at1 ? "TRUE" : "false") << std::endl;
    std::cout << "verify_condition_safely(NOT at(P1)) = " << (proven_not_at1 ? "TRUE" : "false") << std::endl;
    assert(!proven_at1 && !proven_not_at1 && "P1 vs P2 should remain genuinely ambiguous");

    std::cout << "[PASS] Regression + Z3 entailment correctly narrows a multi-conditional-effect "
                 "disjunction via a oneof invariant, without over-claiming on the still-ambiguous residue."
              << std::endl;
    return 0;
}
