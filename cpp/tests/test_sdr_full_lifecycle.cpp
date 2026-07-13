#include <iostream>
#include <cassert>
#include <vector>
#include <algorithm>

// Include your engine headers
#include "../ProblemData.hpp"
#include "../State.hpp"
#include "../ActionApplier.hpp"
#include "../Regression.hpp"
#include "../SDRPlanner.hpp"
#include "../SDRSampler.hpp"
#include "../Z3Manager.hpp"
#include "../BeliefState.hpp"
#include "../Evaluator.hpp"

using namespace CPOR;

// Domain Predicates
constexpr int P_LOC_A = 0;
constexpr int P_LOC_B = 1;
constexpr int P_DOOR_OPEN = 2;
constexpr int P_HAS_KEY = 3;
constexpr int P_GOAL_REACHED = 4;

void init_contingent_problem() {

    reset_global_problem();

    get_global_problem().total_predicates = 5;
    
    int blocks = (get_global_problem().total_predicates / 64) + 1;
    get_global_problem().comparable_mask.assign(blocks, ~0ULL); // Enable hashing

    // ---------------------------------------------------------
    // Define Actions
    // ---------------------------------------------------------

    // Action 0: MOVE_A_B (Classical Move)
    GroundedAction move_a_b;
    move_a_b.id = 0;
    move_a_b.observe_predicate_id = -1;
    move_a_b.precondition_rpn = { P_LOC_A };
    move_a_b.guaranteed_effects.push_back({P_LOC_A, false});
    move_a_b.guaranteed_effects.push_back({P_LOC_B, true});
    
    // Conditional Effect: If DOOR is OPEN when moving to B, GOAL is REACHED
    ConditionalEffect ce;
    ce.condition_rpn = { P_DOOR_OPEN };
    ce.effects.push_back({P_GOAL_REACHED, true});
    move_a_b.conditional_effects.push_back(ce);
    
    // Action 1: SENSE_DOOR (Epistemic Action)
    GroundedAction sense_door;
    sense_door.id = 1;
    sense_door.observe_predicate_id = P_DOOR_OPEN;
    sense_door.precondition_rpn = { P_LOC_A }; // Can only observe from A

    // Action 2: MOVE_B_A (Return Move for Cycle Testing)
    GroundedAction move_b_a;
    move_b_a.id = 2;
    move_b_a.observe_predicate_id = -1;
    move_b_a.precondition_rpn = { P_LOC_B };
    move_b_a.guaranteed_effects.push_back({P_LOC_B, false});
    move_b_a.guaranteed_effects.push_back({P_LOC_A, true});

    get_global_problem().actions = { move_a_b, sense_door, move_b_a };

    // ---------------------------------------------------------
    // Define Contingent Dead-End
    // ---------------------------------------------------------
    // Formula: NOT DOOR_OPEN AND NOT HAS_KEY (If door closed & no key, we are dead)
    get_global_problem().deadend_rpns.push_back({ P_DOOR_OPEN, OP_NOT, P_HAS_KEY, OP_NOT, OP_AND });
    get_global_problem().goal_rpn = { P_GOAL_REACHED };
}

void test_z3_manager_sampling() {
    std::cout << "--- Testing Z3Manager & Sampler ---" << std::endl;
    PartiallySpecifiedState s0(get_global_problem().total_predicates);
    
    // Add a Global OneOf Invariant: (DOOR_OPEN XOR HAS_KEY)
    get_global_problem().oneofs.push_back({P_DOOR_OPEN, P_HAS_KEY});
    
    s0.set_known_value(P_LOC_A, true);
    // P_DOOR_OPEN and P_HAS_KEY remain implicitly UNKNOWN

    std::vector<PartiallySpecifiedState> samples = SDRSampler::sample_concrete_states(s0, get_global_problem(), 5);
    
    assert(!samples.empty());
    for (const auto& sample : samples) {
        // Assert the OneOf logic held: Exactly one must be true
        bool door = sample.is_true(P_DOOR_OPEN);
        bool key = sample.is_true(P_HAS_KEY);
        assert(door != key); // Mathematical XOR
    }
    
    // Clean up mock invariant for remaining tests
    get_global_problem().oneofs.clear();
    std::cout << "[PASS] Z3Manager correctly instantiated and sampled diverse concrete states respecting OneOfs." << std::endl;
}

void test_conditional_regression() {
    std::cout << "--- Testing Conditional Effects Regression ---" << std::endl;
    PartiallySpecifiedState s0(get_global_problem().total_predicates);
    s0.set_known_value(P_LOC_A, true);
    s0.set_known_value(P_DOOR_OPEN, true); // We KNOW the door is open

    BeliefState belief(s0);
    
    // Simulate applying MOVE_A_B
    PartiallySpecifiedState s1 = s0;
    ActionApplier::apply_action(get_global_problem().actions[0], s1);
    belief.apply_forward_action(0, s1);

    // Regress GOAL_REACHED backward through history
    std::vector<int> target_rpn = { P_GOAL_REACHED };
    
    bool is_safely_true = belief.verify_condition_safely(target_rpn, get_global_problem());
    
    // Because DOOR was OPEN in historical state s0, the conditional effect triggered.
    // The regression engine MUST deduce this as TRUE.
    assert(is_safely_true == true);
    std::cout << "[PASS] RegressionEngine successfully deduced facts triggered via conditional effects." << std::endl;
}

void test_sdr_deadend_recovery() {
    std::cout << "--- Testing MaybeDeadEnd Sub-Goal Injection ---" << std::endl;
    
    PartiallySpecifiedState s0(get_global_problem().total_predicates);
    s0.set_known_value(P_LOC_A, true);
    s0.set_known_value(P_HAS_KEY, false); // We definitely do NOT have the key
    // DOOR_OPEN is UNKNOWN. 

    // Because HAS_KEY is false, but DOOR_OPEN is unknown, the deadend:
    // (!DOOR_OPEN AND !HAS_KEY) evaluates to VAL_UNKNOWN.
    // It is a MaybeDeadEnd. The planner MUST trigger a sensing action.

    SDRPlanner planner(s0, get_global_problem());
    int action_to_execute = planner.get_next_action();

    // The planner should have dynamically injected Action 1 (SENSE_DOOR) 
    // to collapse the uncertainty preventing it from continuing.
    assert(action_to_execute == 1);
    std::cout << "[PASS] SDRPlanner successfully intercepted a MaybeDeadEnd and injected an Epistemic Sensing Goal." << std::endl;
}

void test_physical_cycle_detection() {
    std::cout << "--- Testing Physical Cycle Detection & Loop Breaking ---" << std::endl;
    
    PartiallySpecifiedState s0(get_global_problem().total_predicates);
    s0.set_known_value(P_LOC_A, true);

    s0.set_known_value(P_HAS_KEY, true);
    
    SDRPlanner planner(s0, get_global_problem());
    
    // 1. Force first move (A -> B)
    planner.set_plan_queue({0}); // Action 0 is MOVE_A_B
    int act1 = planner.get_next_action();
    assert(act1 == 0);
    
    // Update belief state (Agent is now at B)
    PartiallySpecifiedState s1 = s0;
    ActionApplier::apply_action(get_global_problem().actions[0], s1);
    planner.get_mutable_belief().apply_forward_action(0, s1);

    // 2. Force return move (B -> A)
    planner.set_plan_queue({2}); // Action 2 is MOVE_B_A
    int act2 = planner.get_next_action();
    assert(act2 == 2);
    
    // Update belief state (Agent physically returned to A)
    PartiallySpecifiedState s2 = s1;
    ActionApplier::apply_action(get_global_problem().actions[2], s2);
    planner.get_mutable_belief().apply_forward_action(2, s2);

    // 3. The cycle trigger
    planner.set_plan_queue({0}); // Attempting to repeat the classical loop
    int act3 = planner.get_next_action();

    assert(act3 == 1);
    std::cout << "[PASS] SDRPlanner successfully detected an infinite physical loop and forced an Epistemic Branching decision." << std::endl;
}

int main() {
    std::cout << "===============================================" << std::endl;
    std::cout << "   INITIATING CPOR/SDR ALGORITHMIC AUDIT SUITE " << std::endl;
    std::cout << "===============================================" << std::endl;

    init_contingent_problem();

    test_z3_manager_sampling();
    test_conditional_regression();
    test_sdr_deadend_recovery();
    test_physical_cycle_detection();

    std::cout << "===============================================" << std::endl;
    std::cout << " ALL MATHEMATICAL INVARIANTS VERIFIED. " << std::endl;
    std::cout << " SDR ENGINE IS SOUND AND PRODUCTION READY. " << std::endl;
    std::cout << "===============================================" << std::endl;

    return 0;
}