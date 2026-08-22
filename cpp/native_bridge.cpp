#include "ProblemData.hpp"
#include "State.hpp"
#include "Evaluator.hpp"
#include "CPORSolver.hpp"
#include "BFSSolver.hpp"
#include "FFBridge.hpp"
#include "SDRSampler.hpp"
#include "DeadEndManager.hpp"
#include "DebugStats.hpp"
#include <vector>
#include <iostream>
#include <unordered_set>
#include <stdint.h>

extern "C" {

    void init_problem(int total_predicates, int total_functions = 0) {

        reset_global_problem();
        // A thread may have already solved a different problem before this
        // call (e.g. sequential test cases in one process) -- the SDR
        // sampler's thread-local Z3 state was built for that old problem's
        // predicate count and oneof/deadend constraints, so it must be
        // discarded before this new problem's facts/actions are loaded.
        CPOR::SDRSampler::reset_z3_state();
        // dead_end_cache is keyed purely on state bitmask content with no
        // problem identity -- see the comment on reset_for_new_problem() --
        // so it must be invalidated alongside the Z3 state above.
        CPOR::DeadEndManager::reset_for_new_problem();

        get_global_problem().total_predicates = total_predicates;
        get_global_problem().total_functions = total_functions;

        // Initialize the comparable_mask with all 1s (all variables comparable by default)
        // Python can later zero-out the bits for constants/options via a new API call if needed.
        int blocks = (total_predicates / 64) + 1;
        get_global_problem().comparable_mask.assign(blocks, ~0ULL);
    }

    void add_initial_fact(int fact_id) {
        get_global_problem().initial_true_facts.push_back(fact_id);
    }

    // Sets ProblemDef::is_simple (see its own doc comment in ProblemData.hpp).
    // The Python grounder is the only real caller: it computes this exactly
    // like the legacy C# engine's Domain.IsSimple (false iff any action has a
    // conditional effect) and calls this once per problem load, after
    // init_problem() (which resets is_simple to its safe default, false).
    void set_problem_is_simple(bool is_simple) {
        get_global_problem().is_simple = is_simple;
    }

    void add_initial_function_value(int func_id, double value) {
        // We store it as a pair to be applied during state construction
        get_global_problem().initial_function_values.push_back({func_id, value});
    }

    void set_goal_rpn(int* rpn_array, int length) {
        get_global_problem().goal_rpn.assign(rpn_array, rpn_array + length);
    }

    // Renamed legacy function to populate guaranteed effects
    void create_action(int id, int cost, int* pre_rpn, int pre_len, int obs_id) {
        // Ensure vector capacity to prevent out-of-bounds segfaults
        if (get_global_problem().actions.size() <= static_cast<size_t>(id)) {
            get_global_problem().actions.resize(id + 1);
        }
        
        auto& act = get_global_problem().actions[id];
        act.id = id;
        act.cost = cost;
        act.precondition_rpn.assign(pre_rpn, pre_rpn + pre_len);
        act.observe_predicate_id = obs_id;
    }

    void add_guaranteed_effect(int action_id, int fluent_id, bool val) {
        get_global_problem().actions[action_id].guaranteed_effects.push_back({fluent_id, val});
    }

    void add_conditional_effect(int action_id, int* cond_rpn, int cond_len, int fluent_id, bool val) {
        // Creates a localized RPN block strictly tied to this specific fluent assignment
        ConditionalEffect ce;
        ce.condition_rpn.assign(cond_rpn, cond_rpn + cond_len);
        ce.effects.push_back({fluent_id, val});
        
        get_global_problem().actions[action_id].conditional_effects.push_back(ce);
    }

    void add_nondeterministic_effect(int action_id, int fluent_id) {
        get_global_problem().actions[action_id].non_deterministic_effects.push_back(fluent_id);
    }

    void add_oneof_constraint(int* ids_array, int length) {
        std::vector<int> oneof_vec(ids_array, ids_array + length);
        get_global_problem().oneofs.push_back(oneof_vec);
    }

    //API to load a dead-end formula into the problem definition
    void add_deadend_rpn(int* rpn_array, int length) {
        if (length > 0) {
            std::vector<int> rpn(rpn_array, rpn_array + length);
            get_global_problem().deadend_rpns.push_back(std::move(rpn));
        }
    }

    void add_initial_false_fact(int fact_id) {
        get_global_problem().initial_false_facts.push_back(fact_id);
    }

    void print_problem_stats() {
        std::cout << "=== C++ Native Memory Verification ===" << std::endl;
        std::cout << "Total Predicates: " << get_global_problem().total_predicates << std::endl;
        std::cout << "Initial Facts: " << get_global_problem().initial_true_facts.size()
                  << " true, " << get_global_problem().initial_false_facts.size() << " false" << std::endl;
        std::cout << "Actions Loaded: " << get_global_problem().actions.size() << std::endl;
        std::cout << "======================================" << std::endl;
    }

    void add_oneof_group(int* fact_ids, int length) {
        std::vector<int> group;
        for (int i = 0; i < length; ++i) {
            group.push_back(fact_ids[i]);
        }
        get_global_problem().oneofs.push_back(group);
    }

    void add_conditional_effect_to_action(int action_id, 
                                          int* cond_rpn, int cond_len, 
                                          int* eff_facts, uint8_t* eff_vals, int eff_len) {
        
        // Find the action (assuming chronological insertion for speed)
        for (auto& action : get_global_problem().actions) {
            if (action.id == action_id) {
                ConditionalEffect ce;
                for (int i = 0; i < cond_len; ++i) ce.condition_rpn.push_back(cond_rpn[i]);
                for (int i = 0; i < eff_len; ++i) ce.effects.push_back({eff_facts[i], eff_vals[i] != 0});

                action.conditional_effects.push_back(ce);
                return;
            }
        }
        // action_id must already exist (create_action/add_grounded_action_to_cpp
        // is always called before any of its conditional effects are added --
        // see native_api.py's load_problem_to_cpp). Silently doing nothing here
        // would drop the conditional effect without a trace, corrupting the
        // action's semantics for a reason invisible from either side of the
        // ABI. Surface it the same way other grounding-contract violations in
        // this file already do.
        std::cerr << "CRITICAL: add_conditional_effect_to_action called for unknown action_id "
                  << action_id << " -- conditional effect dropped." << std::endl;
    }


// =====================================================================
    // GLOBAL SOLVER (Persists in memory so Python can read the plan later)
    // =====================================================================
    CPOR::CPORSolver* global_solver = nullptr;

    // Shared by solve_native() and solve_native_cpor_loop(): both algorithms
    // must start from an identically-constructed initial belief for a
    // meaningful pytest comparison between them, so this is factored out
    // rather than duplicated (unlike solve_poc_bfs's simpler, fully-observable
    // variant below, which is deliberately a different, classical-only setup).
    static PartiallySpecifiedState build_owa_initial_state() {
        PartiallySpecifiedState initial_state(get_global_problem().total_predicates, get_global_problem().total_functions);

        // --- OWA INITIALIZATION: Everything is UNKNOWN by default ---
        int blocks = (get_global_problem().total_predicates / 64) + 1;
        initial_state.known_mask.assign(blocks, 0ULL);
        initial_state.value_mask.assign(blocks, 0ULL);
        // --------------------------------------------------

        // 1. EXPLICITLY TRUE FACTS
        for (int fact_id : get_global_problem().initial_true_facts) {
            initial_state.set_known_value(fact_id, true);
        }

        // 2. EXPLICITLY FALSE FACTS
        for (int fact_id : get_global_problem().initial_false_facts) {
            initial_state.set_known_value(fact_id, false);
        }

        // 3. ONEOF CONSTRAINTS (Also sets facts to unknown, pending deductions)
        for (const auto& group : get_global_problem().oneofs) {
            for (int fact_id : group) {
                initial_state.set_unknown(fact_id);
            }
        }

        return initial_state;
    }

    bool solve_native() {
        std::cout << "\n=== Starting Native CPOR AND/OR Search ===" << std::endl;

        if (global_solver) delete global_solver;
        global_solver = new CPOR::CPORSolver();

        PartiallySpecifiedState initial_state = build_owa_initial_state();

        // Build FF's connectivity graph once for this problem load. CPORSolver's
        // heuristic prefers it when available and falls back to a bounded BFS
        // otherwise (see CPORSolver::compute_heuristic) -- never used partially.
        bool ff_ready = CPOR::FFBridge::build(get_global_problem());
        std::cout << "[FF] Relaxed-planning-graph heuristic "
                  << (ff_ready ? "available for this problem." : "not usable for this problem (falling back to bounded BFS heuristic).")
                  << std::endl;

        int root_idx = global_solver->create_root_node(initial_state);

        // Pass an initially empty set to track the current search path for cycle detection
        std::vector<int> current_path_indices;        // Preallocate reasonable depth to prevent vector resizing mid-search
        current_path_indices.reserve(1024);

        std::cout << "Initial State: " << std::endl;
        std::cout << "Known Mask: ";
        for (size_t i = 0; i < initial_state.known_mask.size(); ++i) {
            std::cout << std::hex << initial_state.known_mask[i] << std::dec << " ";
        }
        std::cout << std::endl;
        std::cout << "Value Mask: ";
        for (size_t i = 0; i < initial_state.value_mask.size(); ++i) {
            std::cout << std::hex << initial_state.value_mask[i] << std::dec << " ";
        }
        std::cout << std::endl;


        bool success;
        try {
            success = global_solver->solve_from_node(root_idx, current_path_indices);
        } catch (const std::exception& e) {
            // Same reasoning as CPORSolver::fallback_to_exhaustive_search's
            // identical catch: the exhaustive search can combinatorially
            // explode on some domains (a known, pre-existing FF/solve_from_node
            // characteristic, out of scope to fix here); left uncaught here --
            // this is solve_from_node's OTHER direct call site, invoked at the
            // true root rather than via the fallback -- an allocation failure
            // (or any other exception -- e.g. a malformed RPN tripping
            // Evaluator's/Z3Manager's own std::runtime_error checks) would
            // cross the ctypes boundary as an unhandled exception and abort
            // the whole host process. Report failure instead.
            std::cerr << "[solve_native] CRITICAL: exhaustive search threw " << e.what()
                      << " -- reporting unsolvable instead of crashing." << std::endl;
            success = false;
        }

        if (success) {
            std::cout << ">>> SUCCESS: Contingent Plan Found! <<<" << std::endl;
            std::cout << "Total Universes Explored (Node Count): " << global_solver->get_node_count() << std::endl;
        }
        return success;
    }

    // Opt-in entry point for the new CPOR stack-based outer loop
    // (CPORSolver::solve_cpor_loop), kept separate from solve_native() so the
    // existing exhaustive-search path (and everything that calls it --
    // problem_grounder.py, the rest of the test suite) is completely
    // unaffected while this milestone's control-flow gets verified in
    // isolation. Populates the identical node_pool/PlanNode extraction
    // contract, so get_chosen_action/get_single_child/get_true_child/
    // get_false_child/get_root_node_index/get_node_info all work unchanged
    // against a tree built this way.
    bool solve_native_cpor_loop() {
        std::cout << "\n=== Starting Native CPOR Stack-Based Outer Loop ===" << std::endl;

        if (global_solver) delete global_solver;
        global_solver = new CPOR::CPORSolver();

        PartiallySpecifiedState initial_state = build_owa_initial_state();

        // FFSolver (used internally by SDRPlanner::compute_linear_plan, the
        // OnlinePlan subroutine) and the fallback's compute_heuristic both
        // benefit from/depend on this the same way solve_native() does.
        bool ff_ready = CPOR::FFBridge::build(get_global_problem());
        std::cout << "[FF] Relaxed-planning-graph heuristic "
                  << (ff_ready ? "available for this problem." : "not usable for this problem (falling back to bounded BFS heuristic).")
                  << std::endl;

        int root_idx = global_solver->create_root_node(initial_state);
        bool success = global_solver->solve_cpor_loop(root_idx);

        if (success) {
            std::cout << ">>> SUCCESS: Contingent Plan Found (CPOR loop)! <<<" << std::endl;
            std::cout << "Total Universes Explored (Node Count): " << global_solver->get_node_count() << std::endl;
        }
        std::cout << "Layered fallback (Option 1) invocations this solve: " << global_solver->get_fallback_invocation_count() << std::endl;
        CPOR::DebugStats::print_summary();
        return success;
    }

    // How many times the current global_solver's solve_cpor_loop() call had
    // to invoke the layered fallback (Option 1) to solve_from_node. Only
    // meaningful immediately after a solve_native_cpor_loop() call -- 0 means
    // the online-planner loop resolved everything on its own.
    int get_cpor_loop_fallback_count() {
        return global_solver ? global_solver->get_fallback_invocation_count() : -1;
    }

    // =====================================================================
    // SDR INCREMENTAL SESSION API (native backing for up_cpor.engine.SDRImpl's
    // ActionSelectorMixin interface: get_action()/update(observation) one
    // step at a time, as opposed to solve_native()/solve_native_cpor_loop()'s
    // batch "build the whole plan tree" contract above). Global, single-session
    // state, matching global_solver's own convention -- this engine is used
    // synchronously from a single UP ActionSelector at a time, never
    // concurrently, exactly like the batch entry points already assume.
    // =====================================================================
    CPOR::SDRPlanner* global_sdr_session = nullptr;

    void sdr_session_init() {
        if (global_sdr_session) delete global_sdr_session;

        PartiallySpecifiedState initial_state = build_owa_initial_state();

        bool ff_ready = CPOR::FFBridge::build(get_global_problem());
        std::cout << "[FF] Relaxed-planning-graph heuristic "
                  << (ff_ready ? "available for this problem." : "not usable for this problem (falling back to bounded BFS heuristic).")
                  << std::endl;

        global_sdr_session = new CPOR::SDRPlanner(initial_state, get_global_problem());
    }

    // Returns the next action id to execute, -1 if the goal is already
    // reached, or -2 on failure (confirmed dead end, classical solver found
    // no plan, or the session state machine was used out of order -- e.g.
    // calling this again before sdr_apply_observation() answered a pending
    // sensing action). Mirrors CPOR::SDRPlanner::get_next_action()'s own
    // return contract directly.
    int sdr_get_next_action() {
        if (!global_sdr_session) return -2;
        try {
            return global_sdr_session->get_next_action();
        } catch (const std::exception&) {
            // A malformed RPN reaching Evaluator/Z3Manager (stack underflow,
            // overflow, or a bad translation) would otherwise cross this
            // extern "C" boundary as an uncaught C++ exception -- undefined
            // behavior, in practice std::terminate()/abort() of the whole
            // host process. Report a clean failure instead, exactly like the
            // solved/unsolved sentinel already used by every other return
            // path here.
            return -2;
        }
    }

    // Answers the sensing action most recently returned by
    // sdr_get_next_action(); observation_value is the sensed truth value of
    // that action's observed fluent. Returns false if the session wasn't
    // actually expecting an observation, or if the observation contradicts
    // the problem's oneof invariants.
    bool sdr_apply_observation(bool observation_value) {
        if (!global_sdr_session) return false;
        try {
            return global_sdr_session->apply_observation(observation_value);
        } catch (const std::exception&) {
            // See sdr_get_next_action's identical guard above.
            return false;
        }
    }

    void sdr_session_destroy() {
        if (global_sdr_session) {
            delete global_sdr_session;
            global_sdr_session = nullptr;
        }
    }

    // =====================================================================
    // NATIVE EXECUTION-ENVIRONMENT SIMULATOR (native backing for
    // up_cpor.simulator.SDRSimulator, the Python restoration of the
    // README's "SDR Engine - with SDR Simulated Environment" usage pattern
    // -- previously backed entirely by the legacy C# CPORLib.PlanningModel.Simulator
    // via pythonnet, now removed). Distinct from the SDR SESSION above: that
    // one is the *planner*'s own incremental belief tracking (what SDRImpl
    // thinks is true); this one is the *environment*'s single, fully-resolved
    // ground truth (what's actually true), exactly the same conceptual split
    // unified_planning's own SimulatedExecutionEnvironment makes between the
    // planner under test and the environment driving it. Reuses
    // ActionApplier (the same deterministic effect-application logic
    // CPORSolver/SDRPlanner already trust) and SDRSampler (the same Z3-backed
    // oneof-consistent sampling CPOR's own OnlinePlan witness generation
    // already trusts) rather than introducing a second implementation of
    // either.
    // =====================================================================
    PartiallySpecifiedState* global_sim_ground_truth = nullptr;

    // Samples exactly one fully concrete (every fact known) world consistent
    // with the current problem's initial belief and oneof constraints, and
    // makes it the simulator's ground truth. Must be called after the
    // problem is fully loaded (init_problem() plus every add_*/create_action
    // call), mirroring sdr_session_init()'s own contract.
    bool sim_session_init() {
        if (global_sim_ground_truth) {
            delete global_sim_ground_truth;
            global_sim_ground_truth = nullptr;
        }

        PartiallySpecifiedState initial_belief = build_owa_initial_state();
        auto samples = CPOR::SDRSampler::sample_concrete_states(initial_belief, get_global_problem(), 1);
        if (samples.empty()) {
            // The initial belief has no logically consistent concrete
            // resolution at all (a malformed/contradictory oneof setup) --
            // nothing meaningful to simulate. Report failure rather than
            // leaving global_sim_ground_truth null for later calls to
            // silently misbehave against.
            return false;
        }
        global_sim_ground_truth = new PartiallySpecifiedState(samples[0]);
        return true;
    }

    // Checks action_id's precondition against the concrete ground truth (a
    // fully-resolved state, so PESSIMISTIC/OPTIMISTIC/EXACT all agree) and,
    // if satisfied, applies its effects to it. Returns false (ground truth
    // left untouched) if the action's precondition doesn't actually hold --
    // exactly the same "reject an inapplicable action" contract
    // unified_planning's own SimulatedExecutionEnvironment.apply() has, so
    // Python can raise the same UPUsageError either simulator would.
    bool sim_apply_action(int action_id) {
        if (!global_sim_ground_truth) return false;
        const ProblemDef& problem = get_global_problem();
        if (action_id < 0 || static_cast<size_t>(action_id) >= problem.actions.size()) return false;

        const GroundedAction& action = problem.actions[action_id];
        if (!Evaluator::evaluate(action.precondition_rpn, *global_sim_ground_truth, EvalMode::PESSIMISTIC)) {
            return false;
        }

        try {
            ActionApplier::apply_action(action, *global_sim_ground_truth);
        } catch (const std::exception&) {
            // See sdr_get_next_action's identical guard: never let a
            // malformed-RPN exception cross the extern "C" boundary.
            return false;
        }
        return true;
    }

    // Reads fact_id's truth value out of the CURRENT ground truth -- called
    // by Python right after a successful sim_apply_action() of a sensing
    // action, to read back the fact it just observed. 1/0 for known
    // true/false; -1 is defensive only (SDRSampler guarantees every fact is
    // resolved in a sampled concrete state, so this should never actually
    // fire in practice) and also covers "no active session"/an out-of-range
    // fact_id.
    int sim_get_fact_value(int fact_id) {
        if (!global_sim_ground_truth) return -1;
        if (fact_id < 0) return -1;
        if (global_sim_ground_truth->is_true(fact_id)) return 1;
        if (global_sim_ground_truth->is_false(fact_id)) return 0;
        return -1;
    }

    // Evaluates the problem's goal formula against the current ground truth.
    bool sim_is_goal_reached() {
        if (!global_sim_ground_truth) return false;
        return Evaluator::evaluate(get_global_problem().goal_rpn, *global_sim_ground_truth, EvalMode::PESSIMISTIC);
    }

    void sim_session_destroy() {
        if (global_sim_ground_truth) {
            delete global_sim_ground_truth;
            global_sim_ground_truth = nullptr;
        }
    }

    // =====================================================================
    // PLAN EXTRACTION GETTERS FOR PYTHON
    // =====================================================================
    // node_idx crosses the ctypes boundary as a plain int under Python's
    // control (problem_grounder.py's extract_node recursion, or any other
    // caller), and CPORSolver::get_node()/node_pool's operator[] performs no
    // bounds checking of its own -- an out-of-range or stale idx (e.g. called
    // before any solve, or after a different problem was loaded) would
    // silently read adjacent heap memory rather than fail cleanly. Every
    // getter below validates node_idx against get_node_count() first and
    // returns the same "nothing here" sentinel (-1, matching single/true/
    // false_child_idx's own default and the -1 chosen_action_id already means
    // for a solved-but-actionless leaf) instead of touching node_pool.
    static bool is_valid_node_idx(int node_idx) {
        return global_solver != nullptr && node_idx >= 0 &&
               static_cast<size_t>(node_idx) < global_solver->get_node_count();
    }

    int get_chosen_action(int node_idx) {
        if (!is_valid_node_idx(node_idx)) return -1;
        return global_solver->get_node(node_idx).chosen_action_id;
    }

    // Distinguishes a genuinely failed node from a solved leaf that needed
    // no further action (goal already reached at this belief -- e.g. a
    // sensing branch whose outcome, combined with OneOf/provenance
    // deductions, immediately satisfies the goal). Both cases leave
    // chosen_action_id at its default -1, so callers extracting the plan
    // tree must check this explicitly rather than inferring "failed" from
    // get_chosen_action() == -1 alone.
    int get_node_is_solved(int node_idx) {
        if (!is_valid_node_idx(node_idx)) return 0;
        return global_solver->get_node(node_idx).is_solved ? 1 : 0;
    }

    int get_single_child(int node_idx) {
        if (!is_valid_node_idx(node_idx)) return -1;
        return global_solver->get_node(node_idx).single_child_idx;
    }

    int get_true_child(int node_idx) {
        if (!is_valid_node_idx(node_idx)) return -1;
        return global_solver->get_node(node_idx).true_child_idx;
    }

    int get_false_child(int node_idx) {
        if (!is_valid_node_idx(node_idx)) return -1;
        return global_solver->get_node(node_idx).false_child_idx;
    }

    // Validation interface for Python testing
    // Returns 1 for True, 0 for False, and -1 for Unknown
    int check_initial_state_fluent(int fluent_id) {
        PartiallySpecifiedState initial_state(get_global_problem().total_predicates, get_global_problem().total_functions);
        
        for (int id : get_global_problem().initial_true_facts) {
            initial_state.known_mask[id / 64] |= (1ULL << (id % 64));
            initial_state.value_mask[id / 64] |= (1ULL << (id % 64));
        }
        
        for (int id : get_global_problem().initial_false_facts) {
            initial_state.known_mask[id / 64] |= (1ULL << (id % 64));
            initial_state.value_mask[id / 64] &= ~(1ULL << (id % 64));
        }

        if (initial_state.is_true(fluent_id)) return 1;
        if (initial_state.is_false(fluent_id)) return 0;
        
        // Represents mathematical Open World ignorance
        return -1; 
    }

    // action_id < 0 is already rejected by the >= comparison below today (a
    // negative int promotes to a huge value against the unsigned .size()),
    // but that safety is an accident of signed/unsigned promotion rather than
    // a stated invariant -- made explicit here so it can't be silently lost
    // by some future refactor of the comparison (e.g. switching operand
    // order, or comparing against a signed count instead).
    int get_action_precondition_len(int action_id) {
        if (action_id < 0 || static_cast<size_t>(action_id) >= get_global_problem().actions.size()) return -1;
        return get_global_problem().actions[action_id].precondition_rpn.size();
    }

    int get_action_guaranteed_effect_count(int action_id) {
        if (action_id < 0 || static_cast<size_t>(action_id) >= get_global_problem().actions.size()) return -1;
        return get_global_problem().actions[action_id].guaranteed_effects.size();
    }

    int get_action_conditional_effect_count(int action_id) {
        if (action_id < 0 || static_cast<size_t>(action_id) >= get_global_problem().actions.size()) return -1;
        return get_global_problem().actions[action_id].conditional_effects.size();
    }

    int get_action_observe_id(int action_id) {
        if (action_id < 0 || static_cast<size_t>(action_id) >= get_global_problem().actions.size()) return -1;
        return get_global_problem().actions[action_id].observe_predicate_id;
    }
    // ==================================================================
    // NEW POC API: Adds a full action with RPN preconditions and effects
    // ==================================================================
    void add_grounded_action_to_cpp(int action_id, 
                                    int* pre_rpn, int pre_len, 
                                    int* eff_facts, uint8_t* eff_vals, int eff_len, 
                                    int observe_id) {
        GroundedAction action;
        action.id = action_id;
        action.observe_predicate_id = observe_id;
        
        for (int i = 0; i < pre_len; ++i) {
            action.precondition_rpn.push_back(pre_rpn[i]);
        }
        
        for (int i = 0; i < eff_len; ++i) {
            action.guaranteed_effects.push_back({eff_facts[i], eff_vals[i] != 0});
        }
        
        // Push the base action. Conditional effects will be added via a separate call.
        get_global_problem().actions.push_back(action);
    }

    /// ==================================================================
    // NEW POC API: Executes simple BFS and returns path of action IDs
    // ==================================================================
    int solve_poc_bfs(int* out_action_ids, int max_length) {
        
        // 1. Construct the Initial Epistemic State for the Classical Search
        // We must define this here instead of inside the solver so the solver 
        // remains stateless and usable for mid-execution replanning.
        PartiallySpecifiedState initial_state(get_global_problem().total_predicates);
        
        int blocks = (get_global_problem().total_predicates / 64) + 1;
        
        // For a purely classical BFS POC, we assume total observability at T=0
        initial_state.known_mask.assign(blocks, ~0ULL); // All variables are known
        initial_state.value_mask.assign(blocks, 0);     // Default all to false

        // Inject initial true facts
        for (int id : get_global_problem().initial_true_facts) {
            initial_state.value_mask[id / 64] |= (1ULL << (id % 64));
        }
        
        // Inject initial false facts
        for (int id : get_global_problem().initial_false_facts) {
            initial_state.value_mask[id / 64] &= ~(1ULL << (id % 64));
        }

        // 2. Call the newly architected static DOD solver
        std::vector<int> plan = CPOR::BFSSolver::solve(initial_state, get_global_problem());
        
        // 3. Process the output
        if (plan.empty()) {
            return -1; // Unsolvable or already at goal
        }
        
        int len = std::min((int)plan.size(), max_length);
        for (int i = 0; i < len; ++i) {
            out_action_ids[i] = plan[i];
        }
        
        return len;
    }

        // ---------------------------------------------------------
    // TREE EXTRACTION API
    // ---------------------------------------------------------

    int get_root_node_index() {
        // The root node of the solved plan is always the first node generated
        return 0; 
    }

    // Safely populates the caller's pre-allocated arrays to avoid memory leaks across the C-ABI
    void get_node_info(int idx, int* action_id, int* observe_id, int* num_children, int* children_indices, int* children_obs_values) {
        if (!is_valid_node_idx(idx)) {
            // Same "nothing here" contract as the getters above -- idx is
            // caller-controlled across the ctypes boundary, so this must fail
            // cleanly rather than index node_pool out of bounds.
            *action_id = -1;
            *observe_id = -1;
            *num_children = 0;
            return;
        }
        const CPOR::PlanNode& node = global_solver->get_node(idx);

        *action_id = node.chosen_action_id;
        
        // If it's a leaf/goal node with no action
        if (*action_id == -1) {
            *observe_id = -1;
            *num_children = 0;
            return;
        }

        const GroundedAction& action = get_global_problem().actions[*action_id];
        *observe_id = action.observe_predicate_id;

        if (*observe_id == -1) {
            // Classical Action (OR Node)
            *num_children = 1;
            children_indices[0] = node.single_child_idx;
            children_obs_values[0] = -1; // N/A
        } else {
            // Sensing Action (AND Node)
            *num_children = 2;
            // Branch 1: True
            children_indices[0] = node.true_child_idx;
            children_obs_values[0] = 1; 
            // Branch 2: False
            children_indices[1] = node.false_child_idx;
            children_obs_values[1] = 0; 
        }
    }
} // End of extern "C"