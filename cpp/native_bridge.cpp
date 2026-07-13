#include "ProblemData.hpp"
#include "State.hpp"
#include "Evaluator.hpp"
#include "CPORSolver.hpp"
#include "BFSSolver.hpp"
#include "FFBridge.hpp"
#include "SDRSampler.hpp"
#include "DeadEndManager.hpp"
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
        std::cout << "Initial Facts: " << get_global_problem().initial_true_facts.size() << std::endl;
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

        //DEBUG
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
        } catch (const std::bad_alloc&) {
            // Same reasoning as CPORSolver::fallback_to_exhaustive_search's
            // identical catch: the exhaustive search can combinatorially
            // explode on some domains (a known, pre-existing FF/solve_from_node
            // characteristic, out of scope to fix here); left uncaught here --
            // this is solve_from_node's OTHER direct call site, invoked at the
            // true root rather than via the fallback -- an allocation failure
            // would cross the ctypes boundary as an unhandled exception and
            // abort the whole host process. Report failure instead.
            std::cerr << "[solve_native] CRITICAL: exhaustive search ran out of memory "
                      << "(std::bad_alloc) -- reporting unsolvable instead of crashing." << std::endl;
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
    // PLAN EXTRACTION GETTERS FOR PYTHON
    // =====================================================================
    int get_chosen_action(int node_idx) { 
        return global_solver->get_node(node_idx).chosen_action_id; 
    }
    
    int get_single_child(int node_idx) { 
        return global_solver->get_node(node_idx).single_child_idx; 
    }
    
    int get_true_child(int node_idx) { 
        return global_solver->get_node(node_idx).true_child_idx; 
    }
    
    int get_false_child(int node_idx) { 
        return global_solver->get_node(node_idx).false_child_idx; 
    }

    //DEBUGGING FUNCTION TO VERIFY PROBLEM LOADING
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

    int get_action_precondition_len(int action_id) {
        if (action_id >= get_global_problem().actions.size()) return -1;
        return get_global_problem().actions[action_id].precondition_rpn.size();
    }

    int get_action_guaranteed_effect_count(int action_id) {
        if (action_id >= get_global_problem().actions.size()) return -1;
        return get_global_problem().actions[action_id].guaranteed_effects.size();
    }

    int get_action_conditional_effect_count(int action_id) {
        if (action_id >= get_global_problem().actions.size()) return -1;
        return get_global_problem().actions[action_id].conditional_effects.size();
    }

    int get_action_observe_id(int action_id) {
        if (action_id >= get_global_problem().actions.size()) return -1;
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