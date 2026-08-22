import ctypes
import os
from typing import List, Dict, Any

def get_lib_path() -> str:
    """Resolve the native CPOR shared library path."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(base_dir, "..", "libcpor_core.so"),
        os.path.join(base_dir, "..", "build", "libcpor_core.so"),
        os.path.join(base_dir, "cpp", "build", "libcpor_core.so"),
    ]

    for path in possible_paths:
        normalized = os.path.abspath(path)
        if os.path.exists(normalized):
            return normalized

    searched = ", ".join(os.path.abspath(path) for path in possible_paths)
    raise FileNotFoundError(f"Could not find libcpor_core.so. Looked in: {searched}")

_LIB_PATH = get_lib_path()
cpor_lib = ctypes.CDLL(_LIB_PATH)

# ---------------------------------------------------------
# Define Strict ctypes Signatures (ABI Security)
# ---------------------------------------------------------
cpor_lib.init_problem.argtypes = [ctypes.c_int, ctypes.c_int]
cpor_lib.set_problem_is_simple.argtypes = [ctypes.c_bool]
cpor_lib.set_problem_is_simple.restype = None
cpor_lib.add_initial_fact.argtypes = [ctypes.c_int]
cpor_lib.add_initial_false_fact.argtypes = [ctypes.c_int]
cpor_lib.add_initial_function_value.argtypes = [ctypes.c_int, ctypes.c_double]
cpor_lib.set_goal_rpn.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
cpor_lib.add_oneof_group.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
cpor_lib.create_action.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.c_int,
]
cpor_lib.add_guaranteed_effect.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_bool]
cpor_lib.add_conditional_effect.argtypes = [
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_bool,
]
cpor_lib.add_nondeterministic_effect.argtypes = [ctypes.c_int, ctypes.c_int]
cpor_lib.add_oneof_constraint.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
cpor_lib.add_deadend_rpn.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
cpor_lib.add_nondeterministic_effect.argtypes = [ctypes.c_int, ctypes.c_int]

# CRITICAL FIX: Replaced c_bool with c_uint8 for safe memory alignment
cpor_lib.add_grounded_action_to_cpp.argtypes = [
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int), ctypes.c_int,
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_uint8), ctypes.c_int,
    ctypes.c_int,
]

cpor_lib.add_conditional_effect_to_action.argtypes = [
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int), ctypes.c_int,
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_uint8), ctypes.c_int,
]

# Getters
cpor_lib.solve_poc_bfs.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
cpor_lib.solve_poc_bfs.restype = ctypes.c_int
cpor_lib.solve_native.argtypes = []
cpor_lib.solve_native.restype = ctypes.c_bool
cpor_lib.solve_native_cpor_loop.argtypes = []
cpor_lib.solve_native_cpor_loop.restype = ctypes.c_bool
cpor_lib.get_cpor_loop_fallback_count.argtypes = []
cpor_lib.get_cpor_loop_fallback_count.restype = ctypes.c_int
cpor_lib.sdr_session_init.argtypes = []
cpor_lib.sdr_session_init.restype = None
cpor_lib.sdr_get_next_action.argtypes = []
cpor_lib.sdr_get_next_action.restype = ctypes.c_int
cpor_lib.sdr_apply_observation.argtypes = [ctypes.c_bool]
cpor_lib.sdr_apply_observation.restype = ctypes.c_bool
cpor_lib.sdr_session_destroy.argtypes = []
cpor_lib.sdr_session_destroy.restype = None
cpor_lib.sim_session_init.argtypes = []
cpor_lib.sim_session_init.restype = ctypes.c_bool
cpor_lib.sim_apply_action.argtypes = [ctypes.c_int]
cpor_lib.sim_apply_action.restype = ctypes.c_bool
cpor_lib.sim_get_fact_value.argtypes = [ctypes.c_int]
cpor_lib.sim_get_fact_value.restype = ctypes.c_int
cpor_lib.sim_is_goal_reached.argtypes = []
cpor_lib.sim_is_goal_reached.restype = ctypes.c_bool
cpor_lib.sim_session_destroy.argtypes = []
cpor_lib.sim_session_destroy.restype = None
cpor_lib.get_chosen_action.argtypes = [ctypes.c_int]
cpor_lib.get_chosen_action.restype = ctypes.c_int
cpor_lib.get_node_is_solved.argtypes = [ctypes.c_int]
cpor_lib.get_node_is_solved.restype = ctypes.c_int
cpor_lib.get_single_child.argtypes = [ctypes.c_int]
cpor_lib.get_single_child.restype = ctypes.c_int
cpor_lib.get_true_child.argtypes = [ctypes.c_int]
cpor_lib.get_true_child.restype = ctypes.c_int
cpor_lib.get_false_child.argtypes = [ctypes.c_int]
cpor_lib.get_false_child.restype = ctypes.c_int
cpor_lib.get_root_node_index.argtypes = []
cpor_lib.get_root_node_index.restype = ctypes.c_int
cpor_lib.get_node_info.argtypes = [
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
]
cpor_lib.get_node_info.restype = None

# ---------------------------------------------------------
# The Data Dispatcher (Safe C-Serialization)
# ---------------------------------------------------------
def load_problem_to_cpp(grounded_data: Dict[str, Any]) -> None:
    print("[NativeAPI] Initializing C++ Environment...")
    cpor_lib.init_problem(grounded_data['total_predicates'], grounded_data.get('total_functions', 0))
    # Plan Graph Compaction (Option B) gate: mirrors the legacy C# engine's
    # Domain.IsSimple exactly (see problem_grounder.py's generate_native_problem
    # for where this is computed). Must be set right after init_problem(), which
    # resets is_simple to its safe default (false, compaction off).
    cpor_lib.set_problem_is_simple(bool(grounded_data.get('is_simple', False)))

    # Load Initial State
    for fact in grounded_data['initial_true']:
        cpor_lib.add_initial_fact(fact)
    for fact in grounded_data['initial_false']:
        cpor_lib.add_initial_false_fact(fact)
    for func in grounded_data.get('initial_functions', []):
        cpor_lib.add_initial_function_value(func['id'], ctypes.c_double(func['val']))

    # Load Constraints
    for group in grounded_data.get('oneofs', []):
        c_group = (ctypes.c_int * len(group))(*group)
        cpor_lib.add_oneof_group(c_group, len(group))

    for rpn in grounded_data.get('deadends', []):
        c_rpn = (ctypes.c_int * len(rpn))(*rpn)
        cpor_lib.add_deadend_rpn(c_rpn, len(rpn))

    # Load Actions
    print(f"[NativeAPI] Pushing {len(grounded_data['actions'])} actions to C++...")
    for act in grounded_data['actions']:
        pre_rpn = act['pre_rpn']
        eff_facts = act['eff_facts']
        eff_vals = act['eff_vals']

        c_pre = (ctypes.c_int * len(pre_rpn))(*pre_rpn)
        c_facts = (ctypes.c_int * len(eff_facts))(*eff_facts)
        c_vals = (ctypes.c_uint8 * len(eff_vals))(*[1 if v else 0 for v in eff_vals])

        cpor_lib.add_grounded_action_to_cpp(
            act['id'],
            c_pre, len(pre_rpn),
            c_facts, c_vals, len(eff_facts),
            act['observe_id']
        )

        for ce in act.get('conditional_effects', []):
            cond_rpn = ce['condition_rpn']
            ce_facts = ce['eff_facts']
            ce_vals = ce['eff_vals']

            c_cond = (ctypes.c_int * len(cond_rpn))(*cond_rpn)
            c_ce_facts = (ctypes.c_int * len(ce_facts))(*ce_facts)
            c_ce_vals = (ctypes.c_uint8 * len(ce_vals))(*[1 if v else 0 for v in ce_vals])

            cpor_lib.add_conditional_effect_to_action(
                act['id'],
                c_cond, len(cond_rpn),
                c_ce_facts, c_ce_vals, len(ce_facts)
            )

        for nd_fact in act.get('non_deterministic_effects', []):
            cpor_lib.add_nondeterministic_effect(act['id'], nd_fact)

    # Load Goals
    goal_rpn = grounded_data.get('goal_rpn', [])
    if goal_rpn:
        c_goal = (ctypes.c_int * len(goal_rpn))(*goal_rpn)
        cpor_lib.set_goal_rpn(c_goal, len(goal_rpn))

    print("[NativeAPI] Transfer complete.")

# ---------------------------------------------------------
# Exposed Helper Methods
# ---------------------------------------------------------
def print_problem_stats() -> None:
    cpor_lib.print_problem_stats()

def solve_poc_bfs(max_len: int = 1000) -> List[int]:
    out_array = (ctypes.c_int * max_len)()
    plan_length = cpor_lib.solve_poc_bfs(out_array, max_len)
    if plan_length < 0:
        return []
    return [out_array[i] for i in range(plan_length)]

def solve_native() -> bool:
    return bool(cpor_lib.solve_native())

def solve_native_cpor_loop() -> bool:
    """Opt-in entry point for the stack-based CPOR outer loop
    (CPORSolver::solve_cpor_loop), kept separate from solve_native() so nothing
    that already calls solve_native() (problem_grounder.py's default path) is
    affected. Populates the same node_pool extraction contract, so
    get_chosen_action/get_single_child/get_true_child/get_false_child/
    get_root_node_index all work unchanged against a tree built this way."""
    return bool(cpor_lib.solve_native_cpor_loop())

def get_cpor_loop_fallback_count() -> int:
    """How many times the most recent solve_native_cpor_loop() call had to
    invoke the layered fallback (Option 1) to CPORSolver::solve_from_node.
    Only meaningful immediately after a solve_native_cpor_loop() call."""
    return cpor_lib.get_cpor_loop_fallback_count()

def sdr_session_init() -> None:
    """Starts a fresh incremental SDR session over the problem most recently
    loaded via load_problem_to_cpp -- the native backing for
    up_cpor.engine.SDRImpl's ActionSelectorMixin (get_action()/update()),
    as opposed to solve_native()/solve_native_cpor_loop()'s batch "build the
    whole plan tree up front" contract."""
    cpor_lib.sdr_session_init()

def sdr_get_next_action() -> int:
    """Returns the next action id to execute, -1 if the goal is already
    reached, or -2 on failure (confirmed dead end, classical solver found no
    plan, or the session was used out of order)."""
    return cpor_lib.sdr_get_next_action()

def sdr_apply_observation(value: bool) -> bool:
    """Answers the sensing action most recently returned by
    sdr_get_next_action() with its observed truth value. Returns False if no
    observation was actually pending, or if it contradicts the problem's
    oneof invariants."""
    return bool(cpor_lib.sdr_apply_observation(value))

def sdr_session_destroy() -> None:
    cpor_lib.sdr_session_destroy()

def sim_session_init() -> bool:
    """Starts a fresh native execution-environment simulator over the problem
    most recently loaded via load_problem_to_cpp -- the native backing for
    up_cpor.simulator.SDRSimulator. Samples one fully concrete, oneof-consistent
    ground-truth world (via SDRSampler/Z3) and makes it that simulator's state.
    Returns False if the problem's initial belief has no logically consistent
    concrete resolution at all (a malformed/contradictory oneof setup)."""
    return bool(cpor_lib.sim_session_init())

def sim_apply_action(action_id: int) -> bool:
    """Checks action_id's precondition against the simulator's ground truth
    and, if satisfied, applies its effects to it. Returns False (ground truth
    left untouched) if the action isn't actually applicable there -- mirrors
    unified_planning's own SimulatedExecutionEnvironment.apply() rejecting an
    inapplicable action, so callers can raise the same kind of error either
    simulator would."""
    return bool(cpor_lib.sim_apply_action(action_id))

def sim_get_fact_value(fact_id: int) -> int:
    """Reads fact_id's truth value out of the simulator's current ground
    truth: 1 (true), 0 (false), or -1 (no active session / out-of-range
    fact_id -- SDRSampler guarantees every fact is resolved in a sampled
    concrete state, so -1 should never occur for a valid fact_id otherwise).
    Called right after a successful sim_apply_action() of a sensing action to
    read back the fact it just observed."""
    return cpor_lib.sim_get_fact_value(fact_id)

def sim_is_goal_reached() -> bool:
    """Evaluates the problem's goal formula against the simulator's current
    ground truth."""
    return bool(cpor_lib.sim_is_goal_reached())

def sim_session_destroy() -> None:
    cpor_lib.sim_session_destroy()

def get_chosen_action(node_idx: int) -> int:
    return cpor_lib.get_chosen_action(node_idx)

def get_node_is_solved(node_idx: int) -> bool:
    """True for both a node with a chosen action AND a solved leaf that
    needed no further action (goal already reached at this belief). Callers
    extracting the plan tree must check this instead of inferring "failed"
    from get_chosen_action(node_idx) == -1 alone -- both a failed node and a
    no-action-needed solved leaf leave chosen_action_id at its default -1."""
    return bool(cpor_lib.get_node_is_solved(node_idx))

def get_single_child(node_idx: int) -> int:
    return cpor_lib.get_single_child(node_idx)

def get_true_child(node_idx: int) -> int:
    return cpor_lib.get_true_child(node_idx)

def get_false_child(node_idx: int) -> int:
    return cpor_lib.get_false_child(node_idx)

def get_root_node_index() -> int:
    return cpor_lib.get_root_node_index()

def get_node_info(node_idx: int, action_id, observe_id, num_children, children_indices, children_obs_values) -> None:
    cpor_lib.get_node_info(node_idx, action_id, observe_id, num_children, children_indices, children_obs_values)