import os
import ctypes

def get_lib_path():
    # Attempt to locate the compiled C++ core library
    #BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = "/mnt/c/Users/97258/Desktop/technion/semester6/clairProject"
    # Adjust this path based on where your build.sh actually puts the .so file
    LIB_PATH = os.path.join(BASE_DIR, "cpp", "build", "libcpor_core.so")

    if not os.path.exists(LIB_PATH):
        print(f"WARNING: Native library not found at {LIB_PATH}. Please compile using CMake.")
        raise FileNotFoundError(f"Could not find libcpor_core.so. Looked in: {LIB_PATH}")
    else:
        return ctypes.CDLL(LIB_PATH)

cpor_lib = get_lib_path()

# ---------------------------------------------------------
# Define Strict ctypes Signatures (ABI Security)
# ---------------------------------------------------------
cpor_lib.init_problem.argtypes = [ctypes.c_int, ctypes.c_int]
cpor_lib.add_initial_fact.argtypes = [ctypes.c_int]
cpor_lib.add_initial_false_fact.argtypes = [ctypes.c_int]
cpor_lib.set_goal_rpn.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]

# NEW: OneOf Endpoint
cpor_lib.add_oneof_group.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]

# UPDATED: c_bool replaced with c_uint8
cpor_lib.add_grounded_action_to_cpp.argtypes = [
    ctypes.c_int, 
    ctypes.POINTER(ctypes.c_int), ctypes.c_int, 
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, 
    ctypes.c_int 
]

# NEW: Conditional Effect Endpoint
cpor_lib.add_conditional_effect_to_action.argtypes = [
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int), ctypes.c_int,
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_uint8), ctypes.c_int
]

# ---------------------------------------------------------
# The Loader Function
# ---------------------------------------------------------
def load_problem_to_cpp(grounded_data):
    print("[NativeAPI] Initializing C++ Problem...")
    cpor_lib.init_problem(grounded_data['total_predicates'], 0)
    
    # 1. Load Initial State Facts
    for fact in grounded_data['initial_true']:
        cpor_lib.add_initial_fact(fact)
    for fact in grounded_data['initial_false']:
        cpor_lib.add_initial_false_fact(fact)

    # 2. NEW: Load OneOf Invariants
    if 'oneofs' in grounded_data:
        for group in grounded_data['oneofs']:
            c_group = (ctypes.c_int * len(group))(*group)
            cpor_lib.add_oneof_group(c_group, len(group))
        
    print(f"[NativeAPI] Pushing {len(grounded_data['actions'])} actions to C++...")
    
    # 3. Load Actions
    for act in grounded_data['actions']:
        rpn = act['pre_rpn']
        facts = act['eff_facts']
        vals = act['eff_vals']
        observe_id = act.get('observe_id', -1)
        
        c_rpn = (ctypes.c_int * len(rpn))(*rpn)
        c_facts = (ctypes.c_int * len(facts))(*facts)
        
        # CRITICAL FIX: Cast booleans to 1-byte uint8_t integers
        c_vals = (ctypes.c_uint8 * len(vals))(*[1 if v else 0 for v in vals])
        
        # Push the Base Action
        cpor_lib.add_grounded_action_to_cpp(
            act['id'], 
            c_rpn, len(rpn), 
            c_facts, c_vals, len(facts), 
            observe_id
        )

        # 4. NEW: Push Conditional Effects
        if 'conditional_effects' in act:
            for ce in act['conditional_effects']:
                cond_rpn = ce['condition_rpn']
                ce_facts = ce['eff_facts']
                ce_vals = ce['eff_vals']

                c_cond_rpn = (ctypes.c_int * len(cond_rpn))(*cond_rpn)
                c_ce_facts = (ctypes.c_int * len(ce_facts))(*ce_facts)
                c_ce_vals = (ctypes.c_uint8 * len(ce_vals))(*[1 if v else 0 for v in ce_vals])

                cpor_lib.add_conditional_effect_to_action(
                    act['id'],
                    c_cond_rpn, len(cond_rpn),
                    c_ce_facts, c_ce_vals, len(ce_facts)
                )
        
    # 5. Load Goal
    goal_rpn = grounded_data['goal_rpn']
    c_goal = (ctypes.c_int * len(goal_rpn))(*goal_rpn)
    cpor_lib.set_goal_rpn(c_goal, len(goal_rpn))

    print("[NativeAPI] C++ Environment successfully initialized.")