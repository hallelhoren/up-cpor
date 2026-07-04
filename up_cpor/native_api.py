import os
import ctypes




def get_lib_path():
    # Attempt to locate the compiled C++ core library
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Adjust this path based on where your build.sh actually puts the .so file
    LIB_PATH = os.path.join(BASE_DIR, "cpp", "build", "libcpor_core.so")

    if not os.path.exists(LIB_PATH):
        print(f"WARNING: Native library not found at {LIB_PATH}. Please compile using CMake.")
        raise FileNotFoundError(f"Could not find libcpor_core.so. Looked in: {possible_paths}")
    else:
        return ctypes.CDLL(LIB_PATH)

    

cpor_lib = get_lib_path()

# ---------------------------------------------------------
# Define signatures
# ---------------------------------------------------------
cpor_lib.init_problem.argtypes = [ctypes.c_int, ctypes.c_int]
cpor_lib.add_initial_fact.argtypes = [ctypes.c_int]
cpor_lib.add_initial_false_fact.argtypes = [ctypes.c_int]
cpor_lib.set_goal_rpn.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]

cpor_lib.add_grounded_action_to_cpp.argtypes = [
    ctypes.c_int,                                  # action_id
    ctypes.POINTER(ctypes.c_int), ctypes.c_int,    # pre_rpn, pre_len
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_bool), ctypes.c_int, # eff_facts, eff_vals, eff_len
    ctypes.c_int                                   # observe_id
]

cpor_lib.solve_poc_bfs.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
cpor_lib.solve_poc_bfs.restype = ctypes.c_int


def send_to_cpp_and_solve(grounded_data: dict) -> list:
    """
    Pushes pure python data into the C++ memory space and executes the basic BFS solver.
    """
    print("[NativeAPI] Initializing C++ Problem...")
    cpor_lib.init_problem(grounded_data['total_predicates'], 0)
    
    for fact in grounded_data['initial_true']:
        cpor_lib.add_initial_fact(fact)
    for fact in grounded_data['initial_false']:
        cpor_lib.add_initial_false_fact(fact)
        
    print(f"[NativeAPI] Pushing {len(grounded_data['actions'])} actions to C++...")
    for act in grounded_data['actions']:
        rpn = act['pre_rpn']
        facts = act['eff_facts']
        vals = act['eff_vals']
        
        c_rpn = (ctypes.c_int * len(rpn))(*rpn)
        c_facts = (ctypes.c_int * len(facts))(*facts)
        c_vals = (ctypes.c_bool * len(vals))(*vals)
        
        cpor_lib.add_grounded_action_to_cpp(
            act['id'], 
            c_rpn, len(rpn), 
            c_facts, c_vals, len(facts), 
            -1 # observe_id (-1 for standard action)
        )
        
    goal_rpn = grounded_data['goal_rpn']
    c_goal = (ctypes.c_int * len(goal_rpn))(*goal_rpn)
    cpor_lib.set_goal_rpn(c_goal, len(goal_rpn))
    
    print("[NativeAPI] Invoking C++ POC BFS Solver...")
    max_len = 1000
    out_array = (ctypes.c_int * max_len)()
    
    plan_length = cpor_lib.solve_poc_bfs(out_array, max_len)
    
    if plan_length < 0:
        print("[NativeAPI] C++ returned NO SOLUTION.")
        return []
        
    print(f"[NativeAPI] C++ returned a plan of length {plan_length}.")
    return [out_array[i] for i in range(plan_length)]