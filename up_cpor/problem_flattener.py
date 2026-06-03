import os
import sys
import ctypes
import itertools
from typing import Dict, List

import unified_planning as up
from unified_planning.model import FNode, OperatorKind, SensingAction
from unified_planning.plans import ActionInstance
from unified_planning.plans.contingent_plan import ContingentPlanNode
from unified_planning.model.walkers import Substituter

# ---------------------------------------------------------------------------
# RPN Operator Constants for the C++ Evaluator
# ---------------------------------------------------------------------------
OP_AND = -1
OP_OR = -2
OP_NOT = -3
OP_ONEOF = -4
OP_EQUALS = -5

# ---------------------------------------------------------------------------
# ctypes C++ Library Loader (Replacing pybind11)
# ---------------------------------------------------------------------------
_lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tests', 'libcpor_core.so'))
try:
    cpor_lib = ctypes.CDLL(_lib_path)
    
    # Define argument types to ensure memory safety when passing to C++
    cpor_lib.init_problem.argtypes = [ctypes.c_int]
    cpor_lib.add_initial_fact.argtypes = [ctypes.c_int]
    cpor_lib.add_initial_unknown_fact.argtypes = [ctypes.c_int]
    cpor_lib.set_goal_rpn.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    cpor_lib.add_action.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_bool), ctypes.c_int, ctypes.c_int
    ]
    cpor_lib.add_oneof_constraint.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    
    # Ctypes binding for dead-end evaluations
    cpor_lib.add_deadend_rpn.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    
except OSError:
    print(f"WARNING: Native library not found at {_lib_path}. Please compile using CMake.")


class UpCporConverter:
    """
    Translates unified_planning (UP) objects into the Intermediate Grounded 
    Representation (IGR) and passes them to C++ via native ctypes memory pointers.
    """

    def __init__(self):
        self.fluent_to_id: Dict[str, int] = {}
        self.id_to_fluent: Dict[int, str] = {}
        self.action_id_to_up_action: Dict[int, ActionInstance] = {}
        self.next_id = 0

    def generate_native_problem(self, original_problem: up.model.Problem):
        em = original_problem.environment.expression_manager
        substituter = Substituter(original_problem.environment)

        # 1. Build Fluents Dictionary
        self._build_fluent_dict(original_problem, em)
        
        # Initialize C++ Memory Allocation
        cpor_lib.init_problem(len(self.fluent_to_id))

        # ---------------------------------------------------------
        # 2. Extract Initial State (Open World Assumption Fix)
        # ---------------------------------------------------------
        explicitly_initialized = set()

        for fluent_node, value in original_problem.initial_values.items():
            fluent_str = str(fluent_node)
            explicitly_initialized.add(fluent_str)
            
            fluent_id = self.fluent_to_id.get(fluent_str)
            if fluent_id is not None:
                if value.is_true():
                    cpor_lib.add_initial_fact(fluent_id)
                # If value is false, C++ already defaults to false, so we do nothing.

        # Any grounded fluent that was NOT explicitly initialized must be flagged as UNKNOWN
        for fluent_str, fluent_id in self.fluent_to_id.items():
            if fluent_str not in explicitly_initialized:
                cpor_lib.add_initial_unknown_fact(fluent_id)
        # ---------------------------------------------------------

        # 3. Extract Goals (Combine with AND)
        combined_goal_rpn = []
        for i, g in enumerate(original_problem.goals):
            combined_goal_rpn.extend(self._compile_to_rpn(g))
            if i > 0: combined_goal_rpn.append(OP_AND)
        
        if combined_goal_rpn:
            # Cast python list to C-Array pointer and send to C++
            c_goal_arr = (ctypes.c_int * len(combined_goal_rpn))(*combined_goal_rpn)
            cpor_lib.set_goal_rpn(c_goal_arr, len(combined_goal_rpn))

        # 4. Custom Grounding for Actions
        action_idx = 0
        for action in original_problem.actions:
            param_lists = [list(original_problem.objects(p.type)) for p in action.parameters]

            for combo in itertools.product(*param_lists):
                subs = {p: em.ObjectExp(obj) for p, obj in zip(action.parameters, combo)}
                self.action_id_to_up_action[action_idx] = action

                pre_rpn = []
                for i, p in enumerate(action.preconditions):
                    grounded_p = substituter.substitute(p, subs)
                    pre_rpn.extend(self._compile_to_rpn(grounded_p))
                    if i > 0: pre_rpn.append(OP_AND)

                eff_ids = []
                eff_vals = []
                for eff in action.effects:
                    fluent_str = str(substituter.substitute(eff.fluent, subs))
                    if fluent_str in self.fluent_to_id:
                        eff_ids.append(self.fluent_to_id[fluent_str])
                        eff_vals.append(eff.value.is_true())

                obs_id = -1
                if isinstance(action, SensingAction) and action.observed_fluents:
                    fluent_str = str(substituter.substitute(action.observed_fluents[0], subs))
                    if fluent_str in self.fluent_to_id:
                        obs_id = self.fluent_to_id[fluent_str]

                # ---------------------------------------------------------
                # EXTRACT ACTION COST
                # ---------------------------------------------------------
                action_cost = 1
                # Check UP metrics for explicit action costs
                if hasattr(original_problem, 'quality_metrics'):
                    for metric in original_problem.quality_metrics:
                        if isinstance(metric, up.model.metrics.MinimizeActionCosts):
                            cost_expr = metric.get_action_cost(action)
                            if cost_expr is not None and cost_expr.is_int_constant():
                                action_cost = cost_expr.constant_value()
                # ---------------------------------------------------------

                # Cast arrays to C-pointers
                c_pre = (ctypes.c_int * len(pre_rpn))(*pre_rpn)
                c_eff_ids = (ctypes.c_int * len(eff_ids))(*eff_ids)
                c_eff_vals = (ctypes.c_bool * len(eff_vals))(*eff_vals)

                # Send straight to C++ memory (pass action_cost as the 2nd parameter)
                cpor_lib.add_action(action_idx, action_cost, c_pre, len(pre_rpn), c_eff_ids, c_eff_vals, len(eff_ids), obs_id)
                action_idx += 1

        # 5. Extract ONEOF Constraints
        if hasattr(original_problem, 'oneof_constraints'):
            for oneof_group in original_problem.oneof_constraints:
                group_ids = []
                for fluent in oneof_group:
                    fluent_str = str(substituter.substitute(fluent, subs))
                    if fluent_str in self.fluent_to_id:
                        group_ids.append(self.fluent_to_id[fluent_str])
                
                if group_ids:
                    c_group = (ctypes.c_int * len(group_ids))(*group_ids)
                    cpor_lib.add_oneof_constraint(c_group, len(group_ids))

        # ---------------------------------------------------------
        # 6. Extract Dead-End Constraints (NEW)
        # ---------------------------------------------------------
        # First, check standard UP state invariants
        if hasattr(original_problem, 'state_invariants'):
            for invariant in original_problem.state_invariants:
                rpn_list = self._compile_to_rpn(invariant)
                if rpn_list:
                    c_rpn_array = (ctypes.c_int * len(rpn_list))(*rpn_list)
                    cpor_lib.add_deadend_rpn(c_rpn_array, len(rpn_list))
                    
        # Fallback: check if your parser stores them in a custom attribute
        if hasattr(original_problem, 'dead_ends'):
            for dead_end in original_problem.dead_ends:
                rpn_list = self._compile_to_rpn(dead_end)
                if rpn_list:
                    c_rpn_array = (ctypes.c_int * len(rpn_list))(*rpn_list)
                    cpor_lib.add_deadend_rpn(c_rpn_array, len(rpn_list))
        # ---------------------------------------------------------

        # Trigger C++ to print its internal state to verify it worked!
        cpor_lib.print_problem_stats()

    def _build_fluent_dict(self, problem: up.model.Problem, em):
        for fluent in problem.fluents:
            param_lists = [list(problem.objects(p.type)) for p in fluent.signature]
            if not param_lists:
                fluent_node = em.FluentExp(fluent)
                self._add_fluent_to_dict(str(fluent_node))
            else:
                for combo in itertools.product(*param_lists):
                    obj_args = tuple(em.ObjectExp(obj) for obj in combo)
                    fluent_node = em.FluentExp(fluent, obj_args)
                    self._add_fluent_to_dict(str(fluent_node))

    def _add_fluent_to_dict(self, fluent_str: str):
        if fluent_str not in self.fluent_to_id:
            self.fluent_to_id[fluent_str] = self.next_id
            self.id_to_fluent[self.next_id] = fluent_str
            self.next_id += 1

    def _compile_to_rpn(self, node: FNode) -> List[int]:
        rpn = []
        if node.is_true() or node.is_false():
            return rpn 
        
        if node.node_type == OperatorKind.FLUENT_EXP:
            fluent_str = str(node)
            if fluent_str in self.fluent_to_id:
                rpn.append(self.fluent_to_id[fluent_str])
            return rpn

        for arg in node.args:
            rpn.extend(self._compile_to_rpn(arg))

        if node.node_type == OperatorKind.AND:
            rpn.append(OP_AND)
        elif node.node_type == OperatorKind.OR:
            rpn.append(OP_OR)
        elif node.node_type == OperatorKind.NOT:
            rpn.append(OP_NOT)
        elif node.node_type == OperatorKind.EQUALS:
            rpn.append(OP_EQUALS)
        
        return rpn

    def createActionTree(self, cpp_solution_node, problem) -> ContingentPlanNode:
        # We will update this reconstruction logic in Phase 3 when the Solver is complete!
        pass