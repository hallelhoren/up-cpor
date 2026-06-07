import os
import sys
import ctypes
import itertools
from typing import Dict, List

import unified_planning as up
from unified_planning.model import FNode, OperatorKind
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
OP_TRUE = -6
OP_FALSE = -7

# ---------------------------------------------------------------------------
# ctypes C++ Library Loader
# ---------------------------------------------------------------------------
_lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tests', 'libcpor_core.so'))
try:
    cpor_lib = ctypes.CDLL(_lib_path)
    
    cpor_lib.init_problem.argtypes = [ctypes.c_int, ctypes.c_int]
    cpor_lib.add_initial_fact.argtypes = [ctypes.c_int]
    cpor_lib.add_initial_false_fact.argtypes = [ctypes.c_int]
    cpor_lib.set_goal_rpn.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    cpor_lib.create_action.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int
    ]
    cpor_lib.add_guaranteed_effect.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_bool
    ]
    cpor_lib.add_conditional_effect.argtypes = [
        ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.c_bool
    ]
    cpor_lib.add_nondeterministic_effect.argtypes = [
        ctypes.c_int, ctypes.c_int
    ]
    cpor_lib.add_oneof_constraint.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
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
        # CRITICAL: Persistent memory references to protect ctypes pointers from the Python Garbage Collector.
        self._c_memory_refs: List[ctypes.Array] = []

    def generate_native_problem(self, original_problem: up.model.Problem):
        em = original_problem.environment.expression_manager
        substituter = Substituter(original_problem.environment)

        self._build_fluent_dict(original_problem, em)
        cpor_lib.init_problem(len(self.fluent_to_id),0)

        explicitly_known = set()
        for init_node, value in original_problem.initial_values.items():
            fluent_str = str(init_node)
            fluent_id = self.fluent_to_id.get(fluent_str)
            if fluent_id is not None:
                if value.is_true():
                    cpor_lib.add_initial_fact(fluent_id)
                    explicitly_known.add(fluent_str)
                elif value.is_false():
                    cpor_lib.add_initial_false_fact(fluent_id)
                    explicitly_known.add(fluent_str)

        combined_goal_rpn = []
        for i, g in enumerate(original_problem.goals):
            combined_goal_rpn.extend(self._compile_to_rpn(g))
            if i > 0: combined_goal_rpn.append(OP_AND)
        
        if combined_goal_rpn:
            c_goal_arr = (ctypes.c_int * len(combined_goal_rpn))(*combined_goal_rpn)
            self._c_memory_refs.append(c_goal_arr)
            cpor_lib.set_goal_rpn(c_goal_arr, len(combined_goal_rpn))

        action_idx = 0
        for action in original_problem.actions:
            param_lists = [list(original_problem.objects(p.type)) for p in action.parameters]

            for combo in itertools.product(*param_lists):
                subs = {p: em.ObjectExp(obj) for p, obj in zip(action.parameters, combo)}
                self.action_id_to_up_action[action_idx] = action

                # 1. Compile Preconditions
                pre_rpn = []
                for i, p in enumerate(action.preconditions):
                    grounded_p = substituter.substitute(p, subs)
                    pre_rpn.extend(self._compile_to_rpn(grounded_p))
                    if i > 0: pre_rpn.append(OP_AND)

                # 2. Extract Observation Targets
                obs_id = -1
                if hasattr(action, 'observed_fluents') and action.observed_fluents:
                    fluent_str = str(substituter.substitute(action.observed_fluents[0], subs))
                    if fluent_str in self.fluent_to_id:
                        obs_id = self.fluent_to_id[fluent_str]

                # 3. Calculate Cost
                action_cost = 1
                if hasattr(original_problem, 'quality_metrics'):
                    for metric in original_problem.quality_metrics:
                        if isinstance(metric, up.model.metrics.MinimizeActionCosts):
                            cost_expr = metric.get_action_cost(action)
                            if cost_expr is not None and cost_expr.is_int_constant():
                                action_cost = cost_expr.constant_value()

                # 4. Instantiate Base Action in C++ Memory
                c_pre = (ctypes.c_int * len(pre_rpn))(*pre_rpn)
                self._c_memory_refs.append(c_pre)
                cpor_lib.create_action(action_idx, action_cost, c_pre, len(pre_rpn), obs_id)

                # 5. Route Effects (Guaranteed vs. Conditional)
                for eff in action.effects:
                    fluent_str = str(substituter.substitute(eff.fluent, subs))
                    if fluent_str not in self.fluent_to_id:
                        continue
                    fluent_id = self.fluent_to_id[fluent_str]
                    val = eff.value.is_true()
                    
                    # UP stores the condition of an effect in eff.condition
                    cond_grounded = substituter.substitute(eff.condition, subs)
                    
                    if cond_grounded.is_true():
                        # Strict Deterministic Effect
                        cpor_lib.add_guaranteed_effect(action_idx, fluent_id, val)
                    else:
                        # Conditional Effect (WHEN Clause)
                        cond_rpn = self._compile_to_rpn(cond_grounded)
                        c_cond = (ctypes.c_int * len(cond_rpn))(*cond_rpn)
                        self._c_memory_refs.append(c_cond)
                        cpor_lib.add_conditional_effect(action_idx, c_cond, len(cond_rpn), fluent_id, val)

                # 6. Route Non-Deterministic Effects (If present in domain spec)
                # (Assuming you expand your grounder to handle NonDeterministicAction types)
                if hasattr(action, 'non_deterministic_effects'):
                    for nd_eff in action.non_deterministic_effects:
                        fluent_str = str(substituter.substitute(nd_eff, subs))
                        if fluent_str in self.fluent_to_id:
                            cpor_lib.add_nondeterministic_effect(action_idx, self.fluent_to_id[fluent_str])

                action_idx += 1

        if hasattr(original_problem, 'oneof_constraints'):
            for oneof_group in original_problem.oneof_constraints:
                group_ids = [self.fluent_to_id[str(f)] for f in oneof_group if str(f) in self.fluent_to_id]
                if group_ids:
                    c_group = (ctypes.c_int * len(group_ids))(*group_ids)
                    self._c_memory_refs.append(c_group)
                    cpor_lib.add_oneof_constraint(c_group, len(group_ids))

        for dead_end_list in [getattr(original_problem, 'state_invariants', []), getattr(original_problem, 'dead_ends', [])]:
            for formula in dead_end_list:
                rpn_list = self._compile_to_rpn(formula)
                if rpn_list:
                    c_rpn_array = (ctypes.c_int * len(rpn_list))(*rpn_list)
                    self._c_memory_refs.append(c_rpn_array)
                    cpor_lib.add_deadend_rpn(c_rpn_array, len(rpn_list))

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
        if node.is_true(): return [OP_TRUE]
        if node.is_false(): return [OP_FALSE]
        
        if node.node_type == OperatorKind.FLUENT_EXP:
            fluent_str = str(node)
            if fluent_str in self.fluent_to_id:
                return [self.fluent_to_id[fluent_str]]
            else:
                raise ValueError(f"CRITICAL: Fluent '{fluent_str}' not found in registry. Grounding failed.")

        if node.node_type == OperatorKind.IMPLIES:
            if len(node.args) != 2:
                raise ValueError("IMPLIES node does not have exactly 2 arguments.")
            rpn = self._compile_to_rpn(node.args[0])
            rpn.append(OP_NOT)
            rpn.extend(self._compile_to_rpn(node.args[1]))
            rpn.append(OP_OR)
            return rpn
            
        elif node.node_type == OperatorKind.IFF:
            if len(node.args) != 2:
                raise ValueError("IFF node does not have exactly 2 arguments.")
            rpn = self._compile_to_rpn(node.args[0])
            rpn.extend(self._compile_to_rpn(node.args[1]))
            rpn.append(OP_EQUALS)
            return rpn

        rpn = []
        for arg in node.args:
            rpn.extend(self._compile_to_rpn(arg))

        num_args = len(node.args)
        
        if node.node_type == OperatorKind.AND:
            if num_args > 1:
                rpn.extend([OP_AND] * (num_args - 1))
        elif node.node_type == OperatorKind.OR:
            if num_args > 1:
                rpn.extend([OP_OR] * (num_args - 1))
        elif node.node_type == OperatorKind.EQUALS:
            if num_args > 1:
                rpn.extend([OP_EQUALS] * (num_args - 1))
        elif node.node_type == OperatorKind.NOT:
            if num_args == 1:
                rpn.append(OP_NOT)
            else:
                raise ValueError("NOT node does not have exactly 1 argument.")
        else:
            raise ValueError(f"CRITICAL: Unhandled OperatorKind '{node.node_type}'.")
        
        return rpn

    def createActionTree(self, cpp_solution_node, problem) -> ContingentPlanNode:
        pass