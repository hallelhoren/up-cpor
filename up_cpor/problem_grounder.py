import itertools
from typing import Dict, List, Set

import unified_planning as up
from unified_planning.model import FNode, OperatorKind, Fluent
from unified_planning.plans import ActionInstance
from unified_planning.plans.contingent_plan import ContingentPlanNode
from unified_planning.model.walkers import Substituter, DagWalker
from up_cpor import native_api

# RPN Operator Constants for the C++ Evaluator
OP_AND = -1
OP_OR = -2
OP_NOT = -3
OP_ONEOF = -4
OP_EQUALS = -5
OP_TRUE = -6
OP_FALSE = -7

class StaticEvaluator(up.model.walkers.dag.DagWalker):
    """
    Standard DagWalker implementation to replace static fluents with constants.
    """
    def __init__(self, env, static_fluents: Set[Fluent], initial_values: Dict[FNode, FNode]):
        super().__init__() 
        self.static_fluents = static_fluents
        self.initial_values = initial_values
        self.em = env.expression_manager

    def walk_fluent_exp(self, expression: FNode, args: List[FNode], **kwargs) -> FNode:
        if expression.fluent() in self.static_fluents:
            grounded_fluent = self.em.FluentExp(expression.fluent(), args)
            val = self.initial_values.get(grounded_fluent)
            if val is not None:
                return val
            return self.em.FALSE()
        return self.em.FluentExp(expression.fluent(), args)

    # Required handlers for DagWalker to traverse children
    def walk_and(self, expression, args, **kwargs): return self.em.And(args)
    def walk_or(self, expression, args, **kwargs): return self.em.Or(args)
    def walk_not(self, expression, args, **kwargs): return self.em.Not(args[0])
    def walk_implies(self, expression, args, **kwargs): return self.em.Implies(args[0], args[1])
    def walk_iff(self, expression, args, **kwargs): return self.em.Iff(args[0], args[1])
    def walk_equals(self, expression, args, **kwargs): return self.em.Equals(args[0], args[1])
    def walk_int_constant(self, expression, args, **kwargs): return expression
    def walk_real_constant(self, expression, args, **kwargs): return expression
    def walk_bool_constant(self, expression, args, **kwargs): return expression
    def walk_object_exp(self, expression, args, **kwargs): return expression


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

        self.function_to_id: Dict[str, int] = {}
        self.id_to_function: Dict[int, str] = {}
        self.next_func_id = 0

    def generate_native_problem(self, original_problem: up.model.Problem):
        em = original_problem.environment.expression_manager
        substituter = Substituter(original_problem.environment)
        simplifier = original_problem.environment.simplifier

        # 1. Identify dynamic and static fluents
        dynamic_fluents = set()
        for action in original_problem.actions:
            for eff in action.effects:
                dynamic_fluents.add(eff.fluent.fluent())
        
        static_fluents = set(f for f in original_problem.fluents if f not in dynamic_fluents)
        static_evaluator = StaticEvaluator(original_problem.environment, static_fluents, original_problem.initial_values)

        # 2. Build fluent dictionary ONLY for dynamic fluents
        self._build_fluent_dict(original_problem, em, dynamic_fluents)

        grounded_data = {
            'total_predicates': len(self.fluent_to_id),
            'total_functions': len(self.function_to_id),
            'initial_true': [],
            'initial_false': [],
            'initial_functions': [],
            'goal_rpn': [],
            'actions': [],
            'oneofs': [],
            'deadends': []
        }

        # 3. Load Initial Explicit State (Only for dynamic fluents)
        explicitly_known = set()
        for init_node, value in original_problem.initial_values.items():
            if init_node.fluent() in static_fluents:
                continue

            fluent_str = str(init_node)
            fluent_id = self.fluent_to_id.get(fluent_str)
            if fluent_id is not None:
                if value.is_true():
                    grounded_data['initial_true'].append(fluent_id)
                    explicitly_known.add(fluent_str)
                elif value.is_false():
                    grounded_data['initial_false'].append(fluent_id)
                    explicitly_known.add(fluent_str)
            
            func_id = self.function_to_id.get(fluent_str)
            if func_id is not None:
                if value.is_int_constant() or value.is_real_constant():
                    grounded_data['initial_functions'].append({
                        'id': func_id,
                        'val': float(value.constant_value())
                    })
                    explicitly_known.add(fluent_str)

        # 4. Load Goals (Simplified)
        combined_goal_rpn = []
        for i, g in enumerate(original_problem.goals):
            static_evaluated_g = static_evaluator.walk(g)
            simplified_g = simplifier.simplify(static_evaluated_g)
            combined_goal_rpn.extend(self._compile_to_rpn(simplified_g))
            if i > 0: combined_goal_rpn.append(OP_AND)
        
        grounded_data['goal_rpn'] = combined_goal_rpn

        # 5. Optimistic Reachability Filter
        reachable_facts = set(explicitly_known)
        if hasattr(original_problem, 'oneof_constraints'):
            for group in original_problem.oneof_constraints:
                for f in group:
                    reachable_facts.add(str(f))

        reachable_actions = []
        reachable_action_signatures = set() 
        facts_changed = True

        while facts_changed:
            facts_changed = False
            for action in original_problem.actions:
                param_lists = [list(original_problem.objects(p.type)) for p in action.parameters]
                for combo in itertools.product(*param_lists):
                    signature = (action.name, combo)
                    if signature in reachable_action_signatures:
                        continue 
                    
                    subs = {p: em.ObjectExp(obj) for p, obj in zip(action.parameters, combo)}
                    
                    is_reachable = True
                    simplified_preconditions = []
                    
                    for p in action.preconditions:
                        grounded_p = substituter.substitute(p, subs)
                        static_evaluated_p = static_evaluator.walk(grounded_p)
                        simplified_p = simplifier.simplify(static_evaluated_p)
                        
                        # Strict Pruning: Action mathematically collapses
                        if simplified_p.is_false():
                            is_reachable = False
                            break
                            
                        if not self._is_formula_satisfiable(simplified_p, reachable_facts):
                            is_reachable = False
                            break
                            
                        simplified_preconditions.append(simplified_p)
                    
                    if is_reachable:
                        reachable_actions.append((action, subs, simplified_preconditions))
                        reachable_action_signatures.add(signature)
                        
                        for eff in action.effects:
                            cond_grounded = substituter.substitute(eff.condition, subs)
                            static_evaluated_cond = static_evaluator.walk(cond_grounded)
                            simplified_cond = simplifier.simplify(static_evaluated_cond)
                            
                            if eff.value.is_true() and simplified_cond.is_true():
                                fluent_str = str(substituter.substitute(eff.fluent, subs))
                                if fluent_str not in reachable_facts:
                                    reachable_facts.add(fluent_str)
                                    facts_changed = True
                                    
                        if hasattr(action, 'observed_fluents') and action.observed_fluents:
                            obs_str = str(substituter.substitute(action.observed_fluents[0], subs))
                            if obs_str not in reachable_facts:
                                reachable_facts.add(obs_str)
                                facts_changed = True

        # 6. Compile Survivors to RPN
        action_idx = 0
        for action, subs, simplified_preconditions in reachable_actions:
            actual_params = tuple(subs[p] for p in action.parameters)
            self.action_id_to_up_action[action_idx] = ActionInstance(action, actual_params)
            
            pre_rpn = []
            for i, p in enumerate(simplified_preconditions):
                pre_rpn.extend(self._compile_to_rpn(p))
                if i > 0: pre_rpn.append(OP_AND)

            obs_id = -1
            if hasattr(action, 'observed_fluents') and action.observed_fluents:
                fluent_str = str(substituter.substitute(action.observed_fluents[0], subs))
                if fluent_str in self.fluent_to_id:
                    obs_id = self.fluent_to_id[fluent_str]

            act_dict = {
                'id': action_idx,
                'pre_rpn': pre_rpn,
                'eff_facts': [],
                'eff_vals': [],
                'conditional_effects': [],
                'non_deterministic_effects': [],
                'observe_id': obs_id
            }

            conditional_map = {}

            for eff in action.effects:
                fluent_str = str(substituter.substitute(eff.fluent, subs))
                if fluent_str not in self.fluent_to_id:
                    continue
                
                fluent_id = self.fluent_to_id[fluent_str]
                val = eff.value.is_true() if eff.value.is_bool_constant() else False
                
                cond_grounded = substituter.substitute(eff.condition, subs)
                static_evaluated_cond = static_evaluator.walk(cond_grounded)
                simplified_cond = simplifier.simplify(static_evaluated_cond)
                
                if simplified_cond.is_true():
                    act_dict['eff_facts'].append(fluent_id)
                    act_dict['eff_vals'].append(val)
                elif not simplified_cond.is_false():
                    cond_rpn = tuple(self._compile_to_rpn(simplified_cond))
                    if cond_rpn not in conditional_map:
                        conditional_map[cond_rpn] = {'eff_facts': [], 'eff_vals': []}
                    conditional_map[cond_rpn]['eff_facts'].append(fluent_id)
                    conditional_map[cond_rpn]['eff_vals'].append(val)

            for k, v in conditional_map.items():
                act_dict['conditional_effects'].append({
                    'condition_rpn': list(k),
                    'eff_facts': v['eff_facts'],
                    'eff_vals': v['eff_vals']
                })

            if hasattr(action, 'non_deterministic_effects'):
                for nd_eff in action.non_deterministic_effects:
                    fluent_str = str(substituter.substitute(nd_eff, subs))
                    if fluent_str in self.fluent_to_id:
                        act_dict['non_deterministic_effects'].append(self.fluent_to_id[fluent_str])

            grounded_data['actions'].append(act_dict)
            action_idx += 1

        # 7. Load Constraints and Dead Ends
        if hasattr(original_problem, 'oneof_constraints'):
            for oneof_group in original_problem.oneof_constraints:
                group_ids = [self.fluent_to_id[str(f)] for f in oneof_group if str(f) in self.fluent_to_id]
                if group_ids:
                    grounded_data['oneofs'].append(group_ids)

        for dead_end_list in [getattr(original_problem, 'state_invariants', []), getattr(original_problem, 'dead_ends', [])]:
            for formula in dead_end_list:
                static_evaluated_f = static_evaluator.walk(formula)
                simplified_f = simplifier.simplify(static_evaluated_f)
                
                if not simplified_f.is_false():
                    rpn_list = self._compile_to_rpn(simplified_f)
                    if rpn_list:
                        grounded_data['deadends'].append(rpn_list)

        # Trigger Native Serialization
        native_api.load_problem_to_cpp(grounded_data)
        native_api.print_problem_stats()

    def _is_formula_satisfiable(self, node: FNode, reachable_facts: set) -> bool:
        if node.is_true(): return True
        if node.is_false(): return False

        if node.is_fluent_exp():
            return str(node) in reachable_facts

        if node.is_and():
            for arg in node.args:
                if not self._is_formula_satisfiable(arg, reachable_facts):
                    return False
            return True

        if node.is_or():
            for arg in node.args:
                if self._is_formula_satisfiable(arg, reachable_facts):
                    return True
            return False

        if node.is_not():
            # In an open world scenario, negations are considered logically satisfiable
            # unless mathematically collapsed to False earlier by the simplifier process.
            return True
            
        return True

    def _build_fluent_dict(self, problem: up.model.Problem, em, dynamic_fluents: set):
        for fluent in problem.fluents:
            if fluent not in dynamic_fluents:
                continue
                
            param_lists = [list(problem.objects(p.type)) for p in fluent.signature]
            if not param_lists:
                fluent_node = em.FluentExp(fluent)
                self._route_fluent_by_type(fluent, str(fluent_node))
            else:
                for combo in itertools.product(*param_lists):
                    obj_args = tuple(em.ObjectExp(obj) for obj in combo)
                    fluent_node = em.FluentExp(fluent, obj_args)
                    self._route_fluent_by_type(fluent, str(fluent_node))

    def _route_fluent_by_type(self, fluent, fluent_str: str):
        if fluent.type.is_bool_type():
            if fluent_str not in self.fluent_to_id:
                self.fluent_to_id[fluent_str] = self.next_id
                self.id_to_fluent[self.next_id] = fluent_str
                self.next_id += 1
        elif fluent.type.is_int_type() or fluent.type.is_real_type():
            if fluent_str not in self.function_to_id:
                self.function_to_id[fluent_str] = self.next_func_id
                self.id_to_function[self.next_func_id] = fluent_str
                self.next_func_id += 1

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


def run_my_grounder_and_solve(problem):
    from unified_planning.plans.contingent_plan import ContingentPlanNode
    from unified_planning.model.walkers import Substituter

    print(f"[My Grounder] Grounding and pushing {problem.name} to C++...")
    converter = UpCporConverter()
    converter.generate_native_problem(problem)

    print("[Native API] Invoking YOUR solve_native()...")
    success = native_api.solve_native()

    if not success:
        print("[Native API] CPORSolver returned NO SOLUTION.")
        return None

    print("[Native API] Contingent Plan Found! Extracting tree...")
    substituter = Substituter(problem.environment)
    exp_manager = problem.environment.expression_manager

    def extract_node(node_idx: int) -> ContingentPlanNode:
        if node_idx < 0:
            return None

        action_id = native_api.get_chosen_action(node_idx)
        if action_id == -1:
            return None

        action_instance = converter.action_id_to_up_action[action_id]
        node = ContingentPlanNode(action_instance)

        true_idx = native_api.get_true_child(node_idx)
        false_idx = native_api.get_false_child(node_idx)
        single_idx = native_api.get_single_child(node_idx)

        if true_idx != -1 or false_idx != -1:
            obs_fluent = None
            if hasattr(action_instance.action, 'observed_fluents') and action_instance.action.observed_fluents:
                subs = dict(zip(action_instance.action.parameters, action_instance.actual_parameters))
                obs_fluent = substituter.substitute(action_instance.action.observed_fluents[0], subs)

            if true_idx != -1:
                child_t = extract_node(true_idx)
                if child_t:
                    node.add_child({obs_fluent: exp_manager.TRUE()}, child_t)
            if false_idx != -1:
                child_f = extract_node(false_idx)
                if child_f:
                    node.add_child({obs_fluent: exp_manager.FALSE()}, child_f)
        else:
            child = extract_node(single_idx)
            if child:
                node.add_child({}, child)

        return node

    root_idx = native_api.get_root_node_index()
    root_contingent_node = extract_node(root_idx)
    
    return root_contingent_node

# ====================================================================
# Pure Python Extraction Logic
# ====================================================================
def extract_grounded_problem_data(problem):
    """
    Extracts purely python-native grounded data (IDs and RPN lists) from a unified_planning problem.
    This is the grounding source of truth for the neutral data pipeline.
    """
    from unified_planning.shortcuts import Compiler
    from unified_planning.engines import CompilationKind
    from unified_planning.plans import ActionInstance
    
    print("[Grounder] Compiling grounded problem...")
    with Compiler(problem_kind=problem.kind, compilation_kind=CompilationKind.GROUNDING) as grounder:
        grounded_result = grounder.compile(problem, CompilationKind.GROUNDING)
        g_problem = grounded_result.problem
    
    fluent_to_id = {}
    current_id = 0
    
    # Map fluents to ID
    for f_node in g_problem.initial_values.keys():
        if f_node not in fluent_to_id:
            fluent_to_id[f_node] = current_id
            current_id += 1
            
    initial_true = []
    initial_false = []
    for f_node, val in g_problem.initial_values.items():
        if val.is_true():
            initial_true.append(fluent_to_id[f_node])
        else:
            initial_false.append(fluent_to_id[f_node])
            
    action_map = {}
    actions_data = []
    
    compiler = UpCporConverter() 
    compiler.fluent_to_id = fluent_to_id  # Inject the mapping so compiler can find IDs
    
    print(f"[Grounder] Extracting {len(g_problem.actions)} actions...")
    for i, action in enumerate(g_problem.actions):
        action_map[i] = ActionInstance(action)
        
        # CORRECTED: Iterate over preconditions (since it is a list of FNodes)
        pre_rpn = []
        for p_idx, p in enumerate(action.preconditions):
            pre_rpn.extend(compiler._compile_to_rpn(p))
            if p_idx > 0: 
                pre_rpn.append(OP_AND)
        
        eff_facts = []
        eff_vals = []
        for eff in action.effects:
            if eff.fluent in fluent_to_id:
                eff_facts.append(fluent_to_id[eff.fluent])
                eff_vals.append(eff.value.is_true())
                
        actions_data.append({
            'id': i,
            'pre_rpn': pre_rpn,
            'eff_facts': eff_facts,
            'eff_vals': eff_vals
        })
        
    # CORRECTED: Iterate over goals
    goal_rpn = []
    for g_idx, g in enumerate(g_problem.goals):
        goal_rpn.extend(compiler._compile_to_rpn(g))
        if g_idx > 0:
            goal_rpn.append(OP_AND)
    
    data = {
        'total_predicates': current_id,
        'initial_true': initial_true,
        'initial_false': initial_false,
        'actions': actions_data,
        'goal_rpn': goal_rpn
    }
    
    return data, action_map