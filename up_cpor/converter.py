from __future__ import annotations

import os
import sys

# pythonnet requires a working CLR runtime (Mono on Linux/macOS, .NET
# Framework/Core on Windows) to even import -- on a system where that runtime
# is missing or misconfigured, `import clr` itself raises (empirically,
# clr_loader surfaces this as RuntimeError, not ImportError, so it is not
# caught by the narrower `except ImportError` already used below for the
# CPORLib.dll load). This module's legacy C# CPORLib bridge is optional --
# up_cpor.engine's CPORImpl/SDRImpl route through the native C++ core and
# never import this module -- so a missing runtime here must not prevent
# importing up_cpor.converter itself, only using its C#-backed
# functionality, exactly like the CPORLib.dll fallback immediately below.
try:
    import clr
    import System
    if sys.platform.startswith('win'):
        # use the .NET Framework runtime
        System.Environment.SetEnvironmentVariable("COMPLUS_Version", "v4.0.30319")
    else:
        # use Mono or .NET Core depending on the platform
        if sys.platform.startswith('linux') or sys.platform.startswith('darwin'):
            System.Environment.SetEnvironmentVariable("MONO_ENV_OPTIONS", "--debug")
        elif sys.platform.startswith('openbsd') or sys.platform.startswith('freebsd'):
            System.Environment.SetEnvironmentVariable("DOTNET_ROOT", "/usr/local/share/dotnet")
    clr_runtime_available = True
except Exception as clr_runtime_error:
    clr = None
    System = None
    clr_runtime_available = False
    print(f"WARNING: pythonnet CLR runtime unavailable ({clr_runtime_error}). Legacy C# engine is disabled.")
    print("-> C++ POC components and pure Python modules will continue to work.")

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

# חיפוש חכם ל-DLL של ה-C# בספריות הבנייה
possible_dll_paths = [
    os.path.join(PROJECT_PATH, "CPORLib.dll"),
    os.path.abspath(os.path.join(PROJECT_PATH, "..", "CPORLib", "bin", "Release", "netstandard2.0", "CPORLib.dll")),
    os.path.abspath(os.path.join(PROJECT_PATH, "..", "CPORLib", "bin", "Debug", "netstandard2.0", "CPORLib.dll"))
]

dll_loaded = False
if clr_runtime_available:
    for p in possible_dll_paths:
        if os.path.exists(p):
            clr.AddReference(p)
            dll_loaded = True
            break

    if not dll_loaded:
        print(f"WARNING: CPORLib.dll not found in any of the expected locations.")

try:
    from CPORLib.PlanningModel import Domain, Problem, ParametrizedAction, PlanningAction, Simulator
    from CPORLib.LogicalUtilities import Predicate, ParametrizedPredicate, GroundedPredicate, PredicateFormula, CompoundFormula, Formula
    from CPORLib.Algorithms import CPORPlanner, SDRPlanner
    from CPORLib.Tools import RandomGenerator, Utilities
except ImportError:
    print("WARNING: C# CPORLib failed to load (Mono/WSL issue). Legacy C# engine is disabled.")
    print("-> C++ POC components and pure Python modules will continue to work.")

from unified_planning.model import FNode, OperatorKind, Fluent, Effect
from unified_planning.model.contingent import SensingAction
from unified_planning.plans import ActionInstance
from unified_planning.plans.contingent_plan import ContingentPlanNode
import unified_planning as up
from unified_planning.shortcuts import Bool

from itertools import product
from typing import Dict, Iterable, Optional, Set, Tuple


class CporPlanGraphError(RuntimeError):
    pass

class UpCporConverter:
    @staticmethod
    def set_random_seed(seed: int) -> None:
        RandomGenerator.Init(int(seed))

    def createProblem(self, problem, domain):
        p = Problem(problem.name, domain)
        em = problem.environment.expression_manager

        # Precompute a closed set of hidden fluent expressions for O(1) lookup,
        # including both positive and negated forms.  This avoids creating a
        # Not-expression inside the per-fluent loop.
        raw_hidden = getattr(problem, "hidden_fluents", None) or ()
        hidden_fluents_set = frozenset(raw_hidden) | frozenset(em.Not(f) for f in raw_hidden)

        # Only add True initial values. False values for non-hidden predicates
        # are filled in by PrepareForPlanning() → CompleteKnownState() on the
        # planning path, and the Simulator evaluates absent predicates as False
        # (CWA), so they do not need to be added explicitly here.  Always-
        # constant predicates whose initial value is False (e.g., adj(l1, l2)
        # for non-adjacent pairs in doors15) are also correctly omitted:
        # CompleteKnownState skips always-constant-AND-always-known predicates,
        # and the K-domain writer uses CWA for them in the PDDL output.
        for f, v in problem.initial_values.items():
            if f in hidden_fluents_set:
                continue
            if v.is_true():
                gp = self.__CreatePredicate(f, False, None)
                p.AddKnown(gp)

        compact_hidden_constraints = self.__create_compact_case_hidden_constraints(problem)
        if compact_hidden_constraints is not None:
            for constraint in compact_hidden_constraints:
                cf = self.__CreateFormula(constraint, [], problem)
                p.AddHidden(cf)
        else:
            inferred_case_constraints = self.__infer_missing_case_hidden_literals(problem)

            for c in tuple(problem.or_constraints) + inferred_case_constraints:
                cf = self.__CreateOrFormula(c, [], problem)
                p.AddHidden(cf)

            for c in problem.oneof_constraints:
                cf = self.__CreateOneOfFormula(c, [], problem)
                p.AddHidden(cf)

        goal = CompoundFormula("and")
        for g in problem.goals:
            cp = self.__CreateFormula(g, [], problem)
            goal.AddOperand(cp)
        p.Goal = goal.Simplify()

        return p

    def createCPORPlan(self, c_domain, c_problem):
        solver = CPORPlanner(c_domain, c_problem)
        c_plan = solver.OfflinePlanning()
        return c_plan

    def createSDRPlan(self, c_domain, c_problem):
        solver = SDRPlanner(c_domain, c_problem)
        c_plan = solver.OnlineReplanning()
        return solver, c_plan

    def createSDRSolver(self, c_domain, c_problem):
        solver = SDRPlanner(c_domain, c_problem)
        return solver

    def SDRupdate(self, solver, observation):
        normalized_observation = self.__normalize_sdr_observation(observation)
        applied = solver.SetObservation(normalized_observation)
        return applied

    def __normalize_sdr_observation(self, observation):
        if observation is None or len(observation) == 0:
            return None

        if len(observation) != 1:
            raise ValueError(f"SDR expects at most one grounded observation, got {len(observation)}.")

        fluent_exp, value = next(iter(observation.items()))
        if not isinstance(fluent_exp, FNode) or fluent_exp.node_type != OperatorKind.FLUENT_EXP:
            raise ValueError(f"Unsupported observation key: {fluent_exp!r}")

        if not all(arg.is_object_exp() for arg in fluent_exp.args):
            raise ValueError(f"Observation must be grounded: {fluent_exp}")

        if self.__is_true_observation_value(value):
            return "true"
        if self.__is_false_observation_value(value):
            return "false"

        raise ValueError(f"Unsupported observation value: {value!r}")

    def __is_true_observation_value(self, value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, FNode):
            return value.is_true()
        return False

    def __is_false_observation_value(self, value) -> bool:
        if isinstance(value, bool):
            return not value
        if isinstance(value, FNode):
            return value.is_false()
        return False

    def SDRGet_action(self, solver, problem)  -> ActionInstance:
        c_action = solver.GetAction()
        return self.__convert_SDR_string_to_action_instance(str(c_action), problem)

    def createSDRSimulator(self, problem):
        c_domain = self.createDomain(problem)
        c_problem = self.createProblem(problem, c_domain)
        # SDRPlanner (via PlannerBase) calls PrepareForPlanning() which calls
        # CompleteKnownState(). The Simulator does not, so we call it here so
        # the initial BeliefState has explicit False values for predicates that
        # are initially False. Without this, the SAT solver treats them as
        # unknown and can overflow (MsfLicenseException) on larger domains.
        c_domain.ComputeAlwaysKnown()
        c_problem.CompleteKnownState()
        c_simulator = Simulator(c_domain, c_problem)
        return c_simulator

    def create_c_problem_and_domain(self, problem):
        return problem.actions,

    def SDRSimulatorApply(self, simulator, problem, action: "up.plans.ActionInstance")-> Dict["up.model.FNode", "up.model.FNode"]:
        str_action = str(action)
        str_action = str_action.replace(',', '').replace(')', '').replace('(', ' ')
        str_obser = simulator.Apply(str_action)
        obser = self.__convert_SDR_string_to_observation(str_obser, problem)
        return obser

    def SDRGoal(self, simulator):
        return simulator.GoalReached


    def createDomain(self, problem):
        d = Domain(problem.name)
        for t in problem.user_types:
            if t.father is None:
                d.AddType(t.name)
            else:
                d.AddType(t.name, t.father.name)

        for o in problem.all_objects:
            d.AddConstant(o.name, o.type.name)

        for f in problem.fluents:
            if not self.__fluent_has_groundings(problem, f):
                continue
            pp = self.__CreatePredicate(f, True, [])
            d.AddPredicate(pp)

        for a in problem.actions:
            if not self.__action_has_groundings(problem, a):
                continue
            l = []
            pa = ParametrizedAction(a.name)
            for param in a.parameters:
                l.append(param.name)
                pa.AddParameter(self.__normalize_parameter_name(param.name), param.type.name)
            translated_preconditions = []
            if not a.preconditions is None:
                translated_preconditions.extend(
                    self.__CreateFormula(pre, l, problem) for pre in a.preconditions
                )
            translated_preconditions.extend(
                self.__CreatePromotedPreconditions(a, l, problem)
            )
            if translated_preconditions:
                pa.Preconditions = self.__combine_formulas("and", translated_preconditions)
            if not a.effects is None and len(a.effects) > 0:
                cp = CompoundFormula("and")
                for eff in a.effects:
                    f_eff = self.__CreateEffectFormula(eff, l, problem)
                    cp.SimpleAddOperand(f_eff)
                if len(cp.Operands) > 0:
                    pa.SetEffects(cp)
            if type(a) is SensingAction:
                if not a.observed_fluents is None:
                    for o in a.observed_fluents:
                        pf = self.__CreateFormula(o, l, problem)
                        pa.Observe = pf

            d.AddAction(pa)
        return d

    def createActionTree(self, solution, problem) -> Optional[ContingentPlanNode]:
        if solution is None:
            return None
        return self.__create_action_tree(solution, problem, {}, set())

    def __create_action_tree(
        self,
        solution,
        problem,
        converted_nodes: Dict[int, ContingentPlanNode],
        active_node_ids: Set[int],
    ) -> Optional[ContingentPlanNode]:
        if solution is None:
            return None

        node_id = self.__get_cpor_node_id(solution)
        if node_id in active_node_ids:
            raise CporPlanGraphError(f"Cycle detected while converting CPOR node {node_id}.")
        cached_node = converted_nodes.get(node_id)
        if cached_node is not None:
            return cached_node

        ai = self.__convert_CPOR_string_to_action_instance(str(solution.Action), problem)
        if ai is None:
            return None

        root = ContingentPlanNode(ai)
        converted_nodes[node_id] = root
        active_node_ids.add(node_id)
        try:
            obser = self.__convert_string_to_observation(str(solution.Action), problem)
            if solution.SingleChild:
                child = self.__create_action_tree(
                    solution.SingleChild, problem, converted_nodes, active_node_ids
                )
                if child is not None:
                    root.add_child({}, child)
            if solution.FalseObservationChild and obser:
                child = self.__create_action_tree(
                    solution.FalseObservationChild, problem, converted_nodes, active_node_ids
                )
                if child is not None:
                    observation = {obser: problem.environment.expression_manager.FALSE()}
                    root.add_child(observation, child)
            if solution.TrueObservationChild and obser:
                child = self.__create_action_tree(
                    solution.TrueObservationChild, problem, converted_nodes, active_node_ids
                )
                if child is not None:
                    observation = {obser: problem.environment.expression_manager.TRUE()}
                    root.add_child(observation, child)
        finally:
            active_node_ids.remove(node_id)
        return root

    def __get_cpor_node_id(self, solution) -> int:
        if hasattr(solution, "ID"):
            return int(solution.ID)
        return id(solution)

    def __CreatePredicate(self, f, bAllParameters, lActionParameters) -> ParametrizedPredicate:
        if type(f) is Fluent:
            if (not bAllParameters) and (lActionParameters is None or len(lActionParameters) == 0):
                pp = GroundedPredicate(f.name)
            else:
                pp = ParametrizedPredicate(f.name)
            for param in f.signature:
                bParam = bAllParameters or (param.name in lActionParameters)
                if bParam:
                    pp.AddParameter(self.__normalize_parameter_name(param.name), param.type.name)
                else:
                    pp.AddConstant(param.name, param.type.name)
            return pp
        if type(f) is Effect:
            pp = self.__CreatePredicate(f.fluent, bAllParameters, lActionParameters)
            if self.__is_false_formula_value(f.value):
                pp.Negation = True
            elif not self.__is_true_formula_value(f.value):
                raise ValueError(f"Unsupported effect value: {f.value!r}")
            return pp
        if type(f) is FNode:
            if f.node_type == OperatorKind.FLUENT_EXP:
                predicate_name = f.fluent().name
                predicate_args = f.args
            elif f.node_type == OperatorKind.EQUALS:
                predicate_name = "="
                predicate_args = f.args
            else:
                raise NotImplementedError(f"Unsupported predicate node type: {f.node_type}")

            is_grounded = (
                (not bAllParameters)
                and (lActionParameters is None or len(lActionParameters) == 0)
                and all(arg.is_object_exp() for arg in predicate_args)
            )
            if is_grounded:
                pp = GroundedPredicate(predicate_name)
            else:
                pp = ParametrizedPredicate(predicate_name)
            for arg in predicate_args:
                if arg.is_parameter_exp():
                    param = arg.parameter()
                    pp.AddParameter(self.__normalize_parameter_name(param.name), param.type.name)
                elif arg.is_object_exp():
                    obj = arg.object()
                    pp.AddConstant(obj.name, obj.type.name)
                elif arg.is_variable_exp():
                    variable = arg.variable()
                    pp.AddParameter(self.__normalize_parameter_name(variable.name), variable.type.name)
                else:
                    raise ValueError(f"Unsupported predicate argument: {arg!r}")
            return pp

    def __CreateEffectFormula(self, effect: Effect, lActionParameters, problem) -> Optional[Formula]:
        if not effect.is_assignment():
            raise NotImplementedError(f"Unsupported effect kind: {effect!r}")

        if effect.is_forall():
            return self.__CreateQuantifiedEffectFormula(
                effect,
                lActionParameters,
                problem,
                tuple(effect.forall),
                {},
            )

        return self.__CreateEffectFormulaFromNodes(
            effect.fluent,
            effect.value,
            effect.condition,
            lActionParameters,
            problem,
        )

    def __CreateEffectFormulaFromNodes(
        self,
        fluent_exp: FNode,
        value_exp: FNode,
        condition_exp: FNode,
        lActionParameters,
        problem,
    ) -> Optional[Formula]:
        if value_exp.is_false():
            effect_predicate = self.__CreatePredicate(fluent_exp, False, lActionParameters)
            effect_predicate.Negation = True
        elif value_exp.is_true():
            effect_predicate = self.__CreatePredicate(fluent_exp, False, lActionParameters)
        else:
            raise ValueError(f"Unsupported effect value: {value_exp!r}")

        effect_formula = PredicateFormula(effect_predicate)
        condition_formula = self.__CreateFormula(condition_exp, lActionParameters, problem)
        if self.__is_true_formula(condition_formula):
            return effect_formula
        if self.__is_false_formula(condition_formula):
            return None

        when_formula = CompoundFormula("when")
        when_formula.AddOperand(condition_formula)
        when_formula.AddOperand(effect_formula)
        return when_formula

    def __CreateQuantifiedEffectFormula(
        self,
        effect: Effect,
        lActionParameters,
        problem,
        quantified_variables,
        substitutions,
    ) -> Optional[Formula]:
        if len(quantified_variables) == 0:
            fluent_exp = self.__substitute_expression(effect.fluent, substitutions)
            value_exp = self.__substitute_expression(effect.value, substitutions)
            condition_exp = self.__substitute_expression(effect.condition, substitutions)
            return self.__CreateEffectFormulaFromNodes(
                fluent_exp,
                value_exp,
                condition_exp,
                lActionParameters,
                problem,
            )

        quantified_variable = quantified_variables[0]
        em = problem.environment.expression_manager
        expanded_effects = []
        for obj in self.__iter_objects_for_type(problem, quantified_variable.type):
            next_substitutions = dict(substitutions)
            next_substitutions[quantified_variable] = em.ObjectExp(obj)
            expanded_effect = self.__CreateQuantifiedEffectFormula(
                effect,
                lActionParameters,
                problem,
                quantified_variables[1:],
                next_substitutions,
            )
            if expanded_effect is not None:
                expanded_effects.append(expanded_effect)

        return self.__combine_formulas("and", expanded_effects, empty_value=None)

    def __CreatePromotedPreconditions(self, action, lActionParameters, problem) -> Iterable[Formula]:
        if not action.effects:
            return ()
        if any(not effect.is_assignment() for effect in action.effects):
            return ()
        if any(not effect.is_conditional() for effect in action.effects):
            return ()

        common_conditions = None
        for effect in action.effects:
            current_conditions = {}
            for condition in self.__iter_conjuncts(effect.condition):
                if condition.is_true() or self.__contains_variable_expression(condition):
                    continue
                current_conditions[str(condition)] = condition

            if common_conditions is None:
                common_conditions = current_conditions
            else:
                common_conditions = {
                    key: common_conditions[key]
                    for key in common_conditions.keys() & current_conditions.keys()
                }

            if not common_conditions:
                return ()

        existing_conditions = set()
        for precondition in action.preconditions:
            for condition in self.__iter_conjuncts(precondition):
                existing_conditions.add(str(condition))

        promoted_conditions = []
        for key in sorted(common_conditions.keys()):
            if key in existing_conditions:
                continue
            promoted_conditions.append(
                self.__CreateFormula(common_conditions[key], lActionParameters, problem)
            )
        return tuple(promoted_conditions)

    def __CreateFormula(self, n: FNode, lActionParameters, problem) -> Formula:
        if n.node_type == OperatorKind.BOOL_CONSTANT:
            predicate = Utilities.TRUE_PREDICATE if n.is_true() else Utilities.FALSE_PREDICATE
            return PredicateFormula(predicate)
        if n.node_type == OperatorKind.FLUENT_EXP or n.node_type == OperatorKind.EQUALS:
            pp = self.__CreatePredicate(n, False, lActionParameters)
            pf = PredicateFormula(pp)
            return pf
        if n.node_type == OperatorKind.FORALL or n.node_type == OperatorKind.EXISTS:
            quantified_operator = "and" if n.node_type == OperatorKind.FORALL else "or"
            return self.__CreateQuantifiedFormula(
                n.arg(0),
                tuple(n.variables()),
                quantified_operator,
                lActionParameters,
                problem,
                {},
            )
        if n.node_type == OperatorKind.AND:
            compound_operator = "and"
        elif n.node_type == OperatorKind.OR:
            compound_operator = "or"
        elif n.node_type == OperatorKind.NOT:
            cp = self.__CreateFormula(n.args[0], lActionParameters, problem)
            cp = cp.Negate()
            return cp
        else:
            raise NotImplementedError(f"Unsupported formula node type: {n.node_type}")

        return self.__combine_formulas(
            compound_operator,
            (self.__CreateFormula(nSub, lActionParameters, problem) for nSub in n.args),
        )

    def __CreateQuantifiedFormula(
        self,
        body: FNode,
        quantified_variables,
        operator: str,
        lActionParameters,
        problem,
        substitutions,
    ) -> Formula:
        if len(quantified_variables) == 0:
            substituted_body = self.__substitute_expression(body, substitutions)
            return self.__CreateFormula(substituted_body, lActionParameters, problem)

        quantified_variable = quantified_variables[0]
        em = problem.environment.expression_manager
        expanded_formulas = []
        for obj in self.__iter_objects_for_type(problem, quantified_variable.type):
            next_substitutions = dict(substitutions)
            next_substitutions[quantified_variable] = em.ObjectExp(obj)
            expanded_formulas.append(
                self.__CreateQuantifiedFormula(
                    body,
                    quantified_variables[1:],
                    operator,
                    lActionParameters,
                    problem,
                    next_substitutions,
                )
            )
        return self.__combine_formulas(operator, expanded_formulas)

    def __substitute_expression(self, expression: FNode, substitutions):
        if len(substitutions) == 0:
            return expression
        return expression.substitute(substitutions)

    def __iter_objects_for_type(self, problem, up_type) -> Iterable:
        for obj in sorted(problem.all_objects, key=lambda current_object: current_object.name):
            if self.__object_matches_type(obj, up_type):
                yield obj

    def __type_has_objects(self, problem, up_type) -> bool:
        return any(self.__iter_objects_for_type(problem, up_type))

    def __fluent_has_groundings(self, problem, fluent: Fluent) -> bool:
        return all(self.__type_has_objects(problem, parameter.type) for parameter in fluent.signature)

    def __action_has_groundings(self, problem, action) -> bool:
        return all(self.__type_has_objects(problem, parameter.type) for parameter in action.parameters)

    def __object_matches_type(self, obj, up_type) -> bool:
        current_type = obj.type
        while current_type is not None:
            if current_type == up_type:
                return True
            current_type = getattr(current_type, "father", None)
        return False

    def __combine_formulas(
        self,
        operator: str,
        formulas,
        *,
        empty_value: Optional[Formula] = None,
    ) -> Optional[Formula]:
        normalized_formulas = [formula for formula in formulas if formula is not None]
        if len(normalized_formulas) == 0:
            if empty_value is not None:
                return empty_value
            predicate = Utilities.TRUE_PREDICATE if operator == "and" else Utilities.FALSE_PREDICATE
            return PredicateFormula(predicate)
        if len(normalized_formulas) == 1:
            return normalized_formulas[0]

        compound_formula = CompoundFormula(operator)
        for formula in normalized_formulas:
            compound_formula.SimpleAddOperand(formula)
        return compound_formula

    def __normalize_parameter_name(self, name: str) -> str:
        return name if name.startswith("?") else f"?{name}"

    def __is_hidden_initial_fluent(self, problem, fluent_exp: FNode) -> bool:
        hidden_fluents = getattr(problem, "hidden_fluents", None)
        if hidden_fluents is None:
            return False
        em = problem.environment.expression_manager
        return fluent_exp in hidden_fluents or em.Not(fluent_exp) in hidden_fluents

    def __iter_conjuncts(self, formula: FNode) -> Iterable[FNode]:
        if formula.node_type == OperatorKind.AND:
            for argument in formula.args:
                yield from self.__iter_conjuncts(argument)
            return
        yield formula

    def __contains_variable_expression(self, formula: FNode) -> bool:
        if formula.is_variable_exp():
            return True
        return any(self.__contains_variable_expression(argument) for argument in formula.args)

    def __is_true_formula_value(self, value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, FNode):
            return value.is_true()
        return False

    def __is_false_formula_value(self, value) -> bool:
        if isinstance(value, bool):
            return not value
        if isinstance(value, FNode):
            return value.is_false()
        return False

    def __is_true_formula(self, formula: Formula) -> bool:
        return isinstance(formula, PredicateFormula) and formula.Predicate == Utilities.TRUE_PREDICATE

    def __is_false_formula(self, formula: Formula) -> bool:
        return isinstance(formula, PredicateFormula) and formula.Predicate == Utilities.FALSE_PREDICATE

    def __CreateOrFormula(self, n, lActionParameters, problem) -> Formula:
        cp = CompoundFormula("or")
        for nSub in n:
            fSub = self.__CreateFormula(nSub, lActionParameters, problem)
            cp.SimpleAddOperand(fSub)
        return cp

    def __CreateOneOfFormula(self, n, lActionParameters, problem) -> Formula:
        cp = CompoundFormula("oneof")
        for nSub in n:
            fSub = self.__CreateFormula(nSub, lActionParameters, problem)
            cp.SimpleAddOperand(fSub)
        return cp

    def __infer_missing_case_hidden_literals(self, problem) -> Tuple[Tuple[FNode, FNode], ...]:
        hidden_positive_fluents = {
            str(hidden_fluent): hidden_fluent
            for hidden_fluent in problem.hidden_fluents
            if isinstance(hidden_fluent, FNode)
            and hidden_fluent.node_type == OperatorKind.FLUENT_EXP
            and not self.__is_case_tag_expression(hidden_fluent)
        }
        if len(hidden_positive_fluents) == 0:
            return tuple()

        case_tag_groups = tuple(self.__iter_case_tag_groups(problem))
        if len(case_tag_groups) == 0:
            return tuple()

        explicit_case_assignments: Dict[str, Dict[str, FNode]] = {}
        case_partitioned_fluents: Set[str] = set()
        for case_tag_group in case_tag_groups:
            for case_tag in case_tag_group:
                explicit_case_assignments.setdefault(str(case_tag), {})

        for constraint in problem.or_constraints:
            parsed_assignment = self.__parse_case_assignment(constraint, hidden_positive_fluents)
            if parsed_assignment is None:
                continue
            case_tag, fluent_exp, literal = parsed_assignment
            explicit_case_assignments[str(case_tag)][str(fluent_exp)] = literal
            case_partitioned_fluents.add(str(fluent_exp))

        if len(case_partitioned_fluents) == 0:
            return tuple()

        inferred_constraints = []
        em = problem.environment.expression_manager
        for case_tag_group in case_tag_groups:
            for case_tag in case_tag_group:
                case_assignments = explicit_case_assignments[str(case_tag)]
                for fluent_key in sorted(case_partitioned_fluents):
                    if fluent_key in case_assignments:
                        continue
                    hidden_fluent = hidden_positive_fluents[fluent_key]
                    default_value = problem.initial_value(hidden_fluent)
                    default_is_true = default_value is not None and default_value.is_true()
                    default_literal = hidden_fluent if default_is_true else em.Not(hidden_fluent)
                    inferred_constraints.append((em.Not(case_tag), default_literal))

        return tuple(inferred_constraints)

    def __create_compact_case_hidden_constraints(self, problem) -> Optional[Tuple[FNode, ...]]:
        hidden_positive_fluents = {
            str(hidden_fluent): hidden_fluent
            for hidden_fluent in problem.hidden_fluents
            if isinstance(hidden_fluent, FNode)
            and hidden_fluent.node_type == OperatorKind.FLUENT_EXP
            and not self.__is_case_tag_expression(hidden_fluent)
        }
        if len(hidden_positive_fluents) == 0:
            return None

        case_tag_groups = tuple(self.__iter_case_tag_groups(problem))
        if len(case_tag_groups) != len(problem.oneof_constraints) or len(case_tag_groups) != 1:
            return None

        case_tag_group = case_tag_groups[0]
        inferred_case_constraints = self.__infer_missing_case_hidden_literals(problem)
        all_case_constraints = tuple(problem.or_constraints) + inferred_case_constraints

        case_assignments: Dict[str, Dict[str, FNode]] = {
            str(case_tag): {} for case_tag in case_tag_group
        }
        case_partitioned_fluents: Set[str] = set()
        for constraint in all_case_constraints:
            parsed_assignment = self.__parse_case_assignment(constraint, hidden_positive_fluents)
            if parsed_assignment is None:
                return None
            case_tag, fluent_exp, literal = parsed_assignment
            case_assignments[str(case_tag)][str(fluent_exp)] = literal
            case_partitioned_fluents.add(str(fluent_exp))

        if len(case_partitioned_fluents) == 0:
            return None

        ordered_fluents = [
            hidden_positive_fluents[fluent_key]
            for fluent_key in sorted(case_partitioned_fluents)
        ]
        allowed_assignments = set()
        for case_tag in case_tag_group:
            case_assignment = case_assignments[str(case_tag)]
            if any(str(hidden_fluent) not in case_assignment for hidden_fluent in ordered_fluents):
                return None
            allowed_assignments.add(
                tuple(
                    not case_assignment[str(hidden_fluent)].is_not()
                    for hidden_fluent in ordered_fluents
                )
            )

        total_assignments = 1 << len(ordered_fluents)
        excluded_assignments_count = total_assignments - len(allowed_assignments)
        if excluded_assignments_count < 0 or excluded_assignments_count >= len(case_tag_group):
            return None

        em = problem.environment.expression_manager
        if excluded_assignments_count == 0:
            return tuple(
                em.Or(hidden_fluent, em.Not(hidden_fluent))
                for hidden_fluent in ordered_fluents
            )

        compact_constraints = []
        for assignment in product((False, True), repeat=len(ordered_fluents)):
            if assignment in allowed_assignments:
                continue

            clause_literals = []
            for hidden_fluent, value in zip(ordered_fluents, assignment):
                clause_literals.append(em.Not(hidden_fluent) if value else hidden_fluent)

            if len(clause_literals) == 1:
                compact_constraints.append(clause_literals[0])
            else:
                compact_constraints.append(em.Or(clause_literals))

        return tuple(compact_constraints)

    def __iter_case_tag_groups(self, problem):
        for oneof_constraint in problem.oneof_constraints:
            case_tag_group = []
            for item in oneof_constraint:
                if not isinstance(item, FNode) or item.node_type != OperatorKind.FLUENT_EXP:
                    case_tag_group = []
                    break
                if not self.__is_case_tag_expression(item):
                    case_tag_group = []
                    break
                case_tag_group.append(item)
            if len(case_tag_group) > 0:
                yield tuple(sorted(case_tag_group, key=str))

    def __parse_case_assignment(self, constraint, hidden_positive_fluents):
        if len(constraint) != 2:
            return None

        case_tag = None
        literal = None
        for item in constraint:
            if (
                isinstance(item, FNode)
                and item.is_not()
                and item.arg(0).node_type == OperatorKind.FLUENT_EXP
                and self.__is_case_tag_expression(item.arg(0))
            ):
                case_tag = item.arg(0)
            else:
                literal = item

        if case_tag is None or literal is None:
            return None

        if literal.node_type == OperatorKind.FLUENT_EXP:
            fluent_exp = literal
        elif literal.is_not() and literal.arg(0).node_type == OperatorKind.FLUENT_EXP:
            fluent_exp = literal.arg(0)
        else:
            return None

        if self.__is_case_tag_expression(fluent_exp):
            return None
        if str(fluent_exp) not in hidden_positive_fluents:
            return None

        return case_tag, fluent_exp, literal

    def __is_case_tag_expression(self, fluent_exp: FNode) -> bool:
        return (
            isinstance(fluent_exp, FNode)
            and fluent_exp.node_type == OperatorKind.FLUENT_EXP
            and fluent_exp.fluent().name.startswith("possible_initial_state_case_")
        )

    def __convert_CPOR_string_to_action_instance(self, string, problem) -> 'up.plans.InstantaneousAction':
        if string != 'None':
            assert string[0] == "(" and string[-1] == ")"
            list_str = string[1:-1].replace(":", "").replace('~', ' ').split("\n")
            ac = list_str[0].split(" ")
            action_name = ac[1]
            action_param = ac[2:]
            return self.__convert_string_action_to_action_instance(action_name, action_param, problem)

    def __convert_SDR_string_to_action_instance(self, action_string, problem) -> 'up.plans.InstantaneousAction':
        if action_string != 'None':
            ac = action_string.split()
            action_name = ac[0]
            action_param = ac[1:]
            return self.__convert_string_action_to_action_instance(action_name, action_param, problem)

    def __convert_string_action_to_action_instance(self, action_name, action_param, problem) -> 'up.plans.InstantaneousAction':
        action = problem.action(action_name)
        expr_manager = problem.environment.expression_manager
        param = tuple(expr_manager.ObjectExp(problem.object(o_name)) for o_name in action_param)
        return ActionInstance(action, param)

    def __convert_string_to_observation(self, string, problem):
        if string is not None and string != 'None' and ":observe" in string:
            ob = string.replace("\n", " ").replace(")", "").replace("(", "").split(":observe ")[1]
            obs = ob.split()
            expr_manager = problem.environment.expression_manager
            obse = problem.fluent(obs[0])
            location = tuple(expr_manager.ObjectExp(problem.object(o_name)) for o_name in obs[1:])
            obresv = expr_manager.FluentExp(obse, location)
            return obresv
        return None

    def __convert_SDR_string_to_observation(self, string, problem):
        if string is not None and string != 'None':
            ob = string.replace(")", "").replace("(", "")
            obs = ob.split()
            if obs[0] == "not":
                obs = obs[1:]
                boolean = False
            else:
                boolean = True
            expr_manager = problem.environment.expression_manager
            obse = problem.fluent(obs[0])
            location = tuple(expr_manager.ObjectExp(problem.object(o_name)) for o_name in obs[1:])
            obresv = expr_manager.FluentExp(obse, location)
            return {obresv: Bool(boolean)}
        return None
