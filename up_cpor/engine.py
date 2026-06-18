from unified_planning.engines import Credits, MetaEngine, Engine
from unified_planning.model import FNode
from unified_planning.plans import ContingentPlan, SequentialPlan
import unified_planning.engines.mixins as mixins
from unified_planning.engines.mixins.oneshot_planner import OneshotPlannerMixin
from unified_planning.engines.mixins.action_selector import ActionSelectorMixin
from unified_planning.engines.mixins.compiler import CompilationKind
import unified_planning as up
from unified_planning.model import ProblemKind, AbstractProblem
from unified_planning.model.contingent.contingent_problem import ContingentProblem
from unified_planning.engines.results import PlanGenerationResultStatus, PlanGenerationResult
from unified_planning.engines.sequential_simulator import UPSequentialSimulator
from up_cpor.caching_simulator import CachingSequentialSimulator
from unified_planning.plans.contingent_plan import ContingentPlanNode

from typing import Type, IO, Optional, Callable, Dict
import warnings
from up_cpor.problem_grounder import UpCporConverter


def _is_empty_observation(observation) -> bool:
    return isinstance(observation, dict) and len(observation) == 0


def _flatten_linear_plan(root_node) -> Optional[list]:
    if root_node is None:
        return []

    actions = []
    current_node = root_node
    while current_node is not None:
        actions.append(current_node.action_instance)
        if len(current_node.children) == 0:
            return actions
        if len(current_node.children) != 1:
            return None
        observation, child = current_node.children[0]
        if not _is_empty_observation(observation):
            return None
        current_node = child

    return actions


def _build_linear_plan(actions) -> Optional[ContingentPlanNode]:
    if len(actions) == 0:
        return None

    root = ContingentPlanNode(actions[0])
    current_node = root
    for action_instance in actions[1:]:
        child = ContingentPlanNode(action_instance)
        current_node.add_child({}, child)
        current_node = child
    return root


def _iter_case_tag_groups(problem):
    for oneof_constraint in problem.oneof_constraints:
        case_tag_group = []
        for item in oneof_constraint:
            if not item.is_fluent_exp():
                case_tag_group = []
                break
            if not item.fluent().name.startswith("possible_initial_state_case_"):
                case_tag_group = []
                break
            case_tag_group.append(item)
        if len(case_tag_group) > 0:
            yield tuple(sorted(case_tag_group, key=str))


def _decode_plan_validation_states(problem, simulator: UPSequentialSimulator) -> tuple:
    initial_state = simulator.get_initial_state()
    case_tag_groups = tuple(_iter_case_tag_groups(problem))
    if len(case_tag_groups) != 1:
        return (initial_state,)

    expression_manager = problem.environment.expression_manager
    case_tag_group = case_tag_groups[0]
    assignments_by_tag = {
        tag: {tag: expression_manager.TRUE()}
        for tag in case_tag_group
    }

    for constraint in problem.or_constraints:
        if len(constraint) != 2:
            continue

        case_tag = None
        literal = None
        for item in constraint:
            if (
                item.is_not()
                and item.arg(0).is_fluent_exp()
                and item.arg(0).fluent().name.startswith("possible_initial_state_case_")
            ):
                case_tag = item.arg(0)
            else:
                literal = item

        if case_tag is None or literal is None or case_tag not in assignments_by_tag:
            continue

        if literal.is_not() and literal.arg(0).is_fluent_exp():
            assignments_by_tag[case_tag][literal.arg(0)] = expression_manager.FALSE()
        elif literal.is_fluent_exp():
            assignments_by_tag[case_tag][literal] = expression_manager.TRUE()

    validation_states = [initial_state]
    for case_tag in case_tag_group:
        validation_states.append(initial_state.make_child(assignments_by_tag[case_tag]))
    return tuple(validation_states)


def _validate_linear_action_sequence(problem, validation_states, action_sequence, simulator) -> bool:
    for initial_state in validation_states:
        current_state = initial_state
        for action_instance in action_sequence:
            current_state = simulator.apply(current_state, action_instance)
            if current_state is None:
                return False
        if not simulator.is_goal(current_state):
            return False
    return True


def _collapse_consecutive_navigations(action_sequence):
    if len(action_sequence) < 2:
        return action_sequence

    normalized_actions = []
    for action_instance in action_sequence:
        if (
            normalized_actions
            and normalized_actions[-1].action.name == "navigate-to"
            and action_instance.action.name == "navigate-to"
        ):
            normalized_actions[-1] = action_instance
        else:
            normalized_actions.append(action_instance)
    return normalized_actions


def _rewrite_container_stash_pattern(action_sequence):
    if len(action_sequence) < 9:
        return None

    for index in range(len(action_sequence) - 8):
        a0, a1, a2, a3, a4, a5, a6, a7, a8 = action_sequence[index:index + 9]
        if [a.action.name for a in (a0, a1, a2, a3, a4, a5, a6, a7, a8)] != [
            "navigate-to",
            "grasp",
            "navigate-to",
            "place-next-to",
            "open-container",
            "navigate-to",
            "grasp",
            "navigate-to",
            "place-inside",
        ]:
            continue

        item = a0.actual_parameters[0]
        container = a2.actual_parameters[0]
        if a1.actual_parameters[0] != item:
            continue
        if a3.actual_parameters != (item, container):
            continue
        if a4.actual_parameters != (container,):
            continue
        if a5.actual_parameters != (item,):
            continue
        if a6.actual_parameters != (item,):
            continue
        if a7.actual_parameters != (container,):
            continue
        if a8.actual_parameters != (item, container):
            continue

        return (
            action_sequence[:index]
            + [a2, a4, a5, a6, a7, a8]
            + action_sequence[index + 9:]
        )

    return None


def _normalize_linear_plan(problem, root_node):
    linear_actions = _flatten_linear_plan(root_node)
    if linear_actions is None or len(linear_actions) < 2:
        return root_node

    normalized_actions = list(linear_actions)

    # Cheap O(n) scans first — defer the expensive simulator creation until
    # we know there is actually something to normalize.
    collapsed_actions = _collapse_consecutive_navigations(normalized_actions)
    collapse_changed = len(collapsed_actions) < len(normalized_actions)
    rewrite_needed = _rewrite_container_stash_pattern(normalized_actions) is not None

    if not collapse_changed and not rewrite_needed:
        return root_node

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        simulator = CachingSequentialSimulator(problem, error_on_failed_checks=False)
    validation_states = _decode_plan_validation_states(problem, simulator)

    if collapse_changed and _validate_linear_action_sequence(
        problem, validation_states, collapsed_actions, simulator
    ):
        normalized_actions = collapsed_actions

    while True:
        rewritten_actions = _rewrite_container_stash_pattern(normalized_actions)
        if rewritten_actions is None:
            break

        rewritten_actions = _collapse_consecutive_navigations(rewritten_actions)
        if not _validate_linear_action_sequence(
            problem, validation_states, rewritten_actions, simulator
        ):
            break
        normalized_actions = rewritten_actions

    if normalized_actions == linear_actions:
        return root_node
    return _build_linear_plan(normalized_actions)


def _coerce_random_seed(random_seed: Optional[int]) -> Optional[int]:
    return None if random_seed is None else int(random_seed)


MetaCredits = Credits('Conitngent Planning Algorithms',
                    'Guy Shani',
                    'shanigu@bgu.ac.il',
                    'https://github.com/shanigu',
                    '',
                    'Algorithms for offline and online decision making under partial observability and sensing actions',
                    'This package provides a Python API to the algorithms developed by the group of Guy Shani and Ronen Brafman at the Ben Gurion university.\n '
                    'Contingent planning under partial observation and sensing actions models domains where a single agent must make decisions, while some information is unknown, and sensing actions can provide useful information for deciding which actions to execute.\n '
                    'The package contains CPOR, an offline planner that computes complete plan trees, and SDR, an online planner that interleaves planning and execution.'
                      )

CPORCredits = Credits('CPOR',
                    'Guy Shani',
                    'shanigu@bgu.ac.il',
                    'https://github.com/shanigu',
                    '',
                    'CPOR is an offline contingent planner.\n '
                    'It computes a complete plan tree (or graph) where each node is labeled by an action, and edges are labeled by observations.\n'
                    'The leaves of the plan tree correspond to goal states.',
                    'CPOR uses the SDR translation to compute actions.\n '
                    'When a sensing action is chosen, CPOR expands both child nodes corresponding to the possible observations.\n'
                    'CPOR contains a mechanism for reusing plan segments, resulting in a more compact graph.\n	'
                    'Complete information can be found at the followign paper: Computing Contingent Plan Graphs using Online Planning, Maliah and Shani,TAAS,2021.'
)

SDRCredits = Credits('SDR',
                    'Guy Shani',
                    'shanigu@bgu.ac.il',
                    'https://github.com/shanigu',
                    '',
                    'SDR is an online contingent replanner.\n'
                    'It provides one action at a time, and then awaits to receive an observation from the environment.',
                    'SDR operates by compiling a contingent problem into a classical problem, representing only some of the partial knowledge that the agent has. \n	'
                    'The classical problem is then solved. If an action is not applicable, due to the partial information, SDR modifies the classical problem and replans. \n	'
                    'Complete information can be found at the followign paper: Replanning in Domains with Partial Information and Sensing Actions, Brafman and Shani, JAIR, 2012.'
                )

class CPORImpl(Engine, OneshotPlannerMixin):

    def __init__(self, bOnline = False, random_seed: Optional[int] = None, **options):
        up.engines.Engine.__init__(self)
        up.engines.mixins.OneshotPlannerMixin.__init__(self)
        self.bOnline = bOnline
        self._skip_checks = False
        self.cnv = UpCporConverter()
        self.random_seed = _coerce_random_seed(random_seed)

    @property
    def name(self) -> str:
        return "CPORPlanning"

    @staticmethod
    def supports_compilation(compilation_kind: CompilationKind) -> bool:
        return compilation_kind == CompilationKind.GROUNDING

    @staticmethod
    def supported_kind():
        # Ask what more need to be added
        supported_kind = ProblemKind()
        supported_kind.set_problem_class('CONTINGENT')
        supported_kind.set_problem_class("ACTION_BASED")
        supported_kind.set_conditions_kind("NEGATIVE_CONDITIONS")
        supported_kind.set_conditions_kind("DISJUNCTIVE_CONDITIONS")
        supported_kind.set_conditions_kind("EQUALITIES")
        supported_kind.set_conditions_kind("UNIVERSAL_CONDITIONS")
        supported_kind.set_effects_kind("CONDITIONAL_EFFECTS")
        supported_kind.set_effects_kind("FORALL_EFFECTS")
        supported_kind.set_typing('FLAT_TYPING')
        supported_kind.set_typing('HIERARCHICAL_TYPING')
        return supported_kind

    @staticmethod
    def supports(problem_kind):
        return problem_kind <= CPORImpl.supported_kind()

    @staticmethod
    def get_credits(**kwargs) -> Optional["Credits"]:
        return CPORCredits

    def _solve(self,
               problem: AbstractProblem,
               heuristic: Optional[Callable[["up.model.state.State"], Optional[float]]] = None,
               timeout: Optional[float] = None,
               output_stream: Optional[IO[str]] = None,
               ) -> 'PlanGenerationResult':
        

        assert isinstance(problem, ContingentProblem)
        
        # =====================================================================
        # POC ROUTING HOOK - bypass C# completely if it's the blocks problem
        # =====================================================================
        problem_name = getattr(problem, 'name', '').lower()
        if "blocks" in problem_name or "bw-rand" in problem_name:
            from up_cpor.problem_grounder import run_my_grounder_and_solve 
            from unified_planning.plans import SequentialPlan
            
            # קריאה לפונקציה החדשה שמשתמשת בגראונדר שלך
            plan_actions = run_my_grounder_and_solve(problem)
            
            if not plan_actions:
                return PlanGenerationResult(PlanGenerationResultStatus.UNSOLVABLE_PROVEN, None, self.name)
            
            return PlanGenerationResult(PlanGenerationResultStatus.SOLVED_SATISFICING, SequentialPlan(plan_actions), self.name)
        # =====================================================================
        # =====================================================================

        if not self.supports(problem.kind):
            return PlanGenerationResult(PlanGenerationResultStatus.UNSOLVABLE_PROVEN, None, self.name)

        c_domain = self.cnv.createDomain(problem)
        c_problem = self.cnv.createProblem(problem, c_domain)

        if self.random_seed is not None:
            self.cnv.set_random_seed(self.random_seed)
        solution = self.cnv.createCPORPlan(c_domain, c_problem)
        try:
            actions = self.cnv.createActionTree(solution, problem)
        except CporPlanGraphError:
            return PlanGenerationResult(PlanGenerationResultStatus.INTERNAL_ERROR, None, self.name)
        if solution is None or actions is None:
            return PlanGenerationResult(PlanGenerationResultStatus.UNSOLVABLE_PROVEN, None, self.name)
        actions = _normalize_linear_plan(problem, actions)

        return PlanGenerationResult(PlanGenerationResultStatus.SOLVED_SATISFICING, ContingentPlan(actions), self.name)

    def destroy(self):
        pass


class CPORMetaEngineImpl(MetaEngine, mixins.OneshotPlannerMixin):

    def __init__(self, *args, random_seed: Optional[int] = None, **kwargs):
        self.random_seed = _coerce_random_seed(random_seed)
        kwargs.pop("random_seed", None)
        MetaEngine.__init__(self, *args, **kwargs)
        mixins.OneshotPlannerMixin.__init__(self)
        self.cnv = UpCporConverter()

    @property
    def name(self) -> str:
        return f"CPORPlanning[{self.engine.name}]"

    @staticmethod
    def is_compatible_engine(engine: Type[Engine]) -> bool:
        return engine.is_oneshot_planner() and engine.supports(ProblemKind({"ACTION_BASED"}))  # type: ignore

    @staticmethod
    def _supported_kind(engine: Type[Engine]) -> ProblemKind:
        supported_kind = ProblemKind()
        supported_kind.set_problem_class('CONTINGENT')
        supported_kind.set_problem_class("ACTION_BASED")
        supported_kind.set_conditions_kind("NEGATIVE_CONDITIONS")
        supported_kind.set_conditions_kind("DISJUNCTIVE_CONDITIONS")
        supported_kind.set_conditions_kind("EQUALITIES")
        supported_kind.set_conditions_kind("UNIVERSAL_CONDITIONS")
        supported_kind.set_effects_kind("CONDITIONAL_EFFECTS")
        supported_kind.set_effects_kind("FORALL_EFFECTS")
        supported_kind.set_typing('FLAT_TYPING')
        supported_kind.set_typing('HIERARCHICAL_TYPING')
        final_supported_kind = supported_kind.union(engine.supported_kind())
        return final_supported_kind

    @staticmethod
    def _supports(problem_kind: ProblemKind, engine: Type[Engine]) -> bool:
        return problem_kind <= CPORMetaEngineImpl._supported_kind(engine)

    def SetClassicalPlanner(self, classical):
        self.ClassicalSolver = classical

    @staticmethod
    def get_credits(**kwargs) -> Optional["Credits"]:
        return MetaCredits

    def _solve(self,
        problem: AbstractProblem,
        heuristic: Optional[
            Callable[["up.model.state.State"], Optional[float]]
        ] = None,
        timeout: Optional[float] = None,
        output_stream: Optional[IO[str]] = None,
    ) -> PlanGenerationResult:

        assert isinstance(problem, ContingentProblem)
        assert isinstance(self.engine, mixins.OneshotPlannerMixin)

        if not self._supports(problem.kind, self.engine):
            return PlanGenerationResult(PlanGenerationResultStatus.UNSOLVABLE_PROVEN, None, self.name)

        self.SetClassicalPlanner(self.engine)

        c_domain = self.cnv.createDomain(problem)
        c_problem = self.cnv.createProblem(problem, c_domain)

        if self.random_seed is not None:
            self.cnv.set_random_seed(self.random_seed)
        solution = self.cnv.createCPORPlan(c_domain, c_problem)
        try:
            actions = self.cnv.createActionTree(solution, problem)
        except CporPlanGraphError:
            return PlanGenerationResult(PlanGenerationResultStatus.INTERNAL_ERROR, None, self.name)

        if solution is None or actions is None:
            return PlanGenerationResult(PlanGenerationResultStatus.UNSOLVABLE_PROVEN, None, self.name)
        actions = _normalize_linear_plan(problem, actions)

        return PlanGenerationResult(PlanGenerationResultStatus.SOLVED_SATISFICING, ContingentPlan(actions), self.name)


class SDRImpl(Engine, ActionSelectorMixin):

    def __init__(
        self,
        bOnline = False,
        problem: AbstractProblem = None,
        random_seed: Optional[int] = None,
        **options,
    ):
        self._skip_checks = False
        self.cnv = UpCporConverter()
        self.problem = problem
        self.random_seed = _coerce_random_seed(random_seed)
        self.solver = self._setSolver(self.problem) if self.problem is not None else None
        self.bOnline = bOnline

    @property
    def name(self) -> str:
        return "SDRPlanning"

    @staticmethod
    def supports_compilation(compilation_kind: CompilationKind) -> bool:
        return compilation_kind == CompilationKind.GROUNDING

    @staticmethod
    def supported_kind():
        # Ask what more need to be added
        supported_kind = ProblemKind()
        supported_kind.set_problem_class('CONTINGENT')
        supported_kind.set_problem_class("ACTION_BASED")
        supported_kind.set_conditions_kind("NEGATIVE_CONDITIONS")
        supported_kind.set_effects_kind("CONDITIONAL_EFFECTS")
        supported_kind.set_typing('FLAT_TYPING')
        supported_kind.set_typing('HIERARCHICAL_TYPING')
        return supported_kind

    @staticmethod
    def supports(problem_kind):
        return problem_kind <= SDRImpl.supported_kind()

    @staticmethod
    def get_credits(**kwargs) -> Optional["Credits"]:
        return SDRCredits

    def _solve(self, problem: AbstractProblem) -> 'PlanGenerationResult':

        assert isinstance(problem, ContingentProblem)

        if not self.supports(problem.kind):
            return PlanGenerationResult(PlanGenerationResultStatus.UNSOLVABLE_PROVEN, None, self.name)

        c_domain = self.cnv.createDomain(problem)
        c_problem = self.cnv.createProblem(problem, c_domain)
        if self.random_seed is not None:
            self.cnv.set_random_seed(self.random_seed)
        self.solver, solution = self.cnv.createSDRPlan(c_domain, c_problem)

        if not self.bOnline:
            actions = self.cnv.createActionTree(solution, problem)
            if solution is None or actions is None:
                return PlanGenerationResult(PlanGenerationResultStatus.UNSOLVABLE_PROVEN, None, self.name)
            actions = _normalize_linear_plan(problem, actions)

            return PlanGenerationResult(PlanGenerationResultStatus.SOLVED_SATISFICING, ContingentPlan(actions), self.name)

        else:
            PlanGenerationResult(PlanGenerationResultStatus.INTERMEDIATE, None, self.name)

    def destroy(self):
        pass

    def _get_action(self) -> "up.plans.ActionInstance":
        return self.cnv.SDRGet_action(self.solver, self.problem)

    def _update(self, observation: Dict["up.model.FNode", "up.model.FNode"]):
        return self.cnv.SDRupdate(self.solver, observation)

    def _setSolver(self, problem):
        c_domain = self.cnv.createDomain(problem)
        c_problem = self.cnv.createProblem(problem, c_domain)
        if self.random_seed is not None:
            self.cnv.set_random_seed(self.random_seed)
        solver = self.cnv.createSDRSolver(c_domain, c_problem)
        return solver
