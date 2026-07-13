from unified_planning.engines import Credits, Engine, MetaEngine
from unified_planning.plans import ContingentPlan
from unified_planning.engines.mixins.oneshot_planner import OneshotPlannerMixin
from unified_planning.engines.mixins.action_selector import ActionSelectorMixin
from unified_planning.engines.mixins.compiler import CompilationKind
import unified_planning as up
import unified_planning.engines.mixins as mixins
from unified_planning.model import ProblemKind, AbstractProblem
from unified_planning.model.contingent.contingent_problem import ContingentProblem
from unified_planning.engines.results import PlanGenerationResultStatus, PlanGenerationResult

from typing import Type, IO, Optional, Callable, Dict

from up_cpor import native_api


def _coerce_random_seed(random_seed: Optional[int]) -> Optional[int]:
    return None if random_seed is None else int(random_seed)


CPORCredits = None

SDRCredits = Credits(
    'SDR',
    'Guy Shani',
    'shanigu@bgu.ac.il',
    'https://github.com/shanigu',
    '',
    'SDR is an online contingent replanner.\n'
    'It provides one action at a time, and then awaits to receive an observation from the environment.',
    'SDR operates by compiling a contingent problem into a classical problem, representing only some of the partial knowledge that the agent has.\n'
    'The classical problem is then solved. If an action is not applicable, due to the partial information, SDR modifies the classical problem and replans.\n'
    'Complete information can be found at the following paper: Replanning in Domains with Partial Information and Sensing Actions, Brafman and Shani, JAIR, 2012.'
)

MetaCPORCredits = Credits(
    'Contingent Planning Algorithms',
    'Guy Shani',
    'shanigu@bgu.ac.il',
    'https://github.com/shanigu',
    '',
    'Algorithms for offline and online decision making under partial observability and sensing actions',
    'This package provides a Python API to the algorithms developed by the group of Guy Shani and Ronen Brafman at the Ben Gurion university.\n'
    'Contingent planning under partial observation and sensing actions models domains where a single agent must make decisions, while some information is unknown, and sensing actions can provide useful information for deciding which actions to execute.\n'
    'The package contains CPOR, an offline planner that computes complete plan trees, and SDR, an online planner that interleaves planning and execution.'
)


class CPORImpl(Engine, OneshotPlannerMixin):

    def __init__(self, bOnline = False, random_seed: Optional[int] = None, use_cpor_loop: bool = True, **options):
        up.engines.Engine.__init__(self)
        up.engines.mixins.OneshotPlannerMixin.__init__(self)
        self.bOnline = bOnline
        self._skip_checks = False
        self.random_seed = _coerce_random_seed(random_seed)
        # Selects solve_native_cpor_loop() (the stack-based CPOR outer loop,
        # with the legacy exhaustive search as its own internal layered
        # fallback) vs. the legacy solve_native() path directly -- see
        # up_cpor.problem_grounder.run_my_grounder_and_solve's doc comment.
        self.use_cpor_loop = use_cpor_loop

    @property
    def name(self) -> str:
        return "CPORPlanning"

    @staticmethod
    def supports_compilation(compilation_kind: CompilationKind) -> bool:
        return compilation_kind == CompilationKind.GROUNDING

    @staticmethod
    def supported_kind():
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
        # Orchestration only.
        # Grounding SSoT: up_cpor.problem_grounder.extract_grounded_problem_data.
        # Native interop SSoT: up_cpor.native_api.
        # =====================================================================
        from up_cpor.problem_grounder import run_my_grounder_and_solve
        from unified_planning.plans import ContingentPlan

        root_node = run_my_grounder_and_solve(problem, use_cpor_loop=self.use_cpor_loop)

        if not root_node:
            return PlanGenerationResult(PlanGenerationResultStatus.UNSOLVABLE_PROVEN, None, self.name)

        return PlanGenerationResult(PlanGenerationResultStatus.SOLVED_SATISFICING, ContingentPlan(root_node), self.name)

    def destroy(self):
        pass


class SDRImpl(Engine, ActionSelectorMixin):
    """Native-C++-backed restoration of the README's "SDR Engine - with UP
    Simulated Environment" usage pattern (ActionSelector: get_action()/
    update(observation), one step at a time -- as opposed to CPORImpl's
    batch OneshotPlanner interface). Previously backed by the legacy C#
    CPORLib via pythonnet; removed entirely in commit eb4111c without the
    README or tests/up_test_utils.py being updated to match. This
    reimplementation routes instead through up_cpor.native_api's
    sdr_session_* entry points, which bridge to CPOR::SDRPlanner's own
    already-existing get_next_action()/apply_observation() incremental
    interface in cpp/SDRPlanner.hpp -- previously only ever used internally,
    statelessly, as SDRPlanner::compute_linear_plan's OnlinePlan subroutine
    inside the CPOR outer loop, never exposed to Python at all.
    """

    def __init__(self, problem: AbstractProblem, random_seed: Optional[int] = None, **options):
        up.engines.Engine.__init__(self)
        self._skip_checks = False
        self.random_seed = _coerce_random_seed(random_seed)
        ActionSelectorMixin.__init__(self, problem)

        assert isinstance(problem, ContingentProblem)

        from up_cpor.problem_grounder import UpCporConverter

        self._converter = UpCporConverter()
        self._converter.generate_native_problem(problem)
        native_api.sdr_session_init()

    @property
    def name(self) -> str:
        return "SDRPlanning"

    @staticmethod
    def supports_compilation(compilation_kind: CompilationKind) -> bool:
        return compilation_kind == CompilationKind.GROUNDING

    @staticmethod
    def supported_kind():
        # Same native grounding pipeline as CPORImpl (up_cpor.problem_grounder.
        # UpCporConverter.generate_native_problem), so it handles the same
        # constructs -- SDR's own compile-to-classical step happens natively,
        # inside CPORSolver::compute_heuristic/FFSolver, not in Python.
        return CPORImpl.supported_kind()

    @staticmethod
    def supports(problem_kind):
        return problem_kind <= SDRImpl.supported_kind()

    @staticmethod
    def get_credits(**kwargs) -> Optional["Credits"]:
        return SDRCredits

    def _get_action(self) -> "up.plans.ActionInstance":
        action_id = native_api.sdr_get_next_action()
        if action_id < 0:
            # -1 (goal reached) and -2 (failure/dead end) both mean "nothing
            # left to execute" from the caller's perspective -- ActionSelectorMixin's
            # contract doesn't distinguish them, callers are expected to check
            # is_goal_reached() on their own simulated environment instead.
            return None
        return self._converter.action_id_to_up_action[action_id]

    def _update(self, observation: Dict["up.model.FNode", "up.model.FNode"]):
        if not observation:
            # A just-executed classical (non-sensing) action produces an
            # empty observation dict -- nothing to feed back to the native
            # session, which only tracks state through its own effect
            # application during sdr_get_next_action() itself.
            return
        # A sensing action observes exactly one fluent (GroundedAction::observe_predicate_id
        # is a single predicate id, see cpp/ProblemData.hpp), so this dict
        # always has exactly one entry.
        observed_value = next(iter(observation.values()))
        native_api.sdr_apply_observation(observed_value.is_true())

    def destroy(self):
        native_api.sdr_session_destroy()


class CPORMetaEngineImpl(MetaEngine, mixins.OneshotPlannerMixin):
    """Native-C++-backed restoration of the README's "CPOR Meta Engine"
    usage pattern (MetaCPORPlanning[classical_planner], e.g.
    MetaCPORPlanning[tamer]). Previously backed by the legacy C# CPORLib via
    pythonnet; removed entirely in commit eb4111c without the README or
    tests/up_test_utils.py/tests/test_meta_cpor.py being updated to match.

    Honesty note: same as the pre-deletion implementation, the wrapped
    classical engine (self.engine, e.g. tamer or pyperplan) is validated for
    ProblemKind compatibility but is NOT actually invoked to drive the
    native solve -- our C++ core's classical sub-solver (FFSolver, wrapping
    FF-v2.3) is hardcoded, with no pluggable per-call backend. Wiring a real
    classical-planner swap into the C++ search would be a substantial
    architecture change, out of scope here. This class exists to restore
    MetaCPORPlanning[X] as a resolvable, correctly-solving OneshotPlanner
    name -- matching what tests/test_meta_cpor.py and the README actually
    exercise -- not to make the choice of X change native solving behavior.
    """

    def __init__(self, *args, random_seed: Optional[int] = None, **kwargs):
        self.random_seed = _coerce_random_seed(random_seed)
        kwargs.pop("random_seed", None)
        MetaEngine.__init__(self, *args, **kwargs)
        mixins.OneshotPlannerMixin.__init__(self)

    @property
    def name(self) -> str:
        return f"CPORPlanning[{self.engine.name}]"

    @staticmethod
    def is_compatible_engine(engine: Type[Engine]) -> bool:
        return engine.is_oneshot_planner() and engine.supports(ProblemKind({"ACTION_BASED"}))  # type: ignore

    @staticmethod
    def _supported_kind(engine: Type[Engine]) -> ProblemKind:
        return CPORImpl.supported_kind().union(engine.supported_kind())

    @staticmethod
    def _supports(problem_kind: ProblemKind, engine: Type[Engine]) -> bool:
        return problem_kind <= CPORMetaEngineImpl._supported_kind(engine)

    @staticmethod
    def get_credits(**kwargs) -> Optional["Credits"]:
        return MetaCPORCredits

    def _solve(self,
               problem: AbstractProblem,
               heuristic: Optional[Callable[["up.model.state.State"], Optional[float]]] = None,
               timeout: Optional[float] = None,
               output_stream: Optional[IO[str]] = None,
               ) -> 'PlanGenerationResult':

        assert isinstance(problem, ContingentProblem)
        assert isinstance(self.engine, mixins.OneshotPlannerMixin)

        if not self._supports(problem.kind, self.engine):
            return PlanGenerationResult(PlanGenerationResultStatus.UNSOLVABLE_PROVEN, None, self.name)

        from up_cpor.problem_grounder import run_my_grounder_and_solve
        from unified_planning.plans import ContingentPlan

        root_node = run_my_grounder_and_solve(problem, use_cpor_loop=True)

        if not root_node:
            return PlanGenerationResult(PlanGenerationResultStatus.UNSOLVABLE_PROVEN, None, self.name)

        return PlanGenerationResult(PlanGenerationResultStatus.SOLVED_SATISFICING, ContingentPlan(root_node), self.name)

    def destroy(self):
        pass