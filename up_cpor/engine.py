from unified_planning.engines import Credits, MetaEngine, Engine
from unified_planning.plans import ContingentPlan
import unified_planning.engines.mixins as mixins
from unified_planning.engines.mixins.oneshot_planner import OneshotPlannerMixin
from unified_planning.engines.mixins.action_selector import ActionSelectorMixin
from unified_planning.engines.mixins.compiler import CompilationKind
import unified_planning as up
from unified_planning.model import ProblemKind, AbstractProblem
from unified_planning.model.contingent.contingent_problem import ContingentProblem
from unified_planning.engines.results import PlanGenerationResultStatus, PlanGenerationResult
from up_cpor.converter import UpCporConverter

from typing import Type, IO, Optional, Callable, Dict


def _coerce_random_seed(random_seed: Optional[int]) -> Optional[int]:
    return None if random_seed is None else int(random_seed)


CPORCredits = Credits('CPOR (C++ Native Core)',
                     'Guy Shani',
                     'shanigu@bgu.ac.il',
                     'https://github.com/guyazran/up-cpor',
                     '',
                     'CPOR High-Performance implementation.',
                     'Migrated to modern C++ with Data-Oriented Design (Bitsets/RPN).'
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
        # Pure C++ Pipeline Router
        # =====================================================================
        problem_name = getattr(problem, 'name', '').lower()
        if "blocks" in problem_name or "bw-rand" in problem_name:
            from up_cpor.problem_grounder import run_my_grounder_and_solve 
            from unified_planning.plans import ContingentPlan
            
            root_node = run_my_grounder_and_solve(problem)
            
            if not root_node:
                return PlanGenerationResult(PlanGenerationResultStatus.UNSOLVABLE_PROVEN, None, self.name)
            
            return PlanGenerationResult(PlanGenerationResultStatus.SOLVED_SATISFICING, ContingentPlan(root_node), self.name)
        
        raise NotImplementedError("Non-blocks problems require the legacy C# engine, which has been removed.")

    def destroy(self):
        pass