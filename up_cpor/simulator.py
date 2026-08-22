"""Native Python/C++ restoration of the README's "SDR Engine - with SDR
Simulated Environment" usage pattern. Previously backed entirely by the
legacy C# CPORLib.PlanningModel.Simulator via pythonnet (up_cpor.converter,
now removed); this reimplementation routes instead through
up_cpor.native_api's sim_session_* entry points, which sample one fully
concrete, oneof-consistent ground-truth world (via CPOR::SDRSampler, the
same Z3-backed sampling CPOR's own OnlinePlan witness generation already
trusts) and apply actions to it via CPOR::ActionApplier (the same
deterministic effect-application logic CPORSolver/SDRPlanner already trust)
-- see native_bridge.cpp's "NATIVE EXECUTION-ENVIRONMENT SIMULATOR" section
for the full native-side design rationale.

Distinct from up_cpor.engine.SDRImpl's own native SDR *session*
(sdr_session_init/get_next_action/apply_observation): that one is the
*planner's* incremental belief tracking (what SDRImpl thinks is true);
this one is the *environment's* single ground truth (what's actually
true) -- the same split unified_planning's own SimulatedExecutionEnvironment
makes between the planner under test and the environment driving it.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import unified_planning as up
from unified_planning.exceptions import UPUsageError
from unified_planning.model.contingent import ExecutionEnvironment
from unified_planning.model.contingent.sensing_action import SensingAction
from unified_planning.model.walkers import Substituter

from up_cpor import native_api


class SDRSimulator(ExecutionEnvironment):
    """A native execution environment: samples one concrete, fully-resolved
    world consistent with `problem`'s initial belief and oneof constraints,
    then plays "the environment" for that fixed world -- apply() rejects any
    action whose precondition doesn't actually hold there (raising the same
    UPUsageError unified_planning's own SimulatedExecutionEnvironment raises)
    and reports back what a sensing action observes; is_goal_reached()
    checks the goal formula against it. Drop-in replacement for
    unified_planning.model.contingent.SimulatedExecutionEnvironment wherever
    this project's own SDR-simulator semantics are wanted specifically.
    """

    def __init__(
        self,
        problem: "up.model.contingent.contingent_problem.ContingentProblem",
        random_seed: Optional[int] = None,
    ):
        super().__init__(problem)
        self.problem = problem
        self.random_seed = random_seed
        # NOTE on random_seed: accepted for interface parity with the legacy
        # C# simulator's constructor, but currently a documented no-op --
        # CPOR::SDRSampler's underlying Z3 sampling has no exposed seeding
        # hook yet, so which of the (possibly many) logically consistent
        # concrete worlds gets sampled isn't currently reproducible via this
        # parameter. Not silently pretending otherwise.

        from up_cpor.problem_grounder import UpCporConverter

        # Regrounds `problem` into the native C++ core, exactly like every
        # other engine in up_cpor.engine (CPORImpl/SDRImpl) unconditionally
        # does on construction -- the global C++ problem state can't be
        # trusted to already reflect the right problem otherwise. When used
        # per this module's own docstring pattern (an SDRImpl-backed
        # ActionSelector constructed first, this simulator constructed
        # second, both for the same `problem`), this reground reproduces an
        # identical grounding (deterministic given the same problem), so it
        # doesn't disturb the already-running SDR session's own view of the
        # global problem in practice.
        self._converter = UpCporConverter()
        self._converter.generate_native_problem(problem)
        self._substituter = Substituter(problem.environment)
        self._em = problem.environment.expression_manager

        # Reverse of self._converter.action_id_to_up_action, keyed by value
        # (action name + stringified actual parameters) rather than object
        # identity/equality -- apply() may be called with an ActionInstance
        # built by a *different* UpCporConverter instance (e.g. SDRImpl's
        # own), so matching by value is what makes this robust regardless of
        # which converter produced the instance.
        self._action_key_to_id: Dict[Tuple[str, Tuple[str, ...]], int] = {
            self._action_key(instance): action_id
            for action_id, instance in self._converter.action_id_to_up_action.items()
        }

        if not native_api.sim_session_init():
            raise UPUsageError(
                "SDRSimulator: the problem's initial belief has no logically "
                "consistent concrete resolution (contradictory oneof/unknown "
                "initial constraints) -- nothing to simulate."
            )

    @staticmethod
    def _action_key(action: "up.plans.ActionInstance") -> Tuple[str, Tuple[str, ...]]:
        return (
            action.action.name,
            tuple(str(p) for p in action.actual_parameters),
        )

    def apply(
        self, action: "up.plans.ActionInstance"
    ) -> Dict["up.model.FNode", "up.model.FNode"]:
        action_id = self._action_key_to_id.get(self._action_key(action))
        if action_id is None:
            raise UPUsageError(
                f"SDRSimulator: unrecognized action instance {action} -- not "
                "part of the grounded problem this simulator was built from."
            )

        if not native_api.sim_apply_action(action_id):
            raise UPUsageError("The given action is not applicable!")

        observation: Dict["up.model.FNode", "up.model.FNode"] = {}
        if isinstance(action.action, SensingAction) and action.action.observed_fluents:
            subs = dict(zip(action.action.parameters, action.actual_parameters))
            for observed in action.action.observed_fluents:
                grounded = self._substituter.substitute(observed, subs)
                fact_id = self._converter.fluent_to_id.get(str(grounded))
                if fact_id is None:
                    continue
                value = native_api.sim_get_fact_value(fact_id)
                observation[grounded] = self._em.TRUE() if value == 1 else self._em.FALSE()
        return observation

    def is_goal_reached(self) -> bool:
        return native_api.sim_is_goal_reached()

    def destroy(self) -> None:
        native_api.sim_session_destroy()
