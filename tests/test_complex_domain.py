from itertools import chain, combinations, islice

import pytest
from unified_planning.engines.results import PlanGenerationResultStatus
from unified_planning.engines.sequential_simulator import UPSequentialSimulator
from unified_planning.model.contingent import SensingAction

from cpor_test_utils import TEST_RANDOM_SEED
from up_test_utils import (
    CachingSequentialSimulator,
    make_contingent_problem_from_possible_initial_states,
    make_test_environment,
    use_test_environment,
)
from viplan_hh.viplan_hh_cases import (
    VIPLAN_HH_CASES,
    parse_problem,
    state_specs_to_upstates,
)

CPOR_PLANNER_PARAMS = {"random_seed": TEST_RANDOM_SEED}


def _powerset(iterable):
    """Return all subsets of *iterable* without materializing the powerset."""

    items = list(iterable)
    return chain.from_iterable(combinations(items, r) for r in range(len(items) + 1))


def _iter_subset_specs_for_case(case):
    possible_state_specs = case.possible_state_specs()
    probe = list(islice(_powerset(possible_state_specs), 401))
    if len(probe) <= 400:
        yield from probe
        return

    yield from probe[:200]

    tail = []
    for subset_size in range(len(possible_state_specs), -1, -1):
        for subset in combinations(possible_state_specs, subset_size):
            tail.append(subset)
            if len(tail) >= 200:
                break
        if len(tail) >= 200:
            break
    tail.reverse()
    yield from tail


def _iter_case_subset_params():
    for case in VIPLAN_HH_CASES.values():
        for subset_index, subset in enumerate(_iter_subset_specs_for_case(case)):
            yield pytest.param(
                case,
                subset_index,
                subset,
                id=f"{case.name}[subset_{subset_index}_size_{len(subset)}]",
            )


def _solve_case_with_state_specs(case, state_specs):
    env = make_test_environment(cpor=True)
    with use_test_environment(env):
        problem = parse_problem(case)
        possible_states = state_specs_to_upstates(problem, state_specs)
        contingent_problem = make_contingent_problem_from_possible_initial_states(
            problem, possible_states
        )
        with env.factory.OneshotPlanner(
            name="CPORPlanning", params=CPOR_PLANNER_PARAMS
        ) as planner:
            result = planner.solve(contingent_problem)
    return problem, possible_states, contingent_problem, result


def _observation_signature(observation):
    return tuple(sorted((str(fluent), str(value)) for fluent, value in observation.items()))


def _ground_observation(action_instance, state):
    action = action_instance.action
    assert isinstance(action, SensingAction)

    observation = {}
    substitutions = dict(zip(action.parameters, action_instance.actual_parameters))
    for observed_fluent in action.observed_fluents:
        grounded_fluent = observed_fluent.substitute(substitutions)
        observation[grounded_fluent] = state.get_value(grounded_fluent)
    return observation


def _execute_contingent_plan_from_state(problem, initial_state, plan, *, label, simulator=None):
    if simulator is None:
        simulator = UPSequentialSimulator(problem)
    current_state = initial_state
    node = plan.root_node

    if node is None:
        assert simulator.is_goal(current_state), f"Empty plan does not reach the goal for {label}."
        return

    step = 0
    while node is not None:
        step += 1
        next_state = simulator.apply(current_state, node.action_instance)
        assert next_state is not None, (
            f"Action {node.action_instance} is not applicable at step {step} for {label}."
        )
        current_state = next_state

        if not node.children:
            node = None
            continue

        if isinstance(node.action_instance.action, SensingAction):
            observation = _ground_observation(node.action_instance, current_state)
        else:
            observation = {}

        observation_signature = _observation_signature(observation)
        matching_children = [
            child
            for child_observation, child in node.children
            if _observation_signature(child_observation) == observation_signature
        ]
        available_observations = [
            _observation_signature(child_observation) for child_observation, _ in node.children
        ]
        assert len(matching_children) == 1, (
            f"Expected exactly one matching branch for observation {observation_signature} "
            f"at step {step} for {label}; available branches: {available_observations}."
        )
        node = matching_children[0]

    assert simulator.is_goal(current_state), f"Plan does not reach the goal for {label}."


def _assert_plan_reaches_goal_from_checked_states(problem, possible_states, plan, *, label):
    simulator = CachingSequentialSimulator(problem)
    initial_state = simulator.get_initial_state()
    _execute_contingent_plan_from_state(
        problem, initial_state, plan, label=f"{label}:pddl_initial", simulator=simulator
    )
    for state_index, state in enumerate(possible_states):
        _execute_contingent_plan_from_state(
            problem, state, plan,
            label=f"{label}:possible_state_{state_index}",
            simulator=simulator,
        )


@pytest.mark.parametrize("case", tuple(VIPLAN_HH_CASES.values()), ids=lambda case: case.name)
def test_viplan_hh_cpor_single_state_plan_reaches_goal(case):
    problem, possible_states, _contingent_problem, result = _solve_case_with_state_specs(
        case, case.representative_states[:1]
    )

    assert result.status == PlanGenerationResultStatus.SOLVED_SATISFICING, (
        f"CPORPlanning failed to find a plan for {case.name} with a single representative state: "
        f"{result.status}"
    )
    assert result.plan is not None, f"CPORPlanning returned no plan for {case.name}."

    _assert_plan_reaches_goal_from_checked_states(
        problem,
        possible_states,
        result.plan,
        label=case.name,
    )


@pytest.mark.parametrize("case", tuple(VIPLAN_HH_CASES.values()), ids=lambda case: case.name)
def test_viplan_hh_cpor_representative_uncertainty_plan_reaches_goal(case):
    problem, possible_states, _contingent_problem, result = _solve_case_with_state_specs(
        case, case.representative_states
    )

    assert result.status == PlanGenerationResultStatus.SOLVED_SATISFICING, (
        f"CPORPlanning failed to find a plan for {case.name} with representative uncertainty: "
        f"{result.status}"
    )
    assert result.plan is not None, f"CPORPlanning returned no plan for {case.name}."

    _assert_plan_reaches_goal_from_checked_states(
        problem,
        possible_states,
        result.plan,
        label=case.name,
    )


@pytest.mark.parametrize(
    ("case", "subset_index", "subset_state_specs"),
    tuple(_iter_case_subset_params()),
)
def test_viplan_hh_cpor_subset_plan_reaches_goal(case, subset_index, subset_state_specs):
    state_specs = tuple(subset_state_specs) + (case.true_state,)
    problem, possible_states, _contingent_problem, result = _solve_case_with_state_specs(
        case, state_specs
    )

    assert result.status == PlanGenerationResultStatus.SOLVED_SATISFICING, (
        f"CPORPlanning failed for case={case.name}, subset #{subset_index}: {result.status}"
    )
    assert result.plan is not None, (
        f"CPORPlanning returned no plan for case={case.name}, subset #{subset_index}."
    )

    _assert_plan_reaches_goal_from_checked_states(
        problem,
        possible_states,
        result.plan,
        label=f"{case.name}:subset_{subset_index}",
    )
