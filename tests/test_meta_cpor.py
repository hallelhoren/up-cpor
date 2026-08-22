import pytest
from unified_planning.engines.results import PlanGenerationResultStatus

from cpor_test_utils import TEST_RANDOM_SEED
from domains import DOMAINS
from up_test_utils import make_test_environment, parse_test_problem

CLASSICAL_PLANNERS = ("tamer", "pyperplan")
META_CPOR_PLANNER_PARAMS = {"random_seed": TEST_RANDOM_SEED}


@pytest.mark.parametrize("classical_planner", CLASSICAL_PLANNERS)
@pytest.mark.parametrize("domain", DOMAINS)
def test_meta_cpor_plan_found(domain: str, classical_planner: str):
    env = make_test_environment(meta_cpor=True)
    problem = parse_test_problem(domain, env)

    with env.factory.OneshotPlanner(
        name=f"MetaCPORPlanning[{classical_planner}]",
        params=META_CPOR_PLANNER_PARAMS,
    ) as planner:
        result = planner.solve(problem)

    assert result.status == PlanGenerationResultStatus.SOLVED_SATISFICING, (
        f"MetaCPOR[{classical_planner}] failed to find a plan for {domain}: {result.status}"
    )
