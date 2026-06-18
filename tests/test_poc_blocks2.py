import os
from unified_planning.io import PDDLReader
import unified_planning.environment as environment
from unified_planning.shortcuts import OneshotPlanner
from unified_planning.engines.results import PlanGenerationResultStatus

def run_poc():
    print("Starting POC pipeline for blocks2...")
    reader = PDDLReader()
    
    # נתיבים לקבצי PDDL של blocks2 (הנחה שהם בתיקיית Tests/blocks2)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    domain_path = os.path.join(base_dir, "blocks2", "d.pddl")
    problem_path = os.path.join(base_dir, "blocks2", "p.pddl")
    
    if not os.path.exists(domain_path) or not os.path.exists(problem_path):
        print(f"Error: Could not find PDDL files at {domain_path} or {problem_path}")
        return

    problem = reader.parse_problem(domain_path, problem_path)
    print(f"Parsed problem: {problem.name}")

    env = environment.get_environment()
    # רישום המנוע הקיים (שעכשיו ידע לזהות שזה POC)
    env.factory.add_engine('CPORPlanning', 'up_cpor.engine', 'CPORImpl')

    with OneshotPlanner(name='CPORPlanning') as planner:
        result = planner.solve(problem)
        if result.status == PlanGenerationResultStatus.SOLVED_SATISFICING:
            print("\\n====================================")
            print("POC Success! Plan found:")
            print("====================================")
            if result.plan:
                for idx, action in enumerate(result.plan.actions):
                    print(f"{idx + 1}. {action}")
        else:
            print("\\nPOC Failed to find a solution.")

if __name__ == '__main__':
    run_poc()