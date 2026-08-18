"""Command-line entry point matching the original C# ``RunPlanner`` contract
(``CPORLib/Run.cs``): ``RunPlanner domain_file problem_file output_file
[online/offline]``.

Offline mode (default) grounds and solves the problem with the native
CPOR engine and writes the resulting plan tree as a GraphViz DOT file in
the same ``digraph contingent_plan { ... }`` shape ``CPORPlanner.WritePlan``
produces in the legacy engine (see ``CPORLib/Algorithms/CPORPlanner.cs``),
so existing tooling built around that format (including this repo's own
``tests/cpor_test_utils.py:parse_dot``/``assert_dot_equal``) can consume
output from either engine interchangeably.

Online mode drives the SDR engine step by step against a simulated
environment, mirroring ``Run.cs``'s ``bOnline`` branch (10 episodes,
logging each executed action and its observation) -- note the original
never writes ``output_file`` in this mode either; that's preserved here
for parity, not fixed.
"""
import sys

USAGE = "Usage: RunPlanner domain_file problem_file output_file [online/offline]"


def _action_label(action_instance) -> str:
    """Formats an ActionInstance as ``name~param1~param2~...``, matching
    CPORLib's ``Action.Name`` serialization (space/`` ``-separated
    parameters compiled to ``~`` throughout this codebase -- see e.g.
    up_cpor.converter.__convert_CPOR_string_to_action_instance)."""
    parts = [action_instance.action.name]
    parts.extend(str(p) for p in action_instance.actual_parameters)
    return "~".join(parts)


def write_plan_dot(root_node, output_file: str) -> None:
    """Serializes a unified_planning ContingentPlanNode tree to the same
    DOT structure as CPORPlanner.WritePlan: an invisible ``_nil`` root
    pointing at node 0, one node per action, boxed true/false observation
    nodes for every sensing branch, and an edge list. Node ids are a
    fresh local sequential counter (the legacy engine's ``node.ID`` in the
    label prefix is an internal search-tree id with no equivalent here;
    tests/cpor_test_utils.py's own comparison already strips it via
    ``_ID_PREFIX_RE`` and compares structure/action-name only)."""
    node_ids = {}
    node_labels = {}
    observation_nodes = {}
    edges = []
    next_id = [0]

    def alloc(node):
        node_ids[id(node)] = next_id[0]
        next_id[0] += 1
        return node_ids[id(node)]

    def visit(node):
        if node is None:
            return
        nid = node_ids.get(id(node))
        if nid is None:
            nid = alloc(node)
            action_instance = node.action_instance
            label = f"{nid}){_action_label(action_instance)}"
            node_labels[nid] = label

        if not node.children:
            return

        is_sensing = len(node.children) > 1
        if is_sensing:
            children_by_truth = {}
            for observation, child in node.children:
                truth = next(iter(observation.values())).is_true()
                children_by_truth[truth] = child

            for truth in (False, True):
                child = children_by_truth.get(truth)
                if child is None:
                    continue
                obs_id = next_id[0]
                next_id[0] += 1
                observation_nodes[obs_id] = truth
                edges.append((nid, obs_id))
                child_id = node_ids.get(id(child))
                if child_id is None:
                    child_id = alloc(child)
                    ai = child.action_instance if child.action_instance else None
                    if ai is not None:
                        node_labels[child_id] = f"{child_id}){_action_label(ai)}"
                    else:
                        node_labels[child_id] = f"{child_id}) Goal"
                    edges.append((obs_id, child_id))
                    visit(child)
                else:
                    edges.append((obs_id, child_id))
        else:
            _, child = node.children[0]
            child_id = node_ids.get(id(child))
            if child_id is None:
                child_id = alloc(child)
                ai = child.action_instance if child.action_instance else None
                if ai is not None:
                    node_labels[child_id] = f"{child_id}){_action_label(ai)}"
                else:
                    node_labels[child_id] = f"{child_id}) Goal"
                edges.append((nid, child_id))
                visit(child)
            else:
                edges.append((nid, child_id))

    if root_node is None:
        node_labels[0] = "0) Goal"
    else:
        visit(root_node)

    with open(output_file, "w") as f:
        f.write("digraph contingent_plan {\n")
        f.write('\t_nil [style="invis"];\n')
        for nid in sorted(node_labels):
            f.write(f'\t{nid} [label="{node_labels[nid]}"];\n')
        for obs_id, truth in sorted(observation_nodes.items()):
            f.write(f'\t{obs_id} [label="{str(truth).lower()}" ,shape="box"];\n')
        for src, dst in edges:
            f.write(f"\t{src} -> {dst};\n")
        f.write('\t_nil -> 0 [label=""];\n')
        f.write("}\n")


def _run_offline(domain_file: str, problem_file: str, output_file: str) -> int:
    import unified_planning.environment as up_environment
    from unified_planning.io import PDDLReader
    from unified_planning.engines.results import PlanGenerationResultStatus

    env = up_environment.Environment()
    env.factory.add_engine("CPORPlanning", "up_cpor.engine", "CPORImpl")
    reader = PDDLReader(env)
    problem = reader.parse_problem(domain_file, problem_file)

    with env.factory.OneshotPlanner(name="CPORPlanning") as planner:
        result = planner.solve(problem)

    if result.status != PlanGenerationResultStatus.SOLVED_SATISFICING:
        print(f"No plan found (status: {result.status}).")
        write_plan_dot(None, output_file)
        return 1

    write_plan_dot(result.plan.root_node, output_file)
    print(f"Plan written to {output_file}")
    return 0


def _run_online(domain_file: str, problem_file: str, output_file: str) -> int:
    import unified_planning.environment as up_environment
    from unified_planning.io import PDDLReader
    from unified_planning.model.contingent import SimulatedExecutionEnvironment

    env = up_environment.Environment()
    env.factory.add_engine("SDRPlanning", "up_cpor.engine", "SDRImpl")
    reader = PDDLReader(env)
    problem = reader.parse_problem(domain_file, problem_file)

    CITERATIONS = 10
    successes = 0
    idx = 0
    for _ in range(CITERATIONS):
        sim = SimulatedExecutionEnvironment(problem)
        with env.factory.ActionSelector(name="SDRPlanning", problem=problem) as solver:
            print(f"Starting {problem.name}")
            while not sim.is_goal_reached():
                action = solver.get_action()
                if action is None:
                    print("*", end="")
                    break
                observation = sim.apply(action)
                solver.update(observation)
                print(f"{idx}) Executed {action}, received {observation}")
                idx += 1
        if sim.is_goal_reached():
            successes += 1
    print(f"{successes}/{CITERATIONS} episodes reached the goal.")
    return 0 if successes == CITERATIONS else 1


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 3:
        print(USAGE)
        return 0

    domain_file, problem_file, output_file = argv[0], argv[1], argv[2]
    online = len(argv) > 3 and argv[3] == "online"

    if online:
        return _run_online(domain_file, problem_file, output_file)
    return _run_offline(domain_file, problem_file, output_file)


if __name__ == "__main__":
    sys.exit(main())
