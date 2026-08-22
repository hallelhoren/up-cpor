# CPOR

CPOR is an offline contingent planner that computes a complete plan tree (or graph) where each node is labeled by an action, and edges are labeled by observations. The leaves of the plan tree correspond to goal states. CPOR uses the SDR translation to compute actions. When a sensing action is chosen, CPOR expands both child nodes corresponding to the possible observations. CPOR contains a mechanism for reusing plan segments (Plan Graph Compaction), resulting in a more compact graph.

More information about CPOR can be found in the paper: "Computing Contingent Plan Graphs using Online Planning" by Maliah, Komarnitski and Shani, published in TAAS in 2022.

# SDR

SDR is an online contingent replanner that provides one action at a time and then awaits to receive an observation from the environment. SDR operates by compiling a contingent problem into a classical problem, representing only some of the partial knowledge that the agent has. The classical problem is then solved. If an action is not applicable due to the partial information, SDR modifies the classical problem and replans.

More information about SDR can be found in the paper: "Replanning in Domains with Partial Information and Sensing Actions" by Brafman and Shani, published in JAIR in 2012.

# Installation

Clone the project and open a terminal in the project directory. Create a fresh environment:

```bash
conda create -n up_cpor_env python=3.11
conda activate up_cpor_env
```

Finally, use pip to install the required dependencies:

```bash
pip install -r requirements.txt
```

Build the native planning core and install the `up_cpor` package:

```bash
./build.sh
```

# Usage

CPOR and SDR can be used in different ways, depending on the specific requirements of your project. Here are the four supported usage patterns:

### CPOR Engine

```python
import unified_planning.environment as environment
from unified_planning.shortcuts import OneshotPlanner

env = environment.get_environment()
env.factory.add_engine('CPORPlanning', 'up_cpor.engine', 'CPORImpl')

with OneshotPlanner(name='CPORPlanning') as planner:
    result = planner.solve(problem)
```

### CPOR Meta Engine

```python
import unified_planning.environment as environment
from unified_planning.shortcuts import OneshotPlanner

env = environment.get_environment()
env.factory.add_meta_engine('MetaCPORPlanning', 'up_cpor.engine', 'CPORMetaEngineImpl')

with OneshotPlanner(name='MetaCPORPlanning[tamer]') as planner:
    result = planner.solve(problem)
```

### SDR Engine - with UP Simulated Environment

```python
import unified_planning.environment as environment
from unified_planning.model.contingent import SimulatedExecutionEnvironment
from unified_planning.shortcuts import ActionSelector

env = environment.get_environment()
env.factory.add_engine('SDRPlanning', 'up_cpor.engine', 'SDRImpl')

with ActionSelector(name='SDRPlanning', problem=problem) as solver:
    simulated_env = SimulatedExecutionEnvironment(problem)
    while not simulated_env.is_goal_reached():
        action = solver.get_action()
        observation = simulated_env.apply(action)
        solver.update(observation)
```

### SDR Engine - with SDR Simulated Environment

```python
import unified_planning.environment as environment
from unified_planning.shortcuts import ActionSelector

from up_cpor.simulator import SDRSimulator

env = environment.get_environment()
env.factory.add_engine('SDRPlanning', 'up_cpor.engine', 'SDRImpl')

with ActionSelector(name='SDRPlanning', problem=problem) as solver:
    simulated_env = SDRSimulator(problem)
    while not simulated_env.is_goal_reached():
        action = solver.get_action()
        observation = simulated_env.apply(action)
        solver.update(observation)
```

# Notebooks

The `notebooks` directory contains examples that demonstrate how to use CPOR and SDR on sample contingent-planning domains.
