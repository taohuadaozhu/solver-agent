# Python MultiAlg Port

This package keeps the original project shape:

- `Population` is the base class in `population.py`.
- `MultiMet(Population[float])` is the multimethod solver class in `multimethod.py`.
- `main.py` mirrors the active C++ `main.cpp` path: Sphere + CSA, `NVAR=1000`, `POPSIZE=20`, bounds `[-5, 10]`, `THRESHOLD=1e-6`, default `MAXGEN=100`.
- `MultiMet` exposes all method names from the C++ `MultiMet` class, including GA/PSO/APSO/HS/ACO/COA/DE/CSA/ABCA/POA/ILS/VNS/GRASP/PBILC/BATA/FA/CMA-ES/BA/local-search/memetic operators.
- `PROGRAM_GUIDE.md` contains the detailed Python class, method, algorithm, subfunction, archive, surrogate, and benchmark interface guide.

Core class layout:

- `population.py`: `Population[T]` owns `pop`, `newpop`, `pop_fit`, `newpop_fit`, `gbest`, `gbest_fit`, bounds, sorting helpers, random helpers, `worst_and_best()`, `Elist()`, and convergence statistics.
- `multimethod.py`: `MultiMet` inherits `Population[float]` plus all algorithm mixins. It owns `EvaluFunc`, `ibest`, `velocity`, PSO adaptive state, ACO pheromone memory, ABCA trial state, ADE/SHADE state, meme Q/reward state, and shared lifecycle methods.
- `algorithms/*.py`: each file defines one mixin family, for example `GAMixin`, `PSOMixin`, `DEMixin`, `CMAESMixin`, `LocalSearchMixin`.
- `algorithms/archive.py`, `popvary.py`, `surrogate.py`: reusable archive, population-variation, PCA/RBF/Kriging/quadratic-model helpers.

Standard solver loop:

```python
from python_multialg.multimethod import MultiMet
from python_multialg.problems import Sphere

solver = MultiMet(20, 100, -5.0, 5.0, Sphere)
solver.Initial()

for gen in range(100):
    solver.DE(0.8, 1, 0.5, 0, solver.Popsize)
    solver.Evaluation(True, 0, solver.Popsize)
    solver.pop_update(0, solver.Popsize)
    solver.worst_and_best()
    solver.Elist()

print(solver.gbest_fit)
```

Algorithm method groups:

- GA: `GA`, `NGA`, `LGA`, `VAGA`, `EAGA`; helpers include `select`, `crossover`, `xover`, `mutate`.
- PSO: `PSO`, `CPSO`, `SPSO`, `CMPSO`, `APSO_1` to `APSO_5`, `OLPSO`, `GPSO`, `AESSPSO`; helpers include `Subgradient`, `SparseSubgradient`, `Orthogonal_P`, `LightOrthogonal_P`.
- DE: `DE`, `DE_ARCHIVE_SURROGATE`, `ADE`, `SHADE`, `IMODE`, `SA`, `SADE_AMSS`, `SADE_ATDSC`, `SADE_Sammon`; helpers include `differential_mutate`, `differential_crossover`, archive and surrogate functions.
- Swarm/metaheuristics: `HS`, `ACO`, `COA`, `CSA`, `ABCA`, `POA`, `ILS`, `VNS`, `GRASP`, `PBILC`, `BATA`, `FA`, `BA`.
- Gradient/local optimizers: `Adam`, `BFGS`, `FRCG`, `RMSProp`, `SD`, `SQP`, `Nelder_Mead`.
- PlatEMO-inspired additions: `AutoV`, `CSO`, `DOA`, `ECPO`, `EGO`, `FEP`, `FROFI`, `GWO`, `KMA`, `L2SMEA`, `MFEA`, `MFEA_II`, `MGO`, `MVPA`, `MiSACO`, `OFA`, `SACC_EAM_II`, `SACOSO`, `SAMSO`, `SAPO`, `SSIO_RL`, `WOA`.
- Local search and meme: `newpop_bit_climbing`, `newpop_simplex`, `newpop_pattern_search`, `newpop_gradient_backtracking`, `meme_q_learning`, `meme_biasd_roulette`, and related methods in `LocalSearchMixin`.

Most algorithm calls only write candidate solutions to `solver.newpop`; the caller should then run `Evaluation(True, ...)` and update the population. For conservative local-search or surrogate-filtered algorithms, use `pop_better_update()` when only improving candidates should be accepted.

Run:

```sh
PYTHONPYCACHEPREFIX=/private/tmp/pycache python3 -m python_multialg.main --seed 1
```

Benchmark repeated runs:

```sh
PYTHONPYCACHEPREFIX=/private/tmp/pycache python3 -m python_multialg.benchmark --runs 3 --workers 0
```

Optional NumPy fast path:

```sh
PYTHONPYCACHEPREFIX=/private/tmp/pycache python3 -c "from python_multialg.fast import run_csa_fast; print(run_csa_fast(seed=1))"
```

ABCA plus meme local-search selection:

```sh
PYTHONPYCACHEPREFIX=/private/tmp/pycache python3 -m python_multialg.abca_meme --problems sphere rosenbrock griewank ackley --nvar 1000 --popsize 20 --generations 100 --meme-interval 20 --repeats 3 --workers 0
```

Island-model ABCA plus meme coevolution:

```sh
PYTHONPYCACHEPREFIX=/private/tmp/pycache python3 -m python_multialg.island_meme --problem ackley --island-memes randperm inheritance simple_random biasd_roulette --nvar 1000 --popsize 20 --generations 100 --islands 4 --migration-interval 20 --meme-interval 20 --migrants 1 --workers 0
```

Speed-oriented island model with elite-only meme local search:

```sh
PYTHONPYCACHEPREFIX=/private/tmp/pycache python3 -m python_multialg.island_meme --problem ackley --island-memes randperm inheritance simple_random biasd_roulette --nvar 1000 --popsize 20 --generations 100 --islands 4 --migration-interval 20 --meme-interval 20 --meme-elites 8 --migrants 1 --workers 4
```

Fair serial-vs-island comparison with the same total population size:

```sh
PYTHONPYCACHEPREFIX=/private/tmp/pycache python3 -m python_multialg.compare_serial_parallel --nvar 1000 --serial-popsize 20 --island-popsize 5 --generations 100 --meme-interval 10 --meme-elites 5 --islands 4 --migration-interval 1 --topology pool --island-memes randperm randperm randperm randperm --workers 1
```

Millisecond-level NumPy pool-topology search:

```sh
PYTHONPYCACHEPREFIX=/private/tmp/pycache /Users/lailiyuanjun/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m python_multialg.fast_topology --nvar 1000 --total-pop 100 --islands 4 --generations 100 --migration-interval 1 --local-interval 10 --local-elites 10 --local-dims 8
```

`fast_topology.py` uses benchmark-specific structural candidates and local improvements by default. To run only the generic vectorized pool topology, disable both:

```sh
PYTHONPYCACHEPREFIX=/private/tmp/pycache /Users/lailiyuanjun/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m python_multialg.fast_topology --nvar 1000 --total-pop 100 --islands 4 --generations 100 --migration-interval 1 --local-interval 10 --local-elites 10 --local-dims 8 --no-structural --no-specialized-local --dtype float32
```

Generic quality-oriented batch mode without structural candidates:

```sh
PYTHONPYCACHEPREFIX=/private/tmp/pycache /Users/lailiyuanjun/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m python_multialg.fast_topology --nvar 1000 --total-pop 100 --islands 4 --generations 100 --migration-interval 1 --local-interval 10 --local-elites 10 --local-dims 8 --no-structural --no-specialized-local --offspring 2 --dtype float32 --eval-threads 4
```

Fast internal subspace mode for lower latency:

```sh
PYTHONPYCACHEPREFIX=/private/tmp/pycache /Users/lailiyuanjun/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m python_multialg.fast_topology --nvar 1000 --total-pop 100 --islands 4 --generations 100 --migration-interval 1 --local-interval 10 --local-elites 10 --local-dims 8 --no-structural --no-specialized-local --dtype float32 --active-dims 128 --full-refresh-interval 10
```

`--offspring K` generates K candidates for every individual in one NumPy batch and keeps the best candidate per individual. `--eval-threads K` splits large candidate evaluation batches inside a single problem. `--active-dims K` updates only K random dimensions each generation while still accepting moves with the full objective; `--full-refresh-interval K` periodically runs a full-dimensional generation to reduce subspace drift. `--epoch-size K` batches several generations into one tensor update, but it delays feedback; in current 1000-variable tests it is useful for experimentation, not the default recommendation.

The pure-Python port keeps the C++ method surface and algorithm flow. Some of the heavy local-search and CMA-ES internals are written in a Pythonic form rather than as line-by-line C++ translations; `fast.py` contains a NumPy-specialized CSA path for performance-sensitive Sphere runs. With `NVAR=1000`, the default CSA `MAXGEN` is capped at `100`; pass `--maxgen` when a different budget is needed.

Algorithm layout:

- `population.py`: base population state and common random/math helpers.
- `multimethod.py`: `MultiMet` state initialization, evaluation, shared update helpers, and CLI runner.
- `algorithms/ga.py`, `pso.py`, `de.py`, `csa.py`, `hs.py`, `aco.py`, `coa.py`, `abca.py`, `poa.py`, `ils.py`, `vns.py`, `grasp.py`, `pbilc.py`, `bata.py`, `fa.py`, `cmaes.py`, `ba.py`: one algorithm family per module, exposed as mixins on `MultiMet`.
- `algorithms/local_search.py`: local search and meme-selection operators used by the algorithm families.
