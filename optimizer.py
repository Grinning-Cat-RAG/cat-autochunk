import asyncio
import inspect
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Tuple, Type

from cat.log import log

from .evaluation import WORST_FITNESS, EvaluationResult
from .search_space import Candidate, SearchSpace


def resolve_algorithm(name: str) -> Tuple[Type, Type]:
    """Get the optimizer class of pyVolutionary and its configuration class, by name of the optimizer.

    Raises:
        ValueError: if pyVolutionary has no such optimizer.
    """
    import pyvolutionary
    from pyvolutionary.abstract import OptimizationAbstract
    from pyvolutionary.models import BaseOptimizationConfig

    optimizer_class = getattr(pyvolutionary, name, None)
    config_class = getattr(pyvolutionary, f"{name}Config", None)
    if (
        not inspect.isclass(optimizer_class)
        or not issubclass(optimizer_class, OptimizationAbstract)
        or not inspect.isclass(config_class)
        or not issubclass(config_class, BaseOptimizationConfig)
    ):
        raise ValueError(
            f"Unknown pyVolutionary algorithm '{name}'. Available algorithms: {', '.join(available_algorithms())}"
        )
    return optimizer_class, config_class


def available_algorithms() -> List[str]:
    import pyvolutionary
    from pyvolutionary.abstract import OptimizationAbstract

    return sorted(
        name for name, obj in inspect.getmembers(pyvolutionary, inspect.isclass)
        if issubclass(obj, OptimizationAbstract)
        and obj is not OptimizationAbstract
        and inspect.isclass(getattr(pyvolutionary, f"{name}Config", None))
    )


@dataclass
class EvaluatedCandidate:
    candidate: Candidate
    result: EvaluationResult


@dataclass
class OptimizationOutcome:
    algorithm: str
    evaluations: List[EvaluatedCandidate] = field(default_factory=list)
    budget_exhausted: bool = False
    error: str | None = None

    @property
    def best(self) -> EvaluatedCandidate | None:
        valid = [e for e in self.evaluations if e.result.error is None]
        return max(valid, key=lambda e: e.result.fitness) if valid else None


EvaluateFunction = Callable[[Candidate], Awaitable[EvaluationResult]]


class _EvaluationBridge:
    """Synchronous objective for pyVolutionary (running in a worker thread), evaluating the candidates as coroutines on
    the event loop of the Cat. Each distinct configuration is evaluated once; the budget (number of evaluations and
    time) is enforced here."""

    def __init__(
        self,
        space: SearchSpace,
        evaluate: EvaluateFunction,
        loop: asyncio.AbstractEventLoop,
        outcome: OptimizationOutcome,
        max_evaluations: int,
        deadline: float,
    ):
        self.space = space
        self.evaluate = evaluate
        self.loop = loop
        self.outcome = outcome
        self.max_evaluations = max_evaluations
        self.deadline = deadline
        self.cache: Dict[str, float] = {}

    def register(self, candidate: Candidate, result: EvaluationResult):
        self.cache[candidate.key] = result.fitness
        self.outcome.evaluations.append(EvaluatedCandidate(candidate=candidate, result=result))

    def __call__(self, solution: List[float]) -> float:
        try:
            candidate = self.space.decode(list(solution))
        except Exception as e:
            log.debug(f"AutoChunk: undecodable solution {solution}: {e}")
            return WORST_FITNESS

        if candidate.key in self.cache:
            return self.cache[candidate.key]

        if len(self.cache) >= self.max_evaluations or time.monotonic() > self.deadline:
            self.outcome.budget_exhausted = True
            return WORST_FITNESS

        future = asyncio.run_coroutine_threadsafe(self.evaluate(candidate), self.loop)
        try:
            result = future.result()
        except Exception as e:
            result = EvaluationResult(fitness=WORST_FITNESS, error=str(e))

        self.register(candidate, result)
        log.info(
            f"AutoChunk: evaluated {candidate.name} {candidate.config} -> fitness {result.fitness:.4f}"
            + (f" (error: {result.error})" if result.error else "")
        )
        return result.fitness


async def run_optimization(
    space: SearchSpace,
    evaluate: EvaluateFunction,
    algorithm: str,
    algorithm_parameters: Dict[str, Any],
    population_size: int,
    max_cycles: int,
    max_evaluations: int,
    max_runtime_seconds: float,
    seed: int | None = None,
    baseline: Candidate | None = None,
) -> OptimizationOutcome:
    """Search the best chunker configuration with the chosen pyVolutionary algorithm (maximization of the fitness).

    Args:
        space: the search space.
        evaluate: coroutine evaluating a candidate configuration.
        algorithm: name of the pyVolutionary optimizer class.
        algorithm_parameters: extra parameters of the configuration of the optimizer.
        population_size: size of the population.
        max_cycles: maximum number of generations.
        max_evaluations: maximum number of distinct configurations to evaluate.
        max_runtime_seconds: time budget.
        seed: optional random seed.
        baseline: optional configuration evaluated before the optimization (e.g. the current chunker); it does not
            count towards max_evaluations.
    """
    from pyvolutionary import ContinuousVariable, Task

    outcome = OptimizationOutcome(algorithm=algorithm)
    optimizer_class, config_class = resolve_algorithm(algorithm)

    loop = asyncio.get_running_loop()
    deadline = time.monotonic() + max_runtime_seconds
    bridge = _EvaluationBridge(space, evaluate, loop, outcome, max_evaluations, deadline)

    if baseline is not None:
        try:
            result = await evaluate(baseline)
        except Exception as e:
            result = EvaluationResult(fitness=WORST_FITNESS, error=str(e))
        bridge.register(baseline, result)
        # the baseline is a reference, not part of the budget
        bridge.max_evaluations += 1

    if space.dimension == 0:
        # a single chunker with nothing to tune: the only candidate
        candidate = space.decode([])
        if candidate.key not in bridge.cache:
            bridge.register(candidate, await evaluate(candidate))
        return outcome

    class ChunkerOptimizationTask(Task):
        # pyVolutionary declares the seed as a float, which numpy>=2 refuses in np.random.seed
        seed: int | None = None

        def objective_function(self, x: List[float]) -> float:
            return self.data["objective"](x)

    task = ChunkerOptimizationTask(
        variables=[
            ContinuousVariable(name=name, lower_bound=lower, upper_bound=upper)
            for name, (lower, upper) in zip(space.variable_names, space.bounds())
        ],
        minmax="max",
        seed=seed,
        data={"objective": bridge},
    )

    configuration = config_class(
        **{
            **algorithm_parameters,
            "population_size": population_size,
            "max_cycles": max_cycles,
            # never stop on the pyVolutionary error criterion (|1 - average fitness|), meaningless here
            "fitness_error": None,
        }
    )

    try:
        await asyncio.to_thread(optimizer_class(configuration).optimize, task)
    except Exception as e:
        log.error(f"AutoChunk: optimization with {algorithm} failed: {e}")
        outcome.error = str(e)

    return outcome
