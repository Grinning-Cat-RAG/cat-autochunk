import asyncio

from cat.services.factory.chunker import RecursiveTextChunkerSettings

try:
    from .. import evaluation, optimizer, search_space
except ImportError:  # pytest (see conftest.py)
    from autochunk_plugin import evaluation, optimizer, search_space


def _space():
    return search_space.SearchSpace([
        search_space.build_chunker_space(RecursiveTextChunkerSettings, None, None, {}),
    ])


async def _quadratic_fitness(candidate):
    """Synthetic fitness, best at chunk_size=600 and chunk_overlap=60."""
    size, overlap = candidate.config["chunk_size"], candidate.config["chunk_overlap"]
    fitness = 1 - ((size - 600) / 1000) ** 2 - ((overlap - 60) / 500) ** 2
    return evaluation.EvaluationResult(fitness=fitness, mrr=fitness)


def test_resolve_algorithm():
    optimizer_class, config_class = optimizer.resolve_algorithm("GreyWolfOptimization")
    assert optimizer_class.__name__ == "GreyWolfOptimization"
    assert config_class.__name__ == "GreyWolfOptimizationConfig"

    try:
        optimizer.resolve_algorithm("NotAnAlgorithm")
        raise AssertionError("ValueError expected")
    except ValueError as e:
        assert "GreyWolfOptimization" in str(e)

    algorithms = optimizer.available_algorithms()
    assert {"GreyWolfOptimization", "ParticleSwarmOptimization", "GeneticAlgorithmOptimization"} <= set(algorithms)
    assert "OptimizationAbstract" not in algorithms


def test_optimization_finds_a_good_configuration():
    outcome = asyncio.run(optimizer.run_optimization(
        space=_space(),
        evaluate=_quadratic_fitness,
        algorithm="GreyWolfOptimization",
        algorithm_parameters={},
        population_size=10,
        max_cycles=15,
        max_evaluations=500,
        max_runtime_seconds=120,
        seed=42,
    ))

    assert outcome.error is None
    best = outcome.best
    assert best is not None
    assert abs(best.candidate.config["chunk_size"] - 600) < 150
    assert best.result.fitness > 0.95
    # every distinct configuration is evaluated once
    keys = [e.candidate.key for e in outcome.evaluations]
    assert len(keys) == len(set(keys))


def test_optimization_with_algorithm_parameters():
    outcome = asyncio.run(optimizer.run_optimization(
        space=_space(),
        evaluate=_quadratic_fitness,
        algorithm="ParticleSwarmOptimization",
        algorithm_parameters={"c1": 0.1, "c2": 0.1, "w": [0.35, 1]},
        population_size=6,
        max_cycles=4,
        max_evaluations=100,
        max_runtime_seconds=120,
        seed=1,
    ))
    assert outcome.error is None
    assert outcome.best is not None


def test_optimization_respects_the_budget_and_registers_the_baseline():
    evaluated = []

    async def evaluate(candidate):
        evaluated.append(candidate.key)
        return await _quadratic_fitness(candidate)

    baseline = search_space.Candidate(
        settings_class=RecursiveTextChunkerSettings,
        config={"encoding_name": "cl100k_base", "chunk_size": 256, "chunk_overlap": 64},
    )
    outcome = asyncio.run(optimizer.run_optimization(
        space=_space(),
        evaluate=evaluate,
        algorithm="GreyWolfOptimization",
        algorithm_parameters={},
        population_size=10,
        max_cycles=10,
        max_evaluations=5,
        max_runtime_seconds=120,
        seed=3,
        baseline=baseline,
    ))

    assert outcome.budget_exhausted
    # the baseline plus max_evaluations candidates
    assert len(evaluated) == 6
    assert outcome.evaluations[0].candidate.key == baseline.key


def test_failing_evaluations_are_not_selected():
    async def evaluate(candidate):
        if candidate.config["chunk_size"] > 500:
            raise RuntimeError("boom")
        return await _quadratic_fitness(candidate)

    outcome = asyncio.run(optimizer.run_optimization(
        space=_space(),
        evaluate=evaluate,
        algorithm="GreyWolfOptimization",
        algorithm_parameters={},
        population_size=8,
        max_cycles=3,
        max_evaluations=100,
        max_runtime_seconds=120,
        seed=5,
    ))
    assert any(e.result.error for e in outcome.evaluations)
    assert outcome.best.candidate.config["chunk_size"] <= 500


def test_invalid_algorithm_parameters_are_reported():
    try:
        asyncio.run(optimizer.run_optimization(
            space=_space(),
            evaluate=_quadratic_fitness,
            algorithm="ParticleSwarmOptimization",  # c1, c2 and w are required
            algorithm_parameters={},
            population_size=4,
            max_cycles=2,
            max_evaluations=10,
            max_runtime_seconds=60,
        ))
        raise AssertionError("an error is expected")
    except Exception as e:
        assert "c1" in str(e)
