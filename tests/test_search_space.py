import enum
from typing import Literal, Type

from pydantic import Field

from cat.services.factory.chunker import BaseChunker, ChunkerSettings, RecursiveTextChunkerSettings

try:
    from ..search_space import CHUNKER_VARIABLE_NAME, ParameterKind, SearchSpace, build_chunker_space, repair_config
except ImportError:  # pytest (see conftest.py)
    from autochunk_plugin.search_space import (
        CHUNKER_VARIABLE_NAME, ParameterKind, SearchSpace, build_chunker_space, repair_config,
    )


class _Mode(enum.Enum):
    FAST = "fast"
    SLOW = "slow"


class _FakeChunker(BaseChunker):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def _get_splitter(self):
        return None

    async def split_documents(self, documents):
        return list(documents)


class FakeChunkerSettings(ChunkerSettings):
    max_chunk_size: int = Field(default=400, ge=100, le=800)
    threshold: float = 0.5
    keep_separator: bool = True
    strategy: Literal["a", "b", "c"] = "a"
    mode: _Mode = _Mode.FAST
    language: str = "en"
    api_key: float = 3.0

    @classmethod
    def pyclass(cls) -> Type[_FakeChunker]:
        return _FakeChunker


class RequiredFieldChunkerSettings(ChunkerSettings):
    model_name: str

    @classmethod
    def pyclass(cls) -> Type[_FakeChunker]:
        return _FakeChunker


def _params(space):
    return {p.name: p for p in space.parameters}


def test_recursive_text_chunker_space():
    space = build_chunker_space(RecursiveTextChunkerSettings, None, None, {})
    params = _params(space)

    assert params["encoding_name"].kind == ParameterKind.FIXED
    assert params["encoding_name"].value == "cl100k_base"
    assert params["chunk_size"].kind == ParameterKind.INTEGER
    assert (params["chunk_size"].lower, params["chunk_size"].upper) == (64, 1024)
    assert params["chunk_overlap"].kind == ParameterKind.INTEGER
    assert params["chunk_overlap"].lower == 0
    assert [p.name for p in space.variable_parameters] == ["chunk_size", "chunk_overlap"]


def test_base_config_uses_the_current_configuration_of_the_active_chunker():
    current = {"encoding_name": "o200k_base", "chunk_size": 512, "chunk_overlap": 32}
    space = build_chunker_space(RecursiveTextChunkerSettings, "RecursiveTextChunkerSettings", current, {})

    assert space.base_config == current
    assert _params(space)["encoding_name"].value == "o200k_base"
    # the bounds come from the default value (stable across runs)...
    assert (_params(space)["chunk_size"].lower, _params(space)["chunk_size"].upper) == (64, 1024)

    # ... widened to include the current value
    current = {"encoding_name": "o200k_base", "chunk_size": 3000, "chunk_overlap": 32}
    space = build_chunker_space(RecursiveTextChunkerSettings, "RecursiveTextChunkerSettings", current, {})
    assert (_params(space)["chunk_size"].lower, _params(space)["chunk_size"].upper) == (64, 3000)

    # another chunker is active: the defaults are used
    other = build_chunker_space(RecursiveTextChunkerSettings, "FakeChunkerSettings", current, {})
    assert other.base_config["chunk_size"] == 256


def test_parameter_kinds_and_constraints():
    params = _params(build_chunker_space(FakeChunkerSettings, None, None, {}))

    assert params["max_chunk_size"].kind == ParameterKind.INTEGER
    # the pydantic constraints narrow the heuristic bounds (100..1600)
    assert (params["max_chunk_size"].lower, params["max_chunk_size"].upper) == (100, 800)
    assert params["threshold"].kind == ParameterKind.FLOAT
    assert (params["threshold"].lower, params["threshold"].upper) == (0.0, 1.0)
    assert params["keep_separator"].choices == [False, True]
    assert params["strategy"].choices == ["a", "b", "c"]
    assert params["mode"].choices == ["fast", "slow"]
    assert params["language"].kind == ParameterKind.FIXED
    # secrets are never explored
    assert params["api_key"].kind == ParameterKind.FIXED


def test_overrides():
    overrides = {
        "FakeChunkerSettings.max_chunk_size": {"min": 200, "max": 300},
        "FakeChunkerSettings.threshold": {"fixed": 0.7},
        "FakeChunkerSettings.language": {"choices": ["en", "it"]},
    }
    params = _params(build_chunker_space(FakeChunkerSettings, None, None, overrides))

    assert (params["max_chunk_size"].lower, params["max_chunk_size"].upper) == (200, 300)
    assert params["threshold"].kind == ParameterKind.FIXED and params["threshold"].value == 0.7
    assert params["language"].kind == ParameterKind.DISCRETE and params["language"].choices == ["en", "it"]


def test_chunker_with_required_field_is_skipped_unless_fixed():
    assert build_chunker_space(RequiredFieldChunkerSettings, None, None, {}) is None

    space = build_chunker_space(
        RequiredFieldChunkerSettings, None, None, {"RequiredFieldChunkerSettings.model_name": {"fixed": "m"}},
    )
    assert space.base_config == {"model_name": "m"}


def test_search_space_layout_and_decode():
    spaces = [
        build_chunker_space(RecursiveTextChunkerSettings, None, None, {}),
        build_chunker_space(FakeChunkerSettings, None, None, {}),
    ]
    search_space = SearchSpace(spaces)

    assert search_space.variable_names[0] == CHUNKER_VARIABLE_NAME
    assert search_space.dimension == 1 + 2 + 5  # selector + RecursiveText + Fake (language and api_key fixed)

    lower = [b[0] for b in search_space.bounds()]
    upper = [b[1] for b in search_space.bounds()]

    first = search_space.decode(lower)
    assert first.settings_class is RecursiveTextChunkerSettings
    assert first.config == {"encoding_name": "cl100k_base", "chunk_size": 64, "chunk_overlap": 0}

    # the upper bound of the discrete selector is the last chunker (not reached only by clipping)
    last = search_space.decode(upper)
    assert last.settings_class is FakeChunkerSettings
    assert last.config["max_chunk_size"] == 800
    assert last.config["keep_separator"] is True
    assert last.config["strategy"] == "c"
    assert last.config["mode"] == "slow"
    assert last.config["language"] == "en"

    # middle of the selector interval [0, 2): index 1
    middle = list(lower)
    middle[0] = 1.0
    assert search_space.decode(middle).settings_class is FakeChunkerSettings

    # every decoded configuration is valid for its settings class
    for solution in (lower, upper, middle):
        candidate = search_space.decode(solution)
        candidate.settings_class.model_validate(candidate.config)


def test_single_chunker_has_no_selector():
    search_space = SearchSpace([build_chunker_space(RecursiveTextChunkerSettings, None, None, {})])
    assert CHUNKER_VARIABLE_NAME not in search_space.variable_names
    assert search_space.dimension == 2

    candidate = search_space.decode([300.4, 20.6])
    assert candidate.config["chunk_size"] == 300
    assert candidate.config["chunk_overlap"] == 21


def test_overlap_is_repaired():
    assert repair_config({"chunk_size": 100, "chunk_overlap": 80})["chunk_overlap"] == 50
    assert repair_config({"chunk_size": 100, "chunk_overlap": 20})["chunk_overlap"] == 20
    assert repair_config({"chunk_overlap": 80}) == {"chunk_overlap": 80}

    search_space = SearchSpace([build_chunker_space(RecursiveTextChunkerSettings, None, None, {})])
    candidate = search_space.decode([64, 1024])
    assert candidate.config["chunk_overlap"] == 32


def test_candidate_keys_identify_configurations():
    search_space = SearchSpace([build_chunker_space(RecursiveTextChunkerSettings, None, None, {})])
    assert search_space.decode([300.1, 20.2]).key == search_space.decode([299.9, 19.8]).key
    assert search_space.decode([300, 20]).key != search_space.decode([301, 20]).key
