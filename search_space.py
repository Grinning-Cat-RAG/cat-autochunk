import enum
import json
import math
import types
import typing
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Tuple, Type

from pydantic.fields import FieldInfo

from cat.log import log
from cat.services.factory.chunker import ChunkerSettings
from cat.utils import SUFFIX_TO_CRYPT

CHUNKER_VARIABLE_NAME = "__chunker__"

# Upper bound (exclusive) of a discrete variable [0, n): the solution is floored to the index of the choice. Encoding
# the choices as a continuous interval keeps every choice equally likely for the continuous metaheuristics (the
# DiscreteVariable of pyVolutionary reaches its last choice only when the solution is clipped to the upper bound).
_DISCRETE_EPS = 1e-9


class ParameterKind(str, enum.Enum):
    INTEGER = "integer"
    FLOAT = "float"
    DISCRETE = "discrete"
    FIXED = "fixed"


@dataclass
class ParameterSpec:
    """How a field of a chunker settings class is explored by the optimizer."""
    name: str
    kind: ParameterKind
    lower: float | None = None
    upper: float | None = None
    choices: List[Any] = field(default_factory=list)
    value: Any = None  # for FIXED parameters

    @property
    def is_variable(self) -> bool:
        if self.kind == ParameterKind.FIXED:
            return False
        if self.kind == ParameterKind.DISCRETE:
            return len(self.choices) > 1
        return self.lower is not None and self.upper is not None and self.upper > self.lower

    def bounds(self) -> Tuple[float, float]:
        if self.kind == ParameterKind.DISCRETE:
            return 0.0, len(self.choices) - _DISCRETE_EPS
        return float(self.lower), float(self.upper)  # type: ignore[arg-type]

    def decode(self, value: float) -> Any:
        if self.kind == ParameterKind.DISCRETE:
            return self.choices[min(int(math.floor(value)), len(self.choices) - 1)]
        if self.kind == ParameterKind.INTEGER:
            return int(min(max(round(value), self.lower), self.upper))  # type: ignore[type-var]
        if self.kind == ParameterKind.FLOAT:
            return float(min(max(value, self.lower), self.upper))  # type: ignore[type-var]
        return self.value


@dataclass
class ChunkerSpace:
    settings_class: Type[ChunkerSettings]
    base_config: Dict[str, Any]
    parameters: List[ParameterSpec]

    @property
    def name(self) -> str:
        return self.settings_class.__name__

    @property
    def variable_parameters(self) -> List[ParameterSpec]:
        return [p for p in self.parameters if p.is_variable]


@dataclass
class Candidate:
    """A chunker configuration to be evaluated."""
    settings_class: Type[ChunkerSettings]
    config: Dict[str, Any]

    @property
    def name(self) -> str:
        return self.settings_class.__name__

    @property
    def key(self) -> str:
        return canonical_key(self.name, self.config)


def canonical_key(name: str, config: Dict[str, Any]) -> str:
    return json.dumps({"name": name, "config": config}, sort_keys=True, default=str)


def _unwrap_optional(annotation: Any) -> Tuple[Any, bool]:
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0], True
    return annotation, False


def _numeric_constraints(info: FieldInfo) -> Tuple[float | None, float | None]:
    lower, upper = None, None
    for meta in info.metadata:
        if getattr(meta, "ge", None) is not None:
            lower = meta.ge
        if getattr(meta, "gt", None) is not None:
            lower = meta.gt + (1 if isinstance(meta.gt, int) else 1e-6)
        if getattr(meta, "le", None) is not None:
            upper = meta.le
        if getattr(meta, "lt", None) is not None:
            upper = meta.lt - (1 if isinstance(meta.lt, int) else 1e-6)
    return lower, upper


def _heuristic_bounds(name: str, default: float, is_int: bool) -> Tuple[float, float]:
    """Bounds of a numeric parameter that declares no constraint, based on its default and its name."""
    lname = name.lower()
    if "overlap" in lname:
        return 0, max(default * 4, 256 if is_int else 0.5)
    if default == 0:
        return (0, 10) if is_int else (0.0, 1.0)
    if 0 < default <= 1 and not is_int:
        return 0.0, 1.0
    if default < 0:
        return default * 4, default / 4
    lower, upper = default / 4, default * 4
    if is_int:
        lower, upper = max(1, math.floor(lower)), max(2, math.ceil(upper))
    return lower, upper


def build_parameter_spec(name: str, info: FieldInfo, base_value: Any, override: Dict[str, Any] | None) -> ParameterSpec:
    """Build the exploration spec of a field of a chunker settings class.

    The bounds of a numeric field come from its default value (and its constraints), so that the search space does not
    drift from a run to the next one; they are widened to include the value in use.

    Args:
        name: name of the field.
        info: pydantic field info.
        base_value: the value of the field in the base configuration (current configuration or default).
        override: optional override from the plugin settings ({"min", "max"} | {"choices"} | {"fixed"}).
    """
    override = override or {}

    if "fixed" in override:
        return ParameterSpec(name=name, kind=ParameterKind.FIXED, value=override["fixed"])

    # secrets and credentials are never explored
    if any(suffix in name for suffix in SUFFIX_TO_CRYPT):
        return ParameterSpec(name=name, kind=ParameterKind.FIXED, value=base_value)

    if "choices" in override:
        choices = list(override["choices"])
        if not choices:
            return ParameterSpec(name=name, kind=ParameterKind.FIXED, value=base_value)
        return ParameterSpec(name=name, kind=ParameterKind.DISCRETE, choices=choices)

    annotation, _ = _unwrap_optional(info.annotation)

    if annotation is bool:
        return ParameterSpec(name=name, kind=ParameterKind.DISCRETE, choices=[False, True])

    if typing.get_origin(annotation) is Literal:
        return ParameterSpec(name=name, kind=ParameterKind.DISCRETE, choices=list(typing.get_args(annotation)))

    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        return ParameterSpec(name=name, kind=ParameterKind.DISCRETE, choices=[e.value for e in annotation])

    if annotation in (int, float):
        is_int = annotation is int
        if "min" in override and "max" in override:
            lower, upper = override["min"], override["max"]
        else:
            default = info.get_default(call_default_factory=True) if not info.is_required() else None
            reference = default if isinstance(default, (int, float)) and not isinstance(default, bool) else base_value
            if reference is None or isinstance(reference, bool):
                return ParameterSpec(name=name, kind=ParameterKind.FIXED, value=base_value)
            lower, upper = _heuristic_bounds(name, float(reference), is_int)
            if isinstance(base_value, (int, float)) and not isinstance(base_value, bool):
                lower, upper = min(lower, base_value), max(upper, base_value)

        c_lower, c_upper = _numeric_constraints(info)
        if c_lower is not None:
            lower = max(lower, c_lower)
        if c_upper is not None:
            upper = min(upper, c_upper)
        lower = override.get("min", lower)
        upper = override.get("max", upper)
        if is_int:
            lower, upper = int(math.ceil(lower)), int(math.floor(upper))
        if upper <= lower:
            return ParameterSpec(name=name, kind=ParameterKind.FIXED, value=base_value)
        return ParameterSpec(
            name=name, kind=ParameterKind.INTEGER if is_int else ParameterKind.FLOAT, lower=lower, upper=upper,
        )

    # strings, lists, objects... are not explored
    return ParameterSpec(name=name, kind=ParameterKind.FIXED, value=base_value)


def build_chunker_space(
    settings_class: Type[ChunkerSettings],
    current_name: str | None,
    current_config: Dict[str, Any] | None,
    overrides: Dict[str, Dict[str, Any]],
) -> ChunkerSpace | None:
    """Build the search space of a chunker settings class.

    The base configuration is the current one when the class is the active chunker (so that the non-explored fields
    keep the values chosen by the user), the defaults of the class otherwise. Returns None when the class cannot be
    configured (e.g. required fields without a default value).
    """
    is_current = current_name == settings_class.__name__ and current_config is not None

    base_config: Dict[str, Any] = {}
    for name, info in settings_class.model_fields.items():
        if is_current and name in current_config:  # type: ignore[operator]
            base_config[name] = current_config[name]  # type: ignore[index]
        elif not info.is_required():
            base_config[name] = info.get_default(call_default_factory=True)
        elif f"{settings_class.__name__}.{name}" in overrides and "fixed" in overrides[f"{settings_class.__name__}.{name}"]:
            base_config[name] = overrides[f"{settings_class.__name__}.{name}"]["fixed"]
        else:
            log.warning(
                f"AutoChunk: chunker {settings_class.__name__} skipped, field '{name}' is required and has no value "
                f"(set it through search_space_overrides with {{\"fixed\": ...}})"
            )
            return None

    try:
        base_config = settings_class(**base_config).model_dump(mode="json")
    except Exception as e:
        log.warning(f"AutoChunk: chunker {settings_class.__name__} skipped, invalid base configuration: {e}")
        return None

    parameters = [
        build_parameter_spec(name, info, base_config.get(name), overrides.get(f"{settings_class.__name__}.{name}"))
        for name, info in settings_class.model_fields.items()
    ]
    return ChunkerSpace(settings_class=settings_class, base_config=base_config, parameters=parameters)


class SearchSpace:
    """The (conditional) search space of the chunker optimization.

    The solution vector is: [chunker index] + [parameters of chunker 1] + [parameters of chunker 2] + ... Only the
    parameters of the selected chunker are read when decoding a solution.
    """

    def __init__(self, chunker_spaces: List[ChunkerSpace]):
        if not chunker_spaces:
            raise ValueError("The search space needs at least one chunker")
        self.chunker_spaces = chunker_spaces

        # layout of the solution vector: (variable name, bounds)
        self._layout: List[Tuple[str, Tuple[float, float]]] = []
        self._chunker_selector = ParameterSpec(
            name=CHUNKER_VARIABLE_NAME, kind=ParameterKind.DISCRETE, choices=list(range(len(chunker_spaces))),
        )
        self._offsets: List[Tuple[int, int]] = []

        position = 0
        if self._chunker_selector.is_variable:
            self._layout.append((CHUNKER_VARIABLE_NAME, self._chunker_selector.bounds()))
            position += 1

        for space in chunker_spaces:
            start = position
            for parameter in space.variable_parameters:
                self._layout.append((f"{space.name}.{parameter.name}", parameter.bounds()))
                position += 1
            self._offsets.append((start, position))

    @property
    def dimension(self) -> int:
        return len(self._layout)

    @property
    def variable_names(self) -> List[str]:
        return [name for name, _ in self._layout]

    def bounds(self) -> List[Tuple[float, float]]:
        return [b for _, b in self._layout]

    def decode(self, solution: List[float]) -> Candidate:
        """Decode a solution vector into the chunker configuration to evaluate."""
        if len(solution) != self.dimension:
            raise ValueError(f"Invalid solution size {len(solution)}, expected {self.dimension}")

        index = 0
        if self._chunker_selector.is_variable:
            index = self._chunker_selector.decode(solution[0])

        space = self.chunker_spaces[index]
        start, end = self._offsets[index]
        values = dict(zip([p.name for p in space.variable_parameters], solution[start:end]))

        config = dict(space.base_config)
        for parameter in space.parameters:
            if parameter.name in values:
                config[parameter.name] = parameter.decode(values[parameter.name])
            elif parameter.kind == ParameterKind.FIXED:
                config[parameter.name] = parameter.value

        return Candidate(settings_class=space.settings_class, config=repair_config(config))


def repair_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Repair the common inconsistencies of a chunker configuration: the overlap cannot exceed half of the size."""
    config = dict(config)
    sizes = [k for k in config if "size" in k.lower() and "overlap" not in k.lower()]
    for key in [k for k in config if "overlap" in k.lower()]:
        value = config[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        # the size the overlap refers to: e.g. chunk_overlap -> chunk_size
        prefix = key.lower().replace("overlap", "")
        size_key = next((s for s in sizes if s.lower().replace("size", "") == prefix), sizes[0] if sizes else None)
        if size_key is None:
            continue
        size = config[size_key]
        if isinstance(size, (int, float)) and not isinstance(size, bool) and value > size / 2:
            config[key] = type(value)(size // 2 if isinstance(value, int) else size / 2)
    return config
