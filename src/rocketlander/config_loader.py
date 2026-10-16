"""Load YAML config files into (nested) dataclasses.

Loading is strict: unknown keys, missing keys and wrong types raise ``ConfigError`` with the
location of the problem (e.g. ``nozzles[1].max_thrust``), so a typo in a config file fails
loudly instead of silently falling back to a default.

Supported field types: ``float``, ``int``, ``bool``, ``str``, nested dataclasses,
``list[X]``, ``tuple[X, ...]``, fixed-length tuples such as ``tuple[float, float]``,
``dict[str, X]``, ``Literal[...]``, unions such as ``X | None``, and ``Any``.
"""

import dataclasses
import types
import typing
from pathlib import Path
from typing import Any, Literal, get_args, get_origin, get_type_hints

import yaml


class ConfigError(ValueError):
    """A config file does not match the dataclass it is loaded into."""


def load_config[T](path: str | Path, cls: type[T]) -> T:
    """Read the YAML file at ``path`` and build an instance of the dataclass ``cls``."""
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    try:
        return from_dict(cls, data)
    except ConfigError as e:
        raise ConfigError(f"{path}: {e}") from None


def from_dict[T](cls: type[T], data: Any) -> T:
    """Build an instance of the dataclass ``cls`` from plain data such as parsed YAML."""
    if not (isinstance(cls, type) and dataclasses.is_dataclass(cls)):
        raise TypeError(f"{cls!r} is not a dataclass type")
    return _convert(cls, data, "")


def _convert(tp: Any, value: Any, where: str) -> Any:
    if isinstance(tp, type) and dataclasses.is_dataclass(tp):
        return _build_dataclass(tp, value, where)
    if tp is Any:
        return value

    origin = get_origin(tp) or tp
    args = get_args(tp)

    if origin is Literal:
        if value not in args:
            raise ConfigError(f"{_loc(where)}expected one of {list(args)}, got {value!r}")
        return value

    if origin in (typing.Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        for arg in args:
            if arg is type(None):
                continue
            try:
                return _convert(arg, value, where)
            except ConfigError:
                pass
        raise ConfigError(f"{_loc(where)}{value!r} does not match {tp}")

    if origin in (list, tuple):
        if not isinstance(value, list | tuple):
            raise ConfigError(f"{_loc(where)}expected a list, got {type(value).__name__}")
        if origin is tuple and args and args[-1] is not Ellipsis:
            if len(value) != len(args):
                raise ConfigError(f"{_loc(where)}expected {len(args)} items, got {len(value)}")
            item_types = args
        else:
            item_types = (args[0] if args else Any,) * len(value)
        items = [
            _convert(t, v, f"{where}[{i}]")
            for i, (t, v) in enumerate(zip(item_types, value, strict=True))
        ]
        return tuple(items) if origin is tuple else items

    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{_loc(where)}expected a mapping, got {type(value).__name__}")
        key_type, value_type = args or (Any, Any)
        return {
            _convert(key_type, k, f"{where}.{k}"): _convert(value_type, v, f"{where}.{k}")
            for k, v in value.items()
        }

    # bool is a subclass of int in Python, so it is excluded explicitly for numbers.
    if tp is float and isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    if tp is int and isinstance(value, int) and not isinstance(value, bool):
        return value
    if tp in (bool, str) and isinstance(value, tp):
        return value
    if tp not in (float, int, bool, str):
        raise TypeError(f"unsupported config field type {tp!r}")

    hint = ""
    if tp is float and isinstance(value, str) and _parses_as_float(value):
        hint = " (YAML reads exponents without a decimal point as text: write 1.0e3, not 1e3)"
    raise ConfigError(
        f"{_loc(where)}expected {tp.__name__}, got {type(value).__name__} {value!r}{hint}"
    )


def _build_dataclass(cls: type, value: Any, where: str) -> Any:
    if not isinstance(value, dict):
        raise ConfigError(
            f"{_loc(where)}expected a mapping for {cls.__name__}, got {type(value).__name__}"
        )
    hints = get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls) if f.init}

    unknown = sorted(str(k) for k in set(value) - set(fields))
    if unknown:
        raise ConfigError(
            f"{_loc(where)}unknown key(s) {unknown} for {cls.__name__}; "
            f"valid keys are {sorted(fields)}"
        )

    kwargs = {}
    for name, field in fields.items():
        path = f"{where}.{name}" if where else name
        if name in value:
            kwargs[name] = _convert(hints[name], value[name], path)
        elif field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING:
            raise ConfigError(f"{_loc(path)}missing required key")

    try:
        return cls(**kwargs)
    except ConfigError:
        raise
    except ValueError as e:  # validation in __post_init__
        raise ConfigError(f"{_loc(where)}{e}") from e


def _loc(where: str) -> str:
    return f"{where}: " if where else ""


def _parses_as_float(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True
