from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pytest

from rocketlander.config_loader import ConfigError, from_dict, load_config


@dataclass(frozen=True)
class Nozzle:
    name: str
    max_thrust: float
    position: tuple[float, float]

    def __post_init__(self) -> None:
        if self.max_thrust <= 0:
            raise ValueError(f"max_thrust must be positive, got {self.max_thrust}")


@dataclass(frozen=True)
class Vehicle:
    mass: float
    nozzles: list[Nozzle]
    level: Literal["easy", "hard"] = "easy"
    substeps: int = 5
    label: str | None = None
    gains: dict[str, float] = field(default_factory=dict)


VALID = {
    "mass": 100,
    "nozzles": [
        {"name": "main", "max_thrust": 2000.0, "position": [0.0, -1.24]},
        {"name": "left", "max_thrust": 25, "position": [-0.15, 1.46]},
    ],
}


def test_load_config_builds_nested_dataclasses(tmp_path: Path) -> None:
    path = tmp_path / "vehicle.yaml"
    path.write_text(
        """
mass: 100
level: hard
gains: {kp: 2, kd: 0.5}
nozzles:
  - {name: main, max_thrust: 2000.0, position: [0.0, -1.24]}
  - {name: left, max_thrust: 25, position: [-0.15, 1.46]}
""",
        encoding="utf-8",
    )

    vehicle = load_config(path, Vehicle)

    assert vehicle == Vehicle(
        mass=100.0,
        level="hard",
        gains={"kp": 2.0, "kd": 0.5},
        nozzles=[
            Nozzle("main", 2000.0, (0.0, -1.24)),
            Nozzle("left", 25.0, (-0.15, 1.46)),
        ],
    )
    assert isinstance(vehicle.mass, float)
    assert isinstance(vehicle.nozzles[0].position, tuple)


def test_defaults_are_used_for_missing_optional_keys() -> None:
    vehicle = from_dict(Vehicle, VALID)

    assert vehicle.level == "easy"
    assert vehicle.substeps == 5
    assert vehicle.label is None
    assert vehicle.gains == {}


def test_optional_field_accepts_value_or_none() -> None:
    assert from_dict(Vehicle, {**VALID, "label": "r100"}).label == "r100"
    assert from_dict(Vehicle, {**VALID, "label": None}).label is None


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({**VALID, "mas": 1.0}, r"unknown key\(s\) \['mas'\]"),
        ({"nozzles": VALID["nozzles"]}, r"mass: missing required key"),
        ({**VALID, "mass": "heavy"}, r"mass: expected float, got str"),
        ({**VALID, "mass": True}, r"mass: expected float, got bool"),
        ({**VALID, "substeps": 2.5}, r"substeps: expected int, got float"),
        ({**VALID, "level": "expert"}, r"level: expected one of \['easy', 'hard'\]"),
        ({**VALID, "nozzles": {"main": {}}}, r"nozzles: expected a list"),
        (
            {**VALID, "nozzles": [{**VALID["nozzles"][0], "max_thrust": "big"}]},
            r"nozzles\[0\]\.max_thrust: expected float",
        ),
        (
            {**VALID, "nozzles": [{**VALID["nozzles"][0], "position": [0.0]}]},
            r"nozzles\[0\]\.position: expected 2 items, got 1",
        ),
        (
            {**VALID, "nozzles": [{**VALID["nozzles"][0], "max_thrust": -1.0}]},
            r"nozzles\[0\]: max_thrust must be positive",
        ),
        (None, r"expected a mapping for Vehicle, got NoneType"),
    ],
)
def test_invalid_config_reports_location(data: object, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        from_dict(Vehicle, data)


def test_exponent_without_decimal_point_gets_a_hint() -> None:
    # PyYAML (YAML 1.1) parses 1e3 as the string "1e3", a classic config pitfall.
    with pytest.raises(ConfigError, match=r"write 1\.0e3"):
        from_dict(Vehicle, {**VALID, "mass": "1e3"})


def test_load_config_error_includes_file_path(tmp_path: Path) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text("mass: 100\n", encoding="utf-8")

    with pytest.raises(ConfigError, match=r"broken\.yaml: nozzles: missing required key"):
        load_config(path, Vehicle)


def test_from_dict_rejects_non_dataclass_types() -> None:
    with pytest.raises(TypeError, match="is not a dataclass"):
        from_dict(dict, {})
