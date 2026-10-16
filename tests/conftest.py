from pathlib import Path

import pytest

from rocketlander.rocket_spec import RocketSpec, load_rocket

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture(scope="session")
def rocket() -> RocketSpec:
    return load_rocket(CONFIGS / "rocket" / "r100.yaml")
