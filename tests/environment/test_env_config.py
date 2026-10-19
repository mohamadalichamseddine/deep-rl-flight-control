import pytest

from rocketlander.config_files import ENV_CONFIG_FILE
from rocketlander.environment.env_config import StartRanges, build_env_config


def test_default_config_loads_three_levels() -> None:
    config = build_env_config(ENV_CONFIG_FILE)
    assert config.gravity == 9.81
    assert len(config.levels) == 3
    assert config.levels[0].altitude == (50.0, 80.0)


def test_range_min_must_not_exceed_max() -> None:
    with pytest.raises(ValueError, match="altitude: min 80 is above max 50"):
        StartRanges(
            altitude=(80, 50),
            x=(0, 0),
            vx=(0, 0),
            vy=(0, 0),
            tilt_deg=(0, 0),
            spin_deg_s=(0, 0),
        )
