"""Settings of the landing environment, loaded from a YAML file such as configs/env/default.yaml."""

from dataclasses import dataclass
from pathlib import Path

from rocketlander.config_loader import load_config

Range = tuple[float, float]  # [min, max]


@dataclass(frozen=True)
class LandingCriteria:
    """Limits a touchdown must stay within to count as a safe landing."""

    max_off_center: float  # distance from the pad center [m]
    max_vx: float  # [m/s]
    max_vy: float  # [m/s]
    max_tilt_deg: float
    max_spin_deg_s: float


@dataclass(frozen=True)
class FlightLimits:
    max_x: float  # [m]
    max_altitude: float  # [m]
    max_tilt_deg: float
    time_limit: float  # [s]


@dataclass(frozen=True)
class RewardSettings:
    """Weights of the per-step reward, and the bonus or penalty given when an episode ends."""

    distance_weight: float
    speed_weight: float
    tilt_weight: float
    fuel_weight: float
    landed_bonus: float
    unsafe_touchdown_penalty: float
    penalty_per_limit_missed: float
    lost_control_penalty: float
    time_limit_penalty: float


@dataclass(frozen=True)
class StartRanges:
    """Starting conditions for one difficulty level; each episode draws from these ranges."""

    altitude: Range  # center of mass [m]
    x: Range  # [m]
    vx: Range  # [m/s]
    vy: Range  # [m/s]
    tilt_deg: Range
    spin_deg_s: Range

    def __post_init__(self) -> None:
        for name in ("altitude", "x", "vx", "vy", "tilt_deg", "spin_deg_s"):
            low, high = getattr(self, name)
            if low > high:
                raise ValueError(f"{name}: min {low} is above max {high}")


@dataclass(frozen=True)
class EnvConfig:
    dt: float  # control step [s]
    substeps: int  # Box2D steps per control step
    landing: LandingCriteria
    limits: FlightLimits
    reward: RewardSettings
    levels: tuple[StartRanges, ...]


def load_env_config(path: str | Path) -> EnvConfig:
    return load_config(path, EnvConfig)
