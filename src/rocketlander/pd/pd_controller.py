"""A classical PD autopilot: the baseline the trained network will be compared against."""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from rocketlander.environment.rocket_landing_env import observation_to_state, throttles_to_action
from rocketlander.rocket.rocket_simulator import RocketSimulator


@dataclass(frozen=True)
class PDGains:
    """How strongly the autopilot reacts; explained in configs/pd/gains.yaml."""

    position_gain: float  # [1/s²]
    velocity_gain: float  # [1/s]
    max_tilt_deg: float
    descent_per_height: float  # [1/s]
    touchdown_speed: float  # [m/s]
    max_descent_speed: float  # [m/s]
    vertical_gain: float  # [1/s]
    hold_height: float  # [m]
    off_center_to_stop_descent: float  # |x| [m] + |vx| [m/s]
    tilt_gain: float  # [1/s²]
    spin_gain: float  # [1/s]


def build_pd_gains(path: str | Path) -> PDGains:
    """Read a PD gains file (YAML) and build the gains from it."""
    with Path(path).open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return PDGains(**config)


class PDController:
    """Cascaded PD autopilot, using only the observation (like the network will).

    Outer loop: from position and velocity, a target tilt (the rocket moves sideways by
    tilting its main thrust) and a main throttle (to follow a descent speed that slows near
    the ground). Inner loop: the side nozzles turn the rocket toward the target tilt.

    Build it with ``build_pd_controller(rocket_simulator, pd_gains)``, then call
    ``act(observation)``.
    """

    def __init__(
        self,
        mass: float,
        main_thrust: float,
        side_torque: float,
        inertia: float,
        y_when_standing: float,
        gravity: float,
        pd_gains: PDGains,
    ) -> None:
        self.mass = mass
        self.main_thrust = main_thrust  # [N]
        self.side_torque = side_torque  # torque of one side nozzle at full throttle [N·m]
        self.inertia = inertia  # [kg·m²]
        self.y_when_standing = y_when_standing  # center of mass on the ground [m]
        self.gravity = gravity
        self.pd_gains = pd_gains

    def act(self, observation: np.ndarray) -> np.ndarray:
        """Throttles for [main, left, right] as environment actions in [-1, 1]."""
        state = observation_to_state(observation)
        pd_gains = self.pd_gains
        height = max(state.y - self.y_when_standing, 0.0)  # of the leg tips above the ground

        # Horizontal: tilt so the main thrust pushes toward the pad, and brakes drift.
        # The main thrust points along the rocket axis, (-sin tilt, cos tilt).
        desired_ax = -pd_gains.position_gain * state.x - pd_gains.velocity_gain * state.vx
        max_tilt = math.radians(pd_gains.max_tilt_deg)
        target_tilt = min(max(-math.atan2(desired_ax, self.gravity), -max_tilt), max_tilt)

        # Vertical: follow the descent speed profile; thrust also has to hold the weight,
        # and only its vertical part (cos tilt) counts.
        target_vy = -min(
            pd_gains.max_descent_speed,
            pd_gains.touchdown_speed + pd_gains.descent_per_height * height,
        )
        if height < pd_gains.hold_height:
            off_center = abs(state.x) + abs(state.vx)
            slowdown = 1.0 - off_center / pd_gains.off_center_to_stop_descent
            target_vy *= min(max(slowdown, 0.0), 1.0)
        desired_ay = pd_gains.vertical_gain * (target_vy - state.vy)
        cos_tilt = math.cos(state.tilt)
        main = self.mass * (self.gravity + desired_ay) / (max(cos_tilt, 0.5) * self.main_thrust)

        # Attitude: the left nozzle turns the rocket clockwise (negative), the right one
        # counterclockwise (positive).
        torque = self.inertia * (
            pd_gains.tilt_gain * (target_tilt - state.tilt) - pd_gains.spin_gain * state.spin
        )
        left = max(-torque, 0.0) / self.side_torque
        right = max(torque, 0.0) / self.side_torque

        return throttles_to_action(np.clip([main, left, right], 0.0, 1.0))


def build_pd_controller(rocket_simulator: RocketSimulator, pd_gains: PDGains) -> PDController:
    """The autopilot for this rocket, from its physical numbers, as an engineer designing it
    would know them."""
    rocket_spec = rocket_simulator.rocket_spec
    side = rocket_spec.get_nozzle("left")
    lever_arm = side.position[1] - rocket_simulator.center_of_mass_height
    return PDController(
        mass=rocket_simulator.mass,
        main_thrust=rocket_spec.get_nozzle("main").max_thrust,
        side_torque=side.max_thrust * lever_arm,
        inertia=rocket_simulator.moment_of_inertia,
        y_when_standing=rocket_simulator.y_when_standing,
        gravity=rocket_simulator.gravity,
        pd_gains=pd_gains,
    )
