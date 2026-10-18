"""A classical PD autopilot: the baseline the trained network will be compared against."""

import math
from dataclasses import dataclass

import numpy as np

from rocketlander.environment.rocket_landing_env import RocketLandingEnv
from rocketlander.rocket.rocket_simulator import STANDARD_GRAVITY


@dataclass(frozen=True)
class PDGains:
    # Horizontal: desired sideways acceleration from the position and velocity errors.
    position_gain: float = 0.36  # [1/s²]
    velocity_gain: float = 1.2  # [1/s]
    max_tilt_deg: float = 15.0
    # Vertical: a target descent speed that slows as the ground approaches.
    descent_per_height: float = 0.25  # target speed per m of height [1/s]
    touchdown_speed: float = 0.15  # target speed at the ground [m/s], half the leg limit
    max_descent_speed: float = 12.0  # [m/s]
    vertical_gain: float = 1.5  # desired vertical acceleration per m/s of speed error [1/s]
    # Near the ground, come down only when over the pad and not drifting: the descent slows
    # to a hover while off-center, giving the horizontal loop time to finish.
    hold_height: float = 3.0  # [m]
    off_center_to_stop_descent: float = 1.0  # |x| [m] + |vx| [m/s] at which the descent stops
    # Attitude: desired angular acceleration from the tilt and spin errors. This loop is
    # faster than the horizontal one, which relies on it.
    tilt_gain: float = 4.0  # [1/s²]
    spin_gain: float = 3.6  # [1/s]


DEFAULT_GAINS = PDGains()


class PDController:
    """Cascaded PD autopilot, using only the observation (like the network will).

    Outer loop: from position and velocity, a target tilt (the rocket moves sideways by
    tilting its main thrust) and a main throttle (to follow a descent speed that slows near
    the ground). Inner loop: the side nozzles turn the rocket toward the target tilt.

    Build it with ``PDController.for_env(env)``, then call ``act(observation)``.
    """

    def __init__(
        self,
        mass: float,
        main_thrust: float,
        side_torque: float,
        inertia: float,
        y_when_standing: float,
        gravity: float = STANDARD_GRAVITY,
        gains: PDGains = DEFAULT_GAINS,
    ) -> None:
        self.mass = mass
        self.main_thrust = main_thrust  # [N]
        self.side_torque = side_torque  # torque of one side nozzle at full throttle [N·m]
        self.inertia = inertia  # [kg·m²]
        self.y_when_standing = y_when_standing  # center of mass on the ground [m]
        self.gravity = gravity
        self.gains = gains

    @classmethod
    def for_env(cls, env: RocketLandingEnv, gains: PDGains = DEFAULT_GAINS) -> "PDController":
        """Read the rocket's physical numbers, as an engineer designing it would know them."""
        side = env.rocket.nozzle("left")
        lever_arm = side.position[1] - env.simulator.center_of_mass_height
        return cls(
            mass=env.simulator.mass,
            main_thrust=env.rocket.nozzle("main").max_thrust,
            side_torque=side.max_thrust * lever_arm,
            inertia=env.simulator.moment_of_inertia,
            y_when_standing=env.y_when_standing,
            gains=gains,
        )

    def act(self, observation: np.ndarray) -> np.ndarray:
        """Throttles for [main, left, right] as environment actions in [-1, 1]."""
        x, y, vx, vy, sin_tilt, cos_tilt, spin, _ = (float(v) for v in observation)
        tilt = math.atan2(sin_tilt, cos_tilt)
        gains = self.gains
        height = max(y - self.y_when_standing, 0.0)  # of the leg tips above the ground

        # Horizontal: tilt so the main thrust pushes toward the pad, and brakes drift.
        # The main thrust points along the rocket axis, (-sin tilt, cos tilt).
        desired_ax = -gains.position_gain * x - gains.velocity_gain * vx
        max_tilt = math.radians(gains.max_tilt_deg)
        target_tilt = min(max(-math.atan2(desired_ax, self.gravity), -max_tilt), max_tilt)

        # Vertical: follow the descent speed profile; thrust also has to hold the weight,
        # and only its vertical part (cos tilt) counts.
        target_vy = -min(
            gains.max_descent_speed, gains.touchdown_speed + gains.descent_per_height * height
        )
        if height < gains.hold_height:
            off_center = abs(x) + abs(vx)
            target_vy *= min(max(1.0 - off_center / gains.off_center_to_stop_descent, 0.0), 1.0)
        desired_ay = gains.vertical_gain * (target_vy - vy)
        main = self.mass * (self.gravity + desired_ay) / (max(cos_tilt, 0.5) * self.main_thrust)

        # Attitude: the left nozzle turns the rocket clockwise (negative), the right one
        # counterclockwise (positive).
        torque = self.inertia * (gains.tilt_gain * (target_tilt - tilt) - gains.spin_gain * spin)
        left = max(-torque, 0.0) / self.side_torque
        right = max(torque, 0.0) / self.side_torque

        throttles = np.clip([main, left, right], 0.0, 1.0)
        return (2.0 * throttles - 1.0).astype(np.float32)
