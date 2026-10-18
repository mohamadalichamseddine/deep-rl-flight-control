import math

import numpy as np
import pytest

from rocketlander.environment.rocket_landing_env import RocketLandingEnv
from rocketlander.evaluation.evaluation import run_test_set
from rocketlander.pd.pd_controller import PDController


def _observation(x=0.0, y=20.0, vx=0.0, vy=0.0, tilt=0.0, spin=0.0) -> np.ndarray:
    return np.array([x, y, vx, vy, math.sin(tilt), math.cos(tilt), spin, 100.0], dtype=np.float32)


@pytest.fixture(scope="module")
def controller() -> PDController:
    return PDController.for_env(RocketLandingEnv())


def test_actions_are_valid(controller: PDController) -> None:
    action = controller.act(_observation(x=30.0, vy=-25.0, tilt=0.3, spin=-1.0))
    assert action.shape == (3,) and action.dtype == np.float32
    assert np.all((action >= -1.0) & (action <= 1.0))


def test_hovering_upright_over_the_pad_uses_hover_thrust(controller: PDController) -> None:
    # Starting at exactly the target descent speed for this height, only the weight needs
    # holding: main thrust = weight, side nozzles off.
    gains, height = controller.gains, 20.0 - 1.44
    target_speed = gains.touchdown_speed + gains.descent_per_height * height
    action = controller.act(_observation(vy=-target_speed))
    throttles = (action + 1) / 2
    assert throttles[0] == pytest.approx(100 * 9.81 / 2000, abs=1e-3)
    assert throttles[1:] == pytest.approx([0.0, 0.0], abs=1e-6)


@pytest.mark.parametrize(("tilt", "firing"), [(-0.2, "right"), (0.2, "left")])
def test_side_nozzles_turn_the_rocket_back_upright(
    controller: PDController, tilt: float, firing: str
) -> None:
    # Nose tilted right (tilt < 0) needs a counterclockwise push: the right nozzle.
    throttles = (controller.act(_observation(tilt=tilt)) + 1) / 2
    left, right = throttles[1], throttles[2]
    assert (right > 0 and left == 0) if firing == "right" else (left > 0 and right == 0)


def test_off_to_the_right_it_tilts_left_to_come_back(controller: PDController) -> None:
    # 10 m right of the pad: the target tilt is nose-left (positive), so from upright the
    # left-turning (counterclockwise) right nozzle fires.
    throttles = (controller.act(_observation(x=10.0)) + 1) / 2
    assert throttles[2] > 0 and throttles[1] == 0


@pytest.mark.parametrize("level", [0, 1, 2])
def test_lands_the_first_test_episodes_of_every_level(level: int) -> None:
    env = RocketLandingEnv(level=level)
    results = run_test_set(PDController.for_env(env).act, env, episodes=8)
    assert [r.outcome for r in results] == ["landed"] * 8
