import imageio
import numpy as np
import pygame
import pytest

from rocketlander.environment.env_config import EnvConfig
from rocketlander.environment.rocket_landing_env import RocketLandingEnv
from rocketlander.rocket.rocket_spec import RocketSpec
from rocketlander.rocket.rocket_state import RocketState
from rocketlander.visualization.environment_renderer import HEIGHT, WIDTH
from rocketlander.visualization.video_writer import VideoWriter

MAIN_ON = np.array([1.0, -1.0, -1.0], dtype=np.float32)


def test_rgb_array_mode_returns_frames_that_follow_the_flight(
    rocket_spec: RocketSpec, env_config: EnvConfig
) -> None:
    env = RocketLandingEnv(rocket_spec, env_config, level=0, render_mode="rgb_array")
    env.reset(seed=0)

    first = env.render()
    for _ in range(25):
        env.step(MAIN_ON)
    later = env.render()

    assert first.shape == (HEIGHT, WIDTH, 3)
    assert first.dtype == np.uint8
    assert not np.array_equal(first, later)
    env.close()


def test_render_without_a_render_mode_draws_nothing(
    rocket_spec: RocketSpec, env_config: EnvConfig
) -> None:
    env = RocketLandingEnv(rocket_spec, env_config, level=0, render_mode=None)
    env.reset(seed=0)
    with pytest.warns(UserWarning, match="render_mode=None"):
        assert env.render() is None
    assert env.environment_renderer is None


def test_unknown_render_mode_is_rejected(rocket_spec: RocketSpec, env_config: EnvConfig) -> None:
    with pytest.raises(ValueError, match="render_mode must be one of"):
        RocketLandingEnv(rocket_spec, env_config, level=0, render_mode="video")


def test_human_mode_runs_a_live_window(
    rocket_spec: RocketSpec, env_config: EnvConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")  # an invisible window, for testing
    env = RocketLandingEnv(rocket_spec, env_config, level=0, render_mode="human")
    env.reset(options={"state": RocketState(0.0, 50.0, 0.0, -5.0)})

    for _ in range(5):
        env.step(MAIN_ON)

    assert env.environment_renderer is not None and not env.window_closed
    env.close()
    assert env.environment_renderer is None


def test_the_window_stays_open_until_it_is_closed(
    rocket_spec: RocketSpec, env_config: EnvConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")  # an invisible window, for testing
    env = RocketLandingEnv(rocket_spec, env_config, level=0, render_mode="human")
    env.reset(options={"state": RocketState(0.0, 50.0, 0.0, -5.0)})
    assert not env.window_closed

    pygame.event.post(pygame.event.Event(pygame.QUIT))  # the person closes the window
    env.environment_renderer.wait_until_closed()

    assert env.window_closed
    env.close()


def test_video_writer_holds_the_last_frame(tmp_path) -> None:
    path = tmp_path / "flight.mp4"
    frames = [np.full((48, 64, 3), i * 20, dtype=np.uint8) for i in range(10)]

    with VideoWriter(path, fps=50) as video:
        for frame in frames:
            video.add(frame)
        video.hold(0.2)  # 10 more frames at 50 fps

    assert video.frames == 20
    with imageio.get_reader(path) as reader:
        assert reader.count_frames() == 20
