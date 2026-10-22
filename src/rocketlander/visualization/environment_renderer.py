"""Draws the landing scene with pygame: sky, ground, pad, rocket, flames, trail, and status."""

import math
import os
import random
import time
from collections.abc import Sequence
from typing import Any

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402 (must come after the line above)

from rocketlander.environment.landing_judge import describe_failures  # noqa: E402
from rocketlander.environment.outcome import Outcome  # noqa: E402
from rocketlander.rocket.rocket_simulator import RocketSimulator  # noqa: E402

WIDTH, HEIGHT = 960, 720  # [px]; multiples of 16, as video encoders prefer
MIN_VIEW_HEIGHT = 14.0  # [m] of world shown at the closest zoom, near the ground
MIN_ROCKET_PX = 48  # far away, the rocket is drawn at least this tall so it stays visible
PAD_HALF_WIDTH = 2.0  # [m]
HOLD_AT_END_S = 1.5  # how long the window shows the final frame of an episode
STATUS_PANEL_RIGHT = 310  # [px]

SKY_TOP, SKY_HORIZON = (20, 36, 72), (130, 170, 214)
GROUND, PAD, PAD_TARGET = (92, 76, 58), (140, 140, 146), (240, 196, 40)
PART_COLORS = {"body": (236, 238, 242), "engine block": (88, 96, 110), "leg": (36, 36, 40)}
FLAME_OUTER, FLAME_INNER = (255, 132, 28), (255, 228, 120)
TEXT, DIM_TEXT = (245, 245, 245), (190, 196, 206)
LANDED_COLOR, FAILED_COLOR = (70, 210, 100), (235, 70, 60)

# Flame size at full throttle [m], per nozzle name.
FLAME_LENGTH = {"main": 1.6}
FLAME_WIDTH = {"main": 0.24}
SIDE_FLAME_LENGTH, SIDE_FLAME_WIDTH = 0.5, 0.08

OUTCOME_TITLES = {
    Outcome.LANDED: "LANDED",
    Outcome.UNSAFE_TOUCHDOWN: "UNSAFE TOUCHDOWN",
    Outcome.LOST_CONTROL: "LOST CONTROL",
    Outcome.OUT_OF_BOUNDS: "OUT OF BOUNDS",
    Outcome.TIME_LIMIT: "TIME LIMIT",
}


class EnvironmentRenderer:
    """Draws frames of the landing scene in a window ("human") or off screen ("rgb_array").

    The camera frames the rocket and the pad together and zooms in as the rocket comes down.
    In the window, Space pauses, - and + halve or double the speed, and Esc or closing the
    window sets ``closed``. Frames are paced to real time at ``fps`` × speed.
    """

    def __init__(self, mode: str, fps: int) -> None:
        if mode not in ("human", "rgb_array"):
            raise ValueError(f"unknown render mode {mode!r}")
        self.mode = mode
        self.fps = fps
        self.speed = 1.0
        self.paused = False
        self.closed = False
        self.controls_hint = ""  # extra keys to show at the bottom, e.g. for flying by hand
        pygame.font.init()
        if mode == "human":
            pygame.display.init()
            self.surface = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Rocket landing")
        else:
            self.surface = pygame.Surface((WIDTH, HEIGHT))
        self._font = pygame.font.SysFont("consolas,menlo,dejavusansmono", 16)
        self._title_font = pygame.font.SysFont("segoeui,helvetica,dejavusans", 40, bold=True)
        self._clock = pygame.time.Clock()
        self._sky = _vertical_gradient(SKY_TOP, SKY_HORIZON)
        self._trail: list[tuple[float, float]] = []

    def start_episode(self) -> None:
        self._trail.clear()

    def draw(
        self,
        rocket_simulator: RocketSimulator,
        throttles: Sequence[float],
        time_s: float,
        level: int,
        info: dict[str, Any],
    ) -> None:
        """Draw one frame of the scene onto the surface."""
        state = rocket_simulator.state()
        self._trail.append((state.x, state.y))
        self._set_camera(state.x, state.y)
        # After touchdown the engines are cut, and the numbers shown are those the landing was
        # judged on: the state just before contact, not after Box2D's contact forces.
        shown_state = state
        if rocket_simulator.touchdown is not None:
            throttles = [0.0] * len(throttles)
            shown_state = rocket_simulator.touchdown.state

        self.surface.blit(self._sky, (0, 0))
        self._draw_ground()
        self._draw_trail()
        self._draw_rocket(rocket_simulator, throttles)
        self._draw_scale_bar()
        names = [nozzle.name for nozzle in rocket_simulator.rocket_spec.nozzles]
        self._draw_status(shown_state, names, throttles, time_s, level)
        if "outcome" in info:
            self._draw_outcome(info)
        if self.mode == "human":
            help_text = f"Space pause · -/+ speed ({self.speed:g}x) · Esc quit"
            if self.controls_hint:
                self._text(self.controls_hint, (12, HEIGHT - 50), TEXT)
            self._text(help_text, (12, HEIGHT - 26), DIM_TEXT)

    def show(self, episode_over: bool) -> None:
        """Put the drawn frame in the window and wait to keep real time.

        At the end of an episode, the final frame stays up for a moment.
        """
        pygame.display.flip()
        self._handle_events()
        if self.paused:
            label = self._title_font.render("Paused: press Space", True, TEXT)
            self.surface.blit(label, (WIDTH / 2 - label.get_width() / 2, HEIGHT / 2))
            pygame.display.flip()
            while self.paused and not self.closed:
                self._handle_events()
                self._clock.tick(30)
        self._clock.tick(self.fps * self.speed)
        if episode_over:
            end = time.monotonic() + HOLD_AT_END_S / self.speed
            while time.monotonic() < end and not self.closed:
                self._handle_events()
                self._clock.tick(30)

    def wait_until_closed(self) -> None:
        """Keep the last frame up until the person watching closes the window (or presses Esc)."""
        while not self.closed:
            self._handle_events()
            self._clock.tick(30)

    def to_array(self) -> np.ndarray:
        """The current frame as an RGB image of shape (height, width, 3)."""
        return np.transpose(pygame.surfarray.array3d(self.surface), (1, 0, 2))

    def close(self) -> None:
        if self.mode == "human":
            pygame.display.quit()

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.closed = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.closed = True
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self.speed = max(self.speed / 2, 1 / 8)
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    self.speed = min(self.speed * 2, 8.0)

    def _set_camera(self, x: float, y: float) -> None:
        # Show the rocket and the pad together; the view shrinks as the rocket comes down.
        view_height = max(MIN_VIEW_HEIGHT, 1.3 * y + 6.0, (abs(x) + 12.0) * HEIGHT / WIDTH)
        self._pixels_per_meter = HEIGHT / view_height  # [px/m]
        self._center_x = x / 2
        self._bottom = -0.1 * view_height

    def _to_screen(self, x: float, y: float) -> tuple[float, float]:
        return (
            WIDTH / 2 + (x - self._center_x) * self._pixels_per_meter,
            HEIGHT - (y - self._bottom) * self._pixels_per_meter,
        )

    def _draw_ground(self) -> None:
        _, ground_y = self._to_screen(0.0, 0.0)
        pygame.draw.rect(self.surface, GROUND, (0, ground_y, WIDTH, HEIGHT - ground_y + 1))
        for half_width, color in ((PAD_HALF_WIDTH, PAD), (0.5, PAD_TARGET)):
            left, top = self._to_screen(-half_width, 0.0)
            right, bottom = self._to_screen(half_width, -0.25)
            pygame.draw.rect(self.surface, color, (left, top, right - left, max(bottom - top, 3)))

    def _draw_trail(self) -> None:
        if len(self._trail) >= 2:
            points = [self._to_screen(x, y) for x, y in self._trail]
            pygame.draw.aalines(self.surface, (255, 255, 255), False, points)

    def _draw_rocket(self, rocket_simulator: RocketSimulator, throttles: Sequence[float]) -> None:
        state = rocket_simulator.state()
        rocket_spec = rocket_simulator.rocket_spec
        rocket_height = rocket_spec.body.height - min(leg.tip[1] for leg in rocket_spec.legs)
        pixels_per_meter = max(self._pixels_per_meter, MIN_ROCKET_PX / rocket_height)
        cx, cy = self._to_screen(state.x, state.y)

        def to_screen(point: tuple[float, float]) -> tuple[float, float]:
            # Around the center of mass, so a rocket enlarged for visibility stays in place.
            return (
                cx + (point[0] - state.x) * pixels_per_meter,
                cy - (point[1] - state.y) * pixels_per_meter,
            )

        for nozzle, throttle in zip(rocket_spec.nozzles, throttles, strict=True):
            if throttle <= 0.01:
                continue
            length = FLAME_LENGTH.get(nozzle.name, SIDE_FLAME_LENGTH)
            width = FLAME_WIDTH.get(nozzle.name, SIDE_FLAME_WIDTH)
            origin = rocket_simulator.point_to_world(nozzle.position)
            dx, dy = rocket_simulator.direction_to_world(nozzle.direction)
            exhaust = (-dx, -dy)  # the exhaust leaves opposite to the push
            flicker = random.uniform(0.85, 1.15)
            for fraction, color in ((1.0, FLAME_OUTER), (0.55, FLAME_INNER)):
                flame = _flame(
                    origin, exhaust, length * throttle * flicker * fraction, width * fraction
                )
                pygame.draw.polygon(self.surface, color, [to_screen(p) for p in flame])

        for name, corners in rocket_simulator.shapes():
            points = [to_screen(p) for p in corners]
            pygame.draw.polygon(self.surface, PART_COLORS.get(name, TEXT), points)
            pygame.draw.polygon(self.surface, (20, 20, 24), points, width=1)

    def _draw_scale_bar(self) -> None:
        # A bar of a round length, about 120 px long, so the zoom level is readable.
        target_m = 120 / self._pixels_per_meter
        length_m = min(
            (n for n in (1, 2, 5, 10, 20, 50, 100, 200)), key=lambda n: abs(n - target_m)
        )
        length_px = length_m * self._pixels_per_meter
        x0, y0 = WIDTH - 20 - length_px, HEIGHT - 20
        pygame.draw.line(self.surface, TEXT, (x0, y0), (x0 + length_px, y0), 2)
        for x in (x0, x0 + length_px):
            pygame.draw.line(self.surface, TEXT, (x, y0 - 5), (x, y0 + 5), 2)
        self._text(f"{length_m} m", (x0 + length_px / 2 - 16, y0 - 24), TEXT)

    def _draw_status(
        self,
        state: Any,
        names: Sequence[str],
        throttles: Sequence[float],
        time_s: float,
        level: int,
    ) -> None:
        panel = pygame.Surface((STATUS_PANEL_RIGHT - 10, 196), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 140))
        self.surface.blit(panel, (10, 10))
        lines = [
            f"time {time_s:6.2f} s     level {level}",
            f"altitude {state.y:7.1f} m",
            f"x        {state.x:7.2f} m",
            f"speed  vx {state.vx:6.2f}  vy {state.vy:6.2f} m/s",
            f"tilt {math.degrees(state.tilt):6.1f}°  spin {math.degrees(state.spin):6.1f}°/s",
        ]
        for i, line in enumerate(lines):
            self._text(line, (20, 18 + 22 * i), TEXT)
        for i, (name, throttle) in enumerate(zip(names, throttles, strict=True)):
            y = 18 + 22 * (len(lines) + i)
            self._text(f"{name:<5}", (20, y), DIM_TEXT)
            pygame.draw.rect(self.surface, DIM_TEXT, (80, y + 3, 160, 12), width=1)
            pygame.draw.rect(self.surface, FLAME_OUTER, (80, y + 3, 160 * throttle, 12))
            self._text(f"{throttle:4.0%}", (250, y), DIM_TEXT)

    def _draw_outcome(self, info: dict[str, Any]) -> None:
        outcome = info["outcome"]
        color = LANDED_COLOR if outcome == Outcome.LANDED else FAILED_COLOR
        # Centered in the space to the right of the status panel, one failure per line.
        center_x = (STATUS_PANEL_RIGHT + WIDTH) / 2
        title = self._title_font.render(OUTCOME_TITLES[outcome], True, color)
        self.surface.blit(title, (center_x - title.get_width() / 2, 30))
        for i, reason in enumerate(describe_failures(info["touchdown"], info["failed_checks"])):
            text = self._font.render(reason, True, TEXT)
            self.surface.blit(text, (center_x - text.get_width() / 2, 84 + 22 * i))

    def _text(
        self,
        text: str,
        position: tuple[float, float],
        color: tuple[int, int, int],
        font: pygame.font.Font | None = None,
    ) -> None:
        self.surface.blit((font or self._font).render(text, True, color), position)


def _flame(
    origin: tuple[float, float], direction: tuple[float, float], length: float, width: float
) -> list[tuple[float, float]]:
    """A triangle starting at the nozzle (base ``width`` wide) pointing along ``direction``."""
    (ox, oy), (dx, dy) = origin, direction
    nx, ny = -dy * width / 2, dx * width / 2  # half the base, across the flame
    return [(ox + nx, oy + ny), (ox - nx, oy - ny), (ox + dx * length, oy + dy * length)]


def _vertical_gradient(top: tuple[int, int, int], bottom: tuple[int, int, int]) -> pygame.Surface:
    surface = pygame.Surface((WIDTH, HEIGHT))
    for y in range(HEIGHT):
        f = y / (HEIGHT - 1)
        color = tuple(round(a + (b - a) * f) for a, b in zip(top, bottom, strict=True))
        pygame.draw.line(surface, color, (0, y), (WIDTH, y))
    return surface
