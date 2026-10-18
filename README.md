# Autonomous Rocket Landing
Autonomous rocket landing using deep reinforcement learning (PPO), neural networks, and physics simulation

## Try it

```powershell
uv run python scripts/environment/fly_manually.py        # fly the rocket yourself (arrow keys)
uv run python scripts/environment/fly_manually.py --flights 1 --record runs/my_flight.mp4
uv run python scripts/evaluation/watch_controller.py --level 2  # watch the PD autopilot land
uv run python scripts/evaluation/evaluate_controller.py  # test sets: metrics and plots in runs/
uv run python scripts/rocket/draw_rocket.py --show       # the rocket as Box2D sees it
uv run python scripts/rocket/scripted_flight.py --show   # a fixed command sequence, plotted
uv run python scripts/environment/random_episodes.py     # random actions: env check and speed
```

## Project layout

The code in `src/rocketlander/`, the scripts in `scripts/`, and the tests in `tests/` use the
same folders:

| Folder | What it holds |
|---|---|
| `rocket/` | the rocket: its description, its state, and its physics (Box2D) |
| `environment/` | the landing task: observations, actions, rewards, endings |
| `pd/` | the hand-written PD autopilot, the baseline |
| `ppo/` | the neural network, its PPO training, and the trained network as a controller |
| `evaluation/` | measuring any controller on the fixed test sets |
| `visualization/` | drawing flights: the live window and video files |

Settings live in `configs/`: `rocket/` (the rocket's shape and engines), `env/` (landing
rules, rewards, starting conditions), and `train/` (PPO training settings).

## Development

Requires [uv](https://docs.astral.sh/uv/). It creates the virtual environment in `.venv/` and
installs the exact versions pinned in `uv.lock`, and it downloads Python 3.13 if needed.

```powershell
uv sync                      # create .venv/ and install dependencies
uv run pytest                # run the tests
uv run ruff check .          # lint
uv run ruff format .         # format
```

`uv run <command>` runs inside `.venv/` without activating it. To activate it in your shell instead:

```powershell
.venv\Scripts\Activate.ps1   # PowerShell (Windows)
source .venv/bin/activate    # bash/zsh (Linux, macOS)
```
