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

## Training on GitHub Codespaces

Training spends most of its time computing physics, so more CPU cores make it faster. A
[GitHub Codespace](https://docs.github.com/en/codespaces) is a cloud machine with 2 to 32 cores,
set up from `.devcontainer/devcontainer.json` (Python 3.13, uv, `uv sync`). The free plan
includes 120 core-hours per month: about 7.5 hours on 16 cores.

1. Create a codespace on a branch, with 16 cores, either:
   - on github.com: **Code → Codespaces → ⋯ → New with options**, or
   - from your terminal; `create` asks which machine type to use (the first time, run
     `gh auth refresh -h github.com -s codespace`):

     ```powershell
     gh codespace create -R mohamadalichamseddine/deep-rl-flight-control -b main --idle-timeout 60m
     gh codespace ssh
     ```

2. Start training in the codespace's terminal. `nohup` keeps it running if you disconnect:

   ```bash
   cd /workspaces/deep-rl-flight-control
   mkdir -p runs/ppo
   nohup uv run python scripts/ppo/train_ppo.py --name my_run --parallel-rockets 16 > runs/ppo/my_run.log 2>&1 &
   tail -f runs/ppo/my_run.log    # watch it; Ctrl+C stops watching, not training
   ```

   The codespace stops after the idle timeout (at most 240 minutes) without terminal activity,
   which ends training: keep `tail -f` open, or set a timeout longer than the run.

3. Copy the results to your computer. They exist only in the codespace, because `runs/` is not
   committed:

   ```powershell
   gh codespace cp -r remote:/workspaces/deep-rl-flight-control/runs/ppo/my_run runs/ppo/
   ```

   Or right-click the folder in the codespace's file explorer → **Download**.

4. Stop the codespace when you're done (`gh codespace stop`), as it uses core-hours while it
   runs. Delete it once the results are copied (`gh codespace delete`): a stopped codespace
   still counts toward the 15 GB of free storage.

`--parallel-rockets` also changes the batch: each iteration collects parallel rockets ×
`n_steps` (1,024) steps, so 16 rockets make half as many network updates for the same number of
steps as 8.
