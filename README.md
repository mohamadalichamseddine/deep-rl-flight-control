# Rocket Landing Controller with Deep Reinforcement Learning

A 2D rocket, simulated with Box2D, learns to land on a pad. It is trained with PPO
(Stable-Baselines3) using our own PyTorch network, and compared against a hand-written PD
autopilot.

## Setup

Requires [uv](https://docs.astral.sh/uv/). In the project folder:

```powershell
uv sync      # creates .venv/ with Python 3.13 and the exact versions pinned in uv.lock
```

Every command below starts with `uv run`, which runs it inside `.venv/` without activating it.

## Try it (no training needed)

```powershell
uv run python scripts/environment/fly_manually.py              # fly the rocket yourself: arrow keys, Esc quits
uv run python scripts/evaluation/watch_controller.py           # watch the PD autopilot land
uv run python scripts/evaluation/watch_controller.py --level 2 # the same, from harder starts
uv run python scripts/rocket/draw_rocket.py --show             # the rocket as Box2D sees it
uv run python scripts/rocket/scripted_flight.py --show         # a fixed command sequence, plotted
uv run python scripts/environment/random_episodes.py           # random actions: checks the environment and its speed
```

Levels 0, 1 and 2 are how hard the starting conditions are (height, speed, tilt; see
`configs/env/default.yaml`). Add `--record runs/my_flight.mp4` to `fly_manually.py` or
`watch_controller.py` to save the flights as a video.

## Train a network

### Where to train

- **On your computer.** Training flies 8 rockets at once (`parallel_rockets` in
  `configs/train/ppo.yaml`), ideally one per CPU core; with fewer cores, pass
  `--parallel-rockets` with your number of cores. No GPU is needed: the network is small, and
  most of the time goes into the physics. Keep the computer awake for the whole run, as sleep
  pauses training.
- **On GitHub Codespaces**, a cloud machine with up to 32 cores: see
  [Training on GitHub Codespaces](docs/training-on-codespaces.md).

Use the same `--parallel-rockets` for every level and seed of an experiment. It sets how many
steps go into each network update, so runs with different values can't be compared.

### Steps

Each training run belongs to an experiment, named after what it tries (e.g.
`graded-touchdown-penalty`). Train the levels in order: level 0 starts from random weights, and
each higher level continues from the best network of the level below, with the same experiment
and seed.

1. Train level 0:

   ```powershell
   uv run python scripts/ppo/train_ppo.py --experiment my-experiment --level 0
   ```

2. When it finishes, watch its best network fly:

   ```powershell
   uv run python scripts/evaluation/watch_controller.py --network runs/training/my-experiment/seed0/level0/best_network.zip --level 0
   ```

3. Train levels 1 and 2:

   ```powershell
   uv run python scripts/ppo/train_ppo.py --experiment my-experiment --level 1
   uv run python scripts/ppo/train_ppo.py --experiment my-experiment --level 2
   ```

Other options: `--seed 1` trains another seed of the same experiment, and `--steps` changes the
number of training steps (5,000,000 by default). Ctrl+C stops a run early; it still saves
`final_network.zip` and the training curves.

### Watching progress

- **Console and `training.log`:** every 250,000 steps, the network flies the 50 validation
  starts without exploration noise. Each line shows the landing rate, the average reward, and
  how the flights ended. The best network so far is saved as `best_network.zip`.
- **`training_curves.png`**, in the level folder and redrawn after every validation: the average
  reward and the landing rate of the training and validation flights, and the actor, critic and
  total losses. The numbers are in `metrics.csv`, one row per round (every rocket flies
  `n_steps` steps, then the network is updated).
- **TensorBoard**, live: run `uv run tensorboard --logdir runs/training`, then open
  http://localhost:6006.

## Test a network

Once level 2 is trained, fly its best network on the test sets: 500 fixed starts per level,
never flown during training, and the same for every controller. It tests all three levels; add
`--level 2` to test only one.

```powershell
uv run python scripts/evaluation/evaluate_controller.py --network runs/training/my-experiment/seed0/level2/best_network.zip
uv run python scripts/evaluation/compare_controllers.py --level 2    # side by side with the other tested controllers
```

## Results

Results stay on your computer, in `runs/` (not committed):

```
runs/
  training/<experiment>/seed<N>/level<N>/   best_network.zip, final_network.zip, checkpoints/,
                                            train_config.json, env_config.json, rocket_spec.json,
                                            training.log, metrics.csv, training_curves.png,
                                            tensorboard/
  test/pd/                                  tested_on_level<N>.json and .png
  test/<experiment>/seed<N>/level<N>/       best_network_tested_on_level<N>.json and .png
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

Settings live in `configs/`: `rocket/` (the rocket's shape and engines), `env/` (gravity,
time step, landing rules, rewards, starting conditions), `pd/` (the PD autopilot's gains), and
`train/` (PPO training settings). `src/rocketlander/config_files.py` says where each file is.

## Development

```powershell
uv run pytest                # run the tests
uv run ruff check .          # lint
uv run ruff format .         # format
```

To activate `.venv/` in your shell instead of using `uv run`:

```powershell
.venv\Scripts\Activate.ps1   # PowerShell (Windows)
source .venv/bin/activate    # bash/zsh (Linux, macOS)
```
