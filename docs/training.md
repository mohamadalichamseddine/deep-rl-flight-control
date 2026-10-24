# Training a network

How to train a network, follow its progress, and test it. The [README](../README.md) has the
settings and the results.

## Where to train

- **On your computer.** Training flies 8 rockets at once (`parallel_rockets` in
  `configs/train/ppo.yaml`), ideally one per CPU core; with fewer cores, pass
  `--parallel-rockets` with your number of cores. No GPU is needed: the network is small enough
  for the CPU. Training needs about 4 GB of free memory (each rocket's process loads PyTorch),
  so close large apps first. Keep the computer awake for the whole run, as sleep pauses
  training.
- **On GitHub Codespaces**, a cloud machine, so your computer stays free: see
  [Training on GitHub Codespaces](training-on-codespaces.md).

Use the same `--parallel-rockets` for every level and seed of an experiment. It sets how many
steps go into each network update, so runs with different values can't be compared.

## Steps

Each training run belongs to an experiment, named after what it tries (e.g.
`graded-touchdown-penalty`). Train the levels in order: level 0 starts from random weights, and
each higher level continues from the best network of the level below, with the same experiment
and seed. Every level starts with the same exploration noise (`log_std_init`), so it tries new
moves on its harder starts.

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

To give a finished run more steps, continue it from its `final_network.zip`. It keeps the
settings saved in its folder, and its step count, `metrics.csv` and curves go on where they
stopped:

```powershell
uv run python scripts/ppo/train_ppo.py --experiment my-experiment --level 0 --continue --steps 5000000
```

## Watching progress

- **Console and `training.log`:** every 250,000 steps, the network flies the 50 validation
  starts without exploration noise. Each line shows the landing rate, the average reward, and
  how the flights ended. The best network so far is saved as `best_network.zip`.
- **`training_curves.png`**, in the level folder and redrawn after every validation: the average
  reward and the landing rate of the training and validation flights, and the actor, critic and
  total losses. The numbers are in `metrics.csv`, one row per round (every rocket flies
  `n_steps` steps, then the network is updated).
- **TensorBoard**, live: run `uv run tensorboard --logdir runs/training`, then open
  http://localhost:6006.

## Testing a network

Once level 2 is trained, fly its best network on the test sets: 500 fixed starts per level,
never flown during training, and the same for every controller. It tests all three levels; add
`--level 2` to test only one.

```powershell
uv run python scripts/evaluation/evaluate_controller.py --network runs/training/my-experiment/seed0/level2/best_network.zip
uv run python scripts/evaluation/evaluate_controller.py              # the PD autopilot
uv run python scripts/evaluation/compare_controllers.py --level 2    # side by side with the other tested controllers
```

The console lists the test episodes that failed. Re-fly one in the live window with
`watch_controller.py --network ... --level 2 --test-episode N`.

## Output files

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
