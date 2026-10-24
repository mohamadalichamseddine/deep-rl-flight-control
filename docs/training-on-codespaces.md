# Training on GitHub Codespaces

A [GitHub Codespace](https://docs.github.com/en/codespaces) is a cloud machine, set up from
`.devcontainer/devcontainer.json` (Python 3.13, uv, `uv sync`). A personal account may only be
offered 2- and 4-core machines; to list them:

```powershell
gh api "repos/mohamadalichamseddine/deep-rl-flight-control/codespaces/machines?ref=main" --jq ".machines[].display_name"
```

Training flies 8 rockets at once. On 4 cores they share the cores: the results are the same as
on 8 cores, only about twice as slow. The free plan includes 120 core-hours per month: 30 hours
on 4 cores.

The codespace is built from a branch on GitHub, so push your changes before creating it. The
training steps are the same as on your computer (see [Training a network](training.md)).

## 1. Create a codespace

Create it on a branch, with 4 cores (or 8 if offered), either:

- on github.com: **Code → Codespaces → ⋯ → New with options**, or
- from your terminal. `create` asks which machine type to use. The first time, run
  `gh auth refresh -h github.com -s codespace`.

  ```powershell
  gh codespace create -R mohamadalichamseddine/deep-rl-flight-control -b main --idle-timeout 60m
  gh codespace ssh
  ```

## 2. Start training

In the codespace's terminal. `nohup` keeps training running if you disconnect:

```bash
cd /workspaces/deep-rl-flight-control
mkdir -p runs
nohup uv run python scripts/ppo/train_ppo.py --experiment my-experiment --level 0 > runs/nohup.out 2>&1 &
tail -f runs/nohup.out    # watch it; Ctrl+C stops watching, not training
```

To train levels 0, 1 and 2 one after the other (each starts only if the one before
succeeded):

```bash
nohup sh -c "uv run python scripts/ppo/train_ppo.py --experiment my-experiment --level 0 &&
             uv run python scripts/ppo/train_ppo.py --experiment my-experiment --level 1 &&
             uv run python scripts/ppo/train_ppo.py --experiment my-experiment --level 2" > runs/nohup.out 2>&1 &
```

A bigger machine is only faster with more rockets, e.g. 16 cores with `--parallel-rockets 16`.
That changes the batch, though: each round collects parallel rockets × `n_steps` (1,024)
steps, so 16 rockets make half as many network updates for the same number of steps as 8.
Use the same value for every level and seed of an experiment.

A codespace stops when no terminal has shown input or output for the idle timeout (at most 240
minutes), and after 12 hours in any case. Stopping ends training, but keeps the files saved so
far. Training under `nohup` writes to a file, not a terminal, so it doesn't count as activity:
keep a terminal showing `tail -f runs/nohup.out`, which prints a line at every validation. The
idle timeout also keeps using core-hours after training ends, so keep it short (60 minutes).

## 3. Copy the results to your computer

They exist only in the codespace, because `runs/` is not committed:

```powershell
mkdir -Force runs/training
gh codespace cp -r remote:/workspaces/deep-rl-flight-control/runs/training/my-experiment runs/training/
```

Or right-click the folder in the codespace's file explorer → **Download**.

## 4. Stop and delete the codespace

Stop it when you're done (`gh codespace stop`), as it uses core-hours while it runs. Delete it
once the results are copied (`gh codespace delete`): a stopped codespace still counts toward the
15 GB of free storage.
