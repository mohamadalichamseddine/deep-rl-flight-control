# Autonomous Rocket Landing
Autonomous rocket landing using deep reinforcement learning (PPO), neural networks, and physics simulation

- [Project definition](docs/project.md): problem, success criteria, and scope
- [Implementation plan](docs/plan.md): decisions, architecture, and phases

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
