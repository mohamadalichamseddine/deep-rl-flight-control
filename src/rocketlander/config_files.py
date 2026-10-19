"""Where the project's config files are."""

from pathlib import Path

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
ROCKET_SPEC_FILE = CONFIGS / "rocket" / "r100.yaml"
ENV_CONFIG_FILE = CONFIGS / "env" / "default.yaml"
TRAIN_CONFIG_FILE = CONFIGS / "train" / "ppo.yaml"
PD_GAINS_FILE = CONFIGS / "pd" / "gains.yaml"
