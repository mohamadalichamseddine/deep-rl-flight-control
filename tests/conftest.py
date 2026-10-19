import pytest

from rocketlander.config_files import ENV_CONFIG_FILE, PD_GAINS_FILE, ROCKET_SPEC_FILE
from rocketlander.environment.env_config import EnvConfig, build_env_config
from rocketlander.pd.pd_controller import PDGains, build_pd_gains
from rocketlander.rocket.rocket_spec import RocketSpec, build_rocket_spec


@pytest.fixture(scope="session")
def rocket_spec() -> RocketSpec:
    return build_rocket_spec(ROCKET_SPEC_FILE)


@pytest.fixture(scope="session")
def env_config() -> EnvConfig:
    return build_env_config(ENV_CONFIG_FILE)


@pytest.fixture(scope="session")
def pd_gains() -> PDGains:
    return build_pd_gains(PD_GAINS_FILE)
