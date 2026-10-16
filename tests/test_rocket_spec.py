import pytest

from rocketlander.rocket_spec import Body, EngineBlock, Nozzle, RocketSpec


def test_r100_loads_with_nozzles_in_command_order(rocket: RocketSpec) -> None:
    assert [n.name for n in rocket.nozzles] == ["main", "left", "right"]
    assert rocket.nozzle("main").max_thrust == 2000.0


def test_nozzle_direction_must_be_a_unit_vector() -> None:
    with pytest.raises(ValueError, match="unit vector"):
        Nozzle("main", position=(0.0, 0.0), direction=(0.0, 2.0), max_thrust=100.0)


def test_nozzle_names_must_be_unique(rocket: RocketSpec) -> None:
    main = rocket.nozzle("main")
    with pytest.raises(ValueError, match="unique"):
        RocketSpec(rocket.body, rocket.engine_block, nozzles=(main, main), legs=rocket.legs)


def test_engine_block_must_fit_within_the_body(rocket: RocketSpec) -> None:
    with pytest.raises(ValueError, match="fit within the body"):
        RocketSpec(rocket.body, EngineBlock(height=5.0, mass=20.0), rocket.nozzles, rocket.legs)


def test_masses_must_be_positive() -> None:
    with pytest.raises(ValueError, match="mass must be positive"):
        Body(height=3.0, width=0.3, mass=0.0)


def test_unknown_nozzle_name_is_reported(rocket: RocketSpec) -> None:
    with pytest.raises(KeyError, match="'top'"):
        rocket.nozzle("top")
