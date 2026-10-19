import pytest

from rocketlander.rocket.rocket_spec import Body, EngineBlock, Leg, Nozzle, RocketSpec


def test_r100_loads_with_nozzles_in_command_order(rocket_spec: RocketSpec) -> None:
    assert [n.name for n in rocket_spec.nozzles] == ["main", "left", "right"]
    assert rocket_spec.get_nozzle("main").max_thrust == 2000.0


def test_nozzle_direction_must_be_a_unit_vector() -> None:
    with pytest.raises(ValueError, match="unit vector"):
        Nozzle("main", position=(0.0, 0.0), direction=(0.0, 2.0), max_thrust=100.0)


def test_nozzle_names_must_be_unique(rocket_spec: RocketSpec) -> None:
    main = rocket_spec.get_nozzle("main")
    with pytest.raises(ValueError, match="unique"):
        RocketSpec(
            rocket_spec.body, rocket_spec.engine_block, nozzles=(main, main), legs=rocket_spec.legs
        )


def test_engine_block_must_fit_within_the_body(rocket_spec: RocketSpec) -> None:
    with pytest.raises(ValueError, match="fit within the body"):
        RocketSpec(
            rocket_spec.body,
            EngineBlock(height=5.0, mass=20.0),
            rocket_spec.nozzles,
            rocket_spec.legs,
        )


def test_legs_must_reach_below_the_base(rocket_spec: RocketSpec) -> None:
    legs_at_base_level = (Leg(top=(-0.15, 0.5), tip=(-0.6, 0.0)),)
    with pytest.raises(ValueError, match="engine clears the ground"):
        RocketSpec(
            rocket_spec.body, rocket_spec.engine_block, rocket_spec.nozzles, legs=legs_at_base_level
        )


def test_masses_must_be_positive() -> None:
    with pytest.raises(ValueError, match="mass must be positive"):
        Body(height=3.0, width=0.3, mass=0.0)


def test_unknown_nozzle_name_is_reported(rocket_spec: RocketSpec) -> None:
    with pytest.raises(KeyError, match="'top'"):
        rocket_spec.get_nozzle("top")
