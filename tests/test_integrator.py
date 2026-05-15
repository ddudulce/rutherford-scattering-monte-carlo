"""
Tests for the numerical integrator used in the Rutherford scattering model.
"""

import numpy as np

from src.simulation.integrator import (
    create_initial_particle,
    default_simulation_parameters,
    simulate_single_scattering,
)


def test_create_initial_particle_uses_impact_parameter() -> None:
    parameters = default_simulation_parameters()
    impact_parameter = 2.5

    particle = create_initial_particle(
        impact_parameter=impact_parameter,
        parameters=parameters,
    )

    expected_position = np.array(
        [-parameters.initial_distance, impact_parameter],
        dtype=float,
    )

    expected_velocity = np.array(
        [parameters.initial_speed, 0.0],
        dtype=float,
    )

    np.testing.assert_allclose(particle.position, expected_position)
    np.testing.assert_allclose(particle.velocity, expected_velocity)


def test_energy_is_conserved_for_single_scattering() -> None:
    result = simulate_single_scattering(
        impact_parameter=2.0,
        store_trajectory=True,
    )

    assert result.relative_energy_error < 1.0e-8


def test_scattering_angle_decreases_when_impact_parameter_increases() -> None:
    result_small_b = simulate_single_scattering(
        impact_parameter=1.0,
        store_trajectory=False,
    )

    result_large_b = simulate_single_scattering(
        impact_parameter=5.0,
        store_trajectory=False,
    )

    assert abs(result_small_b.scattering_angle_sim) > abs(
        result_large_b.scattering_angle_sim
    )


def test_numerical_angle_is_close_to_theoretical_angle() -> None:
    result = simulate_single_scattering(
        impact_parameter=2.0,
        store_trajectory=False,
    )

    assert np.isclose(
        result.scattering_angle_sim,
        result.scattering_angle_theory,
        rtol=1.0e-2,
    )