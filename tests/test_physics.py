"""
Tests for the basic physics functions used in the Rutherford scattering model.
"""

import numpy as np

from src.domain.physics import (
    acceleration,
    coulomb_force,
    kinetic_energy,
    potential_energy,
    total_energy,
)


def test_coulomb_force_points_radially_outward() -> None:
    position = np.array([2.0, 0.0])
    nucleus_position = np.array([0.0, 0.0])
    coupling = 1.0

    force = coulomb_force(
        position=position,
        nucleus_position=nucleus_position,
        coupling=coupling,
    )

    expected_force = np.array([0.25, 0.0])

    np.testing.assert_allclose(force, expected_force, rtol=1e-12)


def test_coulomb_force_follows_inverse_square_law() -> None:
    nucleus_position = np.array([0.0, 0.0])
    coupling = 1.0

    force_r1 = coulomb_force(
        position=np.array([1.0, 0.0]),
        nucleus_position=nucleus_position,
        coupling=coupling,
    )

    force_r2 = coulomb_force(
        position=np.array([2.0, 0.0]),
        nucleus_position=nucleus_position,
        coupling=coupling,
    )

    magnitude_r1 = np.linalg.norm(force_r1)
    magnitude_r2 = np.linalg.norm(force_r2)

    assert np.isclose(magnitude_r1 / magnitude_r2, 4.0)


def test_acceleration_is_force_divided_by_mass() -> None:
    position = np.array([2.0, 0.0])
    nucleus_position = np.array([0.0, 0.0])
    particle_mass = 2.0
    coupling = 1.0

    acc = acceleration(
        position=position,
        nucleus_position=nucleus_position,
        particle_mass=particle_mass,
        coupling=coupling,
    )

    expected_acceleration = np.array([0.125, 0.0])

    np.testing.assert_allclose(acc, expected_acceleration, rtol=1e-12)


def test_kinetic_energy() -> None:
    particle_mass = 2.0
    velocity = np.array([3.0, 4.0])

    energy = kinetic_energy(
        particle_mass=particle_mass,
        velocity=velocity,
    )

    assert np.isclose(energy, 25.0)


def test_potential_energy() -> None:
    position = np.array([2.0, 0.0])
    nucleus_position = np.array([0.0, 0.0])
    coupling = 1.0

    energy = potential_energy(
        position=position,
        nucleus_position=nucleus_position,
        coupling=coupling,
    )

    assert np.isclose(energy, 0.5)


def test_total_energy_is_kinetic_plus_potential() -> None:
    position = np.array([2.0, 0.0])
    velocity = np.array([3.0, 4.0])
    nucleus_position = np.array([0.0, 0.0])

    energy = total_energy(
        position=position,
        velocity=velocity,
        nucleus_position=nucleus_position,
        particle_mass=2.0,
        coupling=1.0,
    )

    expected_energy = 25.0 + 0.5

    assert np.isclose(energy, expected_energy)