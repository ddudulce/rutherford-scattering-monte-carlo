"""
Core physics functions for the classical Rutherford scattering model.

This module contains only pure physical calculations: Coulomb force,
acceleration, kinetic energy, potential energy and total mechanical energy.
"""

from __future__ import annotations

import numpy as np


def coulomb_force(
    position: np.ndarray,
    nucleus_position: np.ndarray,
    coupling: float,
    softening: float = 0.0,
) -> np.ndarray:
    """
    Compute the repulsive Coulomb force on the incident particle.

    The vector form of the repulsive Coulomb force is

        F = coupling * r / |r|^3,

    where r is the vector from the nucleus to the incident particle.
    """

    r_vector = position - nucleus_position
    r_squared = float(np.dot(r_vector, r_vector)) + softening**2
    r_cubed = r_squared ** 1.5

    if r_cubed == 0.0:
        raise ValueError("The particle is exactly at the nucleus position.")

    return coupling * r_vector / r_cubed


def acceleration(
    position: np.ndarray,
    nucleus_position: np.ndarray,
    particle_mass: float,
    coupling: float,
    softening: float = 0.0,
) -> np.ndarray:
    """
    Compute the acceleration produced by the Coulomb force.
    """

    force = coulomb_force(
        position=position,
        nucleus_position=nucleus_position,
        coupling=coupling,
        softening=softening,
    )

    return force / particle_mass


def kinetic_energy(particle_mass: float, velocity: np.ndarray) -> float:
    """
    Compute the kinetic energy of the incident particle.

        K = (1/2) m v^2.
    """

    return 0.5 * particle_mass * float(np.dot(velocity, velocity))


def potential_energy(
    position: np.ndarray,
    nucleus_position: np.ndarray,
    coupling: float,
    softening: float = 0.0,
) -> float:
    """
    Compute the repulsive Coulomb potential energy.

        U = coupling / r.
    """

    r_vector = position - nucleus_position
    r = float(np.sqrt(np.dot(r_vector, r_vector) + softening**2))

    if r == 0.0:
        raise ValueError("The particle is exactly at the nucleus position.")

    return coupling / r


def total_energy(
    position: np.ndarray,
    velocity: np.ndarray,
    nucleus_position: np.ndarray,
    particle_mass: float,
    coupling: float,
    softening: float = 0.0,
) -> float:
    """
    Compute the total mechanical energy.

        E = K + U.
    """

    kinetic = kinetic_energy(
        particle_mass=particle_mass,
        velocity=velocity,
    )

    potential = potential_energy(
        position=position,
        nucleus_position=nucleus_position,
        coupling=coupling,
        softening=softening,
    )

    return kinetic + potential