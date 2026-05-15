"""
Theoretical Rutherford scattering relations.

This module contains analytical expressions used to validate the numerical
simulation.
"""

from __future__ import annotations

import numpy as np


def rutherford_scattering_angle(
    impact_parameter: float,
    kinetic_energy: float,
    coupling: float,
    signed: bool = True,
) -> float:
    """
    Compute the classical Rutherford scattering angle.

    The classical relation is

        theta = 2 arctan( coupling / (2 E |b|) ),

    where:
        b is the impact parameter,
        E is the initial kinetic energy,
        coupling = k q_1 q_2.

    If signed=True, the sign of the impact parameter is assigned to the angle.
    """

    if kinetic_energy <= 0:
        raise ValueError("The kinetic energy must be positive.")

    if impact_parameter == 0:
        angle = np.pi
    else:
        angle = 2.0 * np.arctan(
            coupling / (2.0 * kinetic_energy * abs(impact_parameter))
        )

    if signed:
        return float(np.sign(impact_parameter) * angle)

    return float(angle)


def scattering_angle_from_velocity(velocity: np.ndarray) -> float:
    """
    Compute the numerical scattering angle from the final velocity vector.

    The incident beam is assumed to move initially along the positive x-axis.
    Therefore, the scattering angle is measured with respect to that direction.
    """

    vx = float(velocity[0])
    vy = float(velocity[1])

    return float(np.arctan2(vy, vx))


def relative_error(numerical_value: float, theoretical_value: float) -> float:
    """
    Compute the relative error between a numerical and a theoretical value.
    """

    if theoretical_value == 0.0:
        return abs(numerical_value - theoretical_value)

    return abs((numerical_value - theoretical_value) / theoretical_value)