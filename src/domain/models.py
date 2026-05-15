"""
Data models for the Rutherford scattering simulation.

These classes store the physical and numerical information used by the
simulation. They do not perform calculations; they only organize data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


ArrayLike = np.ndarray


@dataclass(frozen=True)
class Particle:
    """
    Incident classical charged particle.
    """

    mass: float
    charge: float
    position: ArrayLike
    velocity: ArrayLike


@dataclass(frozen=True)
class Nucleus:
    """
    Fixed scattering center.

    The nucleus is assumed to be much heavier than the incident particle,
    so its position remains fixed during the simulation.
    """

    charge: float
    position: ArrayLike


@dataclass(frozen=True)
class SimulationParameters:
    """
    Numerical parameters for one scattering simulation.
    """

    initial_distance: float
    initial_speed: float
    time_step: float
    max_steps: int
    softening: float


@dataclass
class Trajectory:
    """
    Numerical trajectory of one incident particle.
    """

    time: ArrayLike
    position: ArrayLike
    velocity: ArrayLike
    energy: ArrayLike


@dataclass
class ScatteringResult:
    """
    Summary of a single Rutherford scattering event.
    """

    impact_parameter: float
    scattering_angle_sim: float
    scattering_angle_theory: Optional[float]
    initial_energy: float
    final_energy: float
    relative_energy_error: float
    final_position: ArrayLike
    final_velocity: ArrayLike
    trajectory: Optional[Trajectory] = None