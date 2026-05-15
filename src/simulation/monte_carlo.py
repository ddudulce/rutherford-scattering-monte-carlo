"""
Monte Carlo routines for the Rutherford scattering simulation.

This module generates random impact parameters and uses the single-particle
integrator to simulate an ensemble of scattering events.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from src.domain.models import ScatteringResult, SimulationParameters
from src.simulation.integrator import (
    default_simulation_parameters,
    simulate_single_scattering,
)


ImpactDistribution = Literal["uniform", "area"]


def sample_impact_parameters(
    n_particles: int,
    b_min: float,
    b_max: float,
    distribution: ImpactDistribution = "area",
    signed: bool = True,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate random impact parameters for the Monte Carlo simulation.

    Parameters
    ----------
    n_particles:
        Number of particles to simulate.
    b_min:
        Minimum absolute impact parameter.
    b_max:
        Maximum absolute impact parameter.
    distribution:
        Sampling rule for the impact parameter.

        "uniform":
            Samples |b| uniformly between b_min and b_max.

        "area":
            Samples |b| uniformly in transverse area, using
            b = sqrt(u (b_max^2 - b_min^2) + b_min^2).

    signed:
        If True, randomly assigns positive and negative signs to b.
        This allows trajectories above and below the scattering center.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Array of sampled impact parameters.
    """

    if n_particles <= 0:
        raise ValueError("n_particles must be positive.")

    if b_min < 0:
        raise ValueError("b_min must be non-negative.")

    if b_max <= b_min:
        raise ValueError("b_max must be greater than b_min.")

    rng = np.random.default_rng(seed)

    if distribution == "uniform":
        magnitudes = rng.uniform(b_min, b_max, size=n_particles)

    elif distribution == "area":
        u = rng.uniform(0.0, 1.0, size=n_particles)
        magnitudes = np.sqrt(
            u * (b_max**2 - b_min**2) + b_min**2
        )

    else:
        raise ValueError("distribution must be either 'uniform' or 'area'.")

    if signed:
        signs = rng.choice([-1.0, 1.0], size=n_particles)
        return signs * magnitudes

    return magnitudes


def run_monte_carlo_scattering(
    n_particles: int,
    b_min: float,
    b_max: float,
    distribution: ImpactDistribution = "area",
    signed: bool = True,
    seed: int | None = None,
    parameters: SimulationParameters | None = None,
    store_trajectories: bool = False,
) -> list[ScatteringResult]:
    """
    Run a Monte Carlo Rutherford scattering experiment.

    Each particle receives a random impact parameter and is then propagated
    using the single-particle integrator.
    """

    if parameters is None:
        parameters = default_simulation_parameters()

    impact_parameters = sample_impact_parameters(
        n_particles=n_particles,
        b_min=b_min,
        b_max=b_max,
        distribution=distribution,
        signed=signed,
        seed=seed,
    )

    results: list[ScatteringResult] = []

    for impact_parameter in impact_parameters:
        result = simulate_single_scattering(
            impact_parameter=float(impact_parameter),
            parameters=parameters,
            store_trajectory=store_trajectories,
        )

        results.append(result)

    return results


def extract_scattering_arrays(
    results: list[ScatteringResult],
) -> dict[str, np.ndarray]:
    """
    Convert a list of scattering results into numerical arrays.

    This is useful for statistics, plotting and saving results.
    """

    impact_parameters = np.array(
        [result.impact_parameter for result in results],
        dtype=float,
    )

    scattering_angle_sim = np.array(
        [result.scattering_angle_sim for result in results],
        dtype=float,
    )

    scattering_angle_theory = np.array(
        [result.scattering_angle_theory for result in results],
        dtype=float,
    )

    initial_energy = np.array(
        [result.initial_energy for result in results],
        dtype=float,
    )

    final_energy = np.array(
        [result.final_energy for result in results],
        dtype=float,
    )

    relative_energy_error = np.array(
        [result.relative_energy_error for result in results],
        dtype=float,
    )

    return {
        "impact_parameter": impact_parameters,
        "scattering_angle_sim": scattering_angle_sim,
        "scattering_angle_theory": scattering_angle_theory,
        "initial_energy": initial_energy,
        "final_energy": final_energy,
        "relative_energy_error": relative_energy_error,
    }