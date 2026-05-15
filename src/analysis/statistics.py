"""
Statistical analysis tools for the Rutherford Monte Carlo simulation.

This module computes mean values, standard deviations and basic summaries
from the scattering results.
"""

from __future__ import annotations

import numpy as np

from src.domain.models import ScatteringResult


def mean_value(values: np.ndarray) -> float:
    """
    Compute the mean value of a numerical array.
    """

    if values.size == 0:
        raise ValueError("The input array cannot be empty.")

    return float(np.mean(values))


def standard_deviation(values: np.ndarray) -> float:
    """
    Compute the sample standard deviation of a numerical array.
    """

    if values.size < 2:
        return 0.0

    return float(np.std(values, ddof=1))


def root_mean_square(values: np.ndarray) -> float:
    """
    Compute the root mean square of a numerical array.
    """

    if values.size == 0:
        raise ValueError("The input array cannot be empty.")

    return float(np.sqrt(np.mean(values**2)))


def angular_statistics(
    scattering_angles: np.ndarray,
) -> dict[str, float]:
    """
    Compute basic statistics for the scattering angles.

    The angles may be signed, so this function reports both signed and
    absolute-angle quantities.
    """

    if scattering_angles.size == 0:
        raise ValueError("The scattering angle array cannot be empty.")

    absolute_angles = np.abs(scattering_angles)

    return {
        "mean_angle": mean_value(scattering_angles),
        "mean_absolute_angle": mean_value(absolute_angles),
        "std_angle": standard_deviation(scattering_angles),
        "std_absolute_angle": standard_deviation(absolute_angles),
        "rms_angle": root_mean_square(scattering_angles),
        "min_angle": float(np.min(scattering_angles)),
        "max_angle": float(np.max(scattering_angles)),
    }


def energy_statistics(
    initial_energy: np.ndarray,
    final_energy: np.ndarray,
    relative_energy_error: np.ndarray,
) -> dict[str, float]:
    """
    Compute statistics related to energy conservation.
    """

    if initial_energy.size == 0:
        raise ValueError("The energy arrays cannot be empty.")

    return {
        "mean_initial_energy": mean_value(initial_energy),
        "mean_final_energy": mean_value(final_energy),
        "mean_relative_energy_error": mean_value(relative_energy_error),
        "max_relative_energy_error": float(np.max(relative_energy_error)),
    }


def summarize_scattering_results(
    results: list[ScatteringResult],
) -> dict[str, float]:
    """
    Build a compact statistical summary from a list of scattering results.
    """

    if len(results) == 0:
        raise ValueError("The results list cannot be empty.")

    scattering_angles = np.array(
        [result.scattering_angle_sim for result in results],
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

    summary = {
        "n_particles": float(len(results)),
    }

    summary.update(angular_statistics(scattering_angles))
    summary.update(
        energy_statistics(
            initial_energy=initial_energy,
            final_energy=final_energy,
            relative_energy_error=relative_energy_error,
        )
    )

    return summary


def print_summary(summary: dict[str, float]) -> None:
    """
    Print a readable summary of the Monte Carlo statistics.
    """

    print("Monte Carlo summary")
    print("-" * 40)

    for key, value in summary.items():
        if key == "n_particles":
            print(f"{key}: {int(value)}")
        else:
            print(f"{key}: {value:.6e}")