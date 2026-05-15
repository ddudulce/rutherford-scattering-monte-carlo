"""
Statistical plots for the Rutherford Monte Carlo simulation.

This module generates the main figures used in the report:
    - scattering angle vs impact parameter,
    - angular histogram,
    - energy conservation plot,
    - thermal extension plots.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.domain.models import ScatteringResult


def ensure_figure_directory(output_path: str | Path) -> Path:
    """
    Create the parent directory for a figure if it does not already exist.
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    return path


def plot_scattering_angle_vs_impact_parameter(
    results: list[ScatteringResult],
    output_path: str | Path = "figures/scattering_angle_vs_impact_parameter.png",
) -> Path:
    """
    Plot the absolute scattering angle as a function of the absolute impact
    parameter.

    Since positive and negative impact parameters only indicate whether the
    particle passes above or below the nucleus, the physically relevant
    comparison with Rutherford's formula is made using |b| and |theta|.
    """

    if len(results) == 0:
        raise ValueError("The results list cannot be empty.")

    output_path = ensure_figure_directory(output_path)

    impact_parameters = np.array(
        [abs(result.impact_parameter) for result in results],
        dtype=float,
    )

    scattering_angle_sim = np.array(
        [abs(result.scattering_angle_sim) for result in results],
        dtype=float,
    )

    scattering_angle_theory = np.array(
        [abs(result.scattering_angle_theory) for result in results],
        dtype=float,
    )

    sorted_indices = np.argsort(impact_parameters)

    plt.figure(figsize=(8, 5))
    plt.scatter(
        impact_parameters,
        scattering_angle_sim,
        s=18,
        alpha=0.7,
        label="Numerical simulation",
    )
    plt.plot(
        impact_parameters[sorted_indices],
        scattering_angle_theory[sorted_indices],
        linewidth=1.8,
        label="Theoretical Rutherford angle",
    )

    plt.xlabel("Absolute impact parameter |b|")
    plt.ylabel("Absolute scattering angle |θ| [rad]")
    plt.title("Absolute scattering angle as a function of absolute impact parameter")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path




def plot_scattering_angle_histogram(
    results: list[ScatteringResult],
    output_path: str | Path = "figures/scattering_angle_histogram.png",
    bins: int = 30,
) -> Path:
    """
    Plot the histogram of absolute scattering angles.
    """

    if len(results) == 0:
        raise ValueError("The results list cannot be empty.")

    output_path = ensure_figure_directory(output_path)

    absolute_angles = np.array(
        [abs(result.scattering_angle_sim) for result in results],
        dtype=float,
    )

    plt.figure(figsize=(8, 5))
    plt.hist(absolute_angles, bins=bins, density=True, alpha=0.75)

    plt.xlabel("|θ| [rad]")
    plt.ylabel("Probability density")
    plt.title("Distribution of absolute scattering angles")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def plot_energy_conservation(
    result: ScatteringResult,
    output_path: str | Path = "figures/energy_conservation.png",
) -> Path:
    """
    Plot the relative variation of the total mechanical energy as a function
    of time for one trajectory.

    This is clearer than plotting E(t) directly, because the total energy is
    almost constant and only changes by very small numerical errors.
    """

    if result.trajectory is None:
        raise ValueError("The result must include a stored trajectory.")

    output_path = ensure_figure_directory(output_path)

    time = result.trajectory.time
    energy = result.trajectory.energy

    initial_energy = energy[0]

    if initial_energy == 0.0:
        relative_energy_variation = energy - initial_energy
    else:
        relative_energy_variation = (energy - initial_energy) / initial_energy

    plt.figure(figsize=(8, 5))
    plt.plot(time, relative_energy_variation, linewidth=1.8)

    plt.xlabel("Time")
    plt.ylabel("(E(t) - E(0)) / E(0)")
    plt.title("Relative energy conservation during one scattering event")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def plot_relative_energy_error(
    results: list[ScatteringResult],
    output_path: str | Path = "figures/relative_energy_error.png",
) -> Path:
    """
    Plot the relative energy error for all simulated particles.
    """

    if len(results) == 0:
        raise ValueError("The results list cannot be empty.")

    output_path = ensure_figure_directory(output_path)

    relative_errors = np.array(
        [result.relative_energy_error for result in results],
        dtype=float,
    )

    particle_ids = np.arange(1, len(results) + 1)

    plt.figure(figsize=(8, 5))
    plt.scatter(particle_ids, relative_errors, s=18, alpha=0.7)

    plt.xlabel("Particle index")
    plt.ylabel("Relative energy error")
    plt.title("Relative energy error for the Monte Carlo ensemble")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def plot_mean_energy_vs_temperature(
    thermal_dataframe: pd.DataFrame,
    output_path: str | Path = "figures/mean_energy_vs_temperature.png",
) -> Path:
    """
    Plot the thermal extension result <E> vs T.

    Error bars correspond to the standard error of the mean, SEM = s/sqrt(N).
    """

    required_columns = {
        "temperature",
        "mean_initial_energy",
        "std_initial_energy",
        "n_particles",
    }

    if not required_columns.issubset(thermal_dataframe.columns):
        raise ValueError(
            "The thermal dataframe does not contain the required columns."
        )

    output_path = ensure_figure_directory(output_path)

    energy_sem = (
        thermal_dataframe["std_initial_energy"]
        / np.sqrt(thermal_dataframe["n_particles"])
    )

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        thermal_dataframe["temperature"],
        thermal_dataframe["mean_initial_energy"],
        yerr=energy_sem,
        marker="o",
        capsize=4,
        linewidth=1.5,
    )

    plt.xlabel("Temperature T")
    plt.ylabel("Mean initial energy <E>")
    plt.title("Thermal extension: mean energy vs temperature")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def plot_mean_scattering_angle_vs_temperature(
    thermal_dataframe: pd.DataFrame,
    output_path: str | Path = "figures/mean_scattering_angle_vs_temperature.png",
) -> Path:
    """
    Plot the thermal extension result <|θ|> vs T.

    Error bars correspond to the standard error of the mean, SEM = s/sqrt(N).
    """

    required_columns = {
        "temperature",
        "mean_absolute_scattering_angle",
        "std_absolute_scattering_angle",
        "n_particles",
    }

    if not required_columns.issubset(thermal_dataframe.columns):
        raise ValueError(
            "The thermal dataframe does not contain the required columns."
        )

    output_path = ensure_figure_directory(output_path)

    angle_sem = (
        thermal_dataframe["std_absolute_scattering_angle"]
        / np.sqrt(thermal_dataframe["n_particles"])
    )

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        thermal_dataframe["temperature"],
        thermal_dataframe["mean_absolute_scattering_angle"],
        yerr=angle_sem,
        marker="o",
        capsize=4,
        linewidth=1.5,
    )

    plt.xlabel("Temperature T")
    plt.ylabel("Mean absolute scattering angle <|θ|> [rad]")
    plt.title("Thermal extension: mean scattering angle vs temperature")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path