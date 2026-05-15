"""
High-level experiment routines for the Rutherford Monte Carlo project.

This module connects the Monte Carlo simulation, statistical analysis and
file output utilities. It also includes the thermal extension used to study
how the mean energy and mean scattering angle change with temperature.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.statistics import summarize_scattering_results
from src.domain.models import SimulationParameters
from src.simulation.integrator import (
    default_simulation_parameters,
    simulate_single_scattering,
)
from src.simulation.monte_carlo import (
    run_monte_carlo_scattering,
    sample_impact_parameters,
)
from src.utils.io import save_results_csv, save_summary_json


def run_base_experiment(
    n_particles: int = 200,
    b_min: float = 0.5,
    b_max: float = 8.0,
    seed: int | None = 42,
    parameters: SimulationParameters | None = None,
    output_directory: str | Path = "data",
) -> tuple[list, dict[str, float]]:
    """
    Run the main Rutherford Monte Carlo experiment.

    This is the central simulation of the project. It samples random impact
    parameters and computes the corresponding scattering angles.
    """

    if parameters is None:
        parameters = default_simulation_parameters()

    output_directory = Path(output_directory)

    results = run_monte_carlo_scattering(
        n_particles=n_particles,
        b_min=b_min,
        b_max=b_max,
        distribution="area",
        signed=True,
        seed=seed,
        parameters=parameters,
        store_trajectories=False,
    )

    summary = summarize_scattering_results(results)

    save_results_csv(
        results=results,
        output_path=output_directory / "results.csv",
    )

    save_summary_json(
        summary=summary,
        output_path=output_directory / "summary.json",
    )

    return results, summary


def run_thermal_extension(
    temperatures: list[float] | np.ndarray,
    n_particles_per_temperature: int = 100,
    b_min: float = 0.5,
    b_max: float = 8.0,
    seed: int | None = 123,
    parameters: SimulationParameters | None = None,
    output_path: str | Path = "data/thermal_extension.csv",
) -> pd.DataFrame:
    """
    Run the thermal extension of the Rutherford scattering simulation.

    In the dimensionless model, we set k_B = 1 and m = 1. For a two-dimensional
    thermal velocity distribution, the speed is sampled from a Rayleigh
    distribution with scale sqrt(T). This produces an ensemble whose mean
    kinetic energy grows linearly with temperature.

    For each temperature, the function reports:
        - mean initial energy,
        - standard deviation of initial energy,
        - mean absolute scattering angle,
        - standard deviation of the absolute scattering angle.
    """

    if parameters is None:
        parameters = default_simulation_parameters()

    temperatures = np.asarray(temperatures, dtype=float)

    if np.any(temperatures <= 0):
        raise ValueError("All temperatures must be positive.")

    rng = np.random.default_rng(seed)

    rows = []

    for temperature in temperatures:
        impact_parameters = sample_impact_parameters(
            n_particles=n_particles_per_temperature,
            b_min=b_min,
            b_max=b_max,
            distribution="area",
            signed=True,
            seed=int(rng.integers(0, 1_000_000_000)),
        )

        speeds = rng.rayleigh(
            scale=np.sqrt(temperature),
            size=n_particles_per_temperature,
        )

        initial_energies = []
        absolute_angles = []

        for impact_parameter, speed in zip(impact_parameters, speeds):
            thermal_parameters = replace(
                parameters,
                initial_speed=float(speed),
            )

            result = simulate_single_scattering(
                impact_parameter=float(impact_parameter),
                parameters=thermal_parameters,
                store_trajectory=False,
            )

            initial_energies.append(result.initial_energy)
            absolute_angles.append(abs(result.scattering_angle_sim))

        initial_energies_array = np.array(initial_energies, dtype=float)
        absolute_angles_array = np.array(absolute_angles, dtype=float)

        rows.append(
            {
                "temperature": float(temperature),
                "mean_initial_energy": float(np.mean(initial_energies_array)),
                "std_initial_energy": float(np.std(initial_energies_array, ddof=1)),
                "mean_absolute_scattering_angle": float(
                    np.mean(absolute_angles_array)
                ),
                "std_absolute_scattering_angle": float(
                    np.std(absolute_angles_array, ddof=1)
                ),
                "n_particles": int(n_particles_per_temperature),
            }
        )

    dataframe = pd.DataFrame(rows)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)

    return dataframe