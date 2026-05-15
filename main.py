"""
Main execution script for the Rutherford scattering Monte Carlo project.

This script runs:
    1. The base Monte Carlo Rutherford scattering experiment.
    2. The thermal extension.
    3. The generation of all main figures.
"""

from __future__ import annotations

import numpy as np

from src.analysis.statistics import print_summary
from src.simulation.experiment import run_base_experiment, run_thermal_extension
from src.simulation.integrator import simulate_single_scattering
from src.visualization.plots import (
    plot_energy_conservation,
    plot_mean_energy_vs_temperature,
    plot_mean_scattering_angle_vs_temperature,
    plot_relative_energy_error,
    plot_scattering_angle_histogram,
    plot_scattering_angle_vs_impact_parameter,
)
from src.visualization.trajectories import (
    generate_sample_trajectories,
    plot_final_configuration,
    plot_initial_configuration,
    plot_trajectories,
)


def main() -> None:
    """
    Run the full numerical experiment and generate output files.
    """

    print("Running Rutherford Monte Carlo simulation...")
    print("-" * 50)

    results, summary = run_base_experiment(
        n_particles=300,
        b_min=0.5,
        b_max=8.0,
        seed=42,
        output_directory="data",
    )

    print_summary(summary)

    print("\nGenerating trajectory figures...")
    sample_trajectory_results = generate_sample_trajectories(
        impact_parameters=np.array([-5.0, -2.0, -1.0, 1.0, 2.0, 5.0])
    )

    plot_initial_configuration(
        results=sample_trajectory_results,
        output_path="figures/initial_configuration.png",
    )

    plot_final_configuration(
        results=sample_trajectory_results,
        output_path="figures/final_configuration.png",
    )

    plot_trajectories(
        results=sample_trajectory_results,
        output_path="figures/sample_trajectories.png",
    )

    print("Generating statistical figures...")

    plot_scattering_angle_vs_impact_parameter(
        results=results,
        output_path="figures/scattering_angle_vs_impact_parameter.png",
    )

    plot_scattering_angle_histogram(
        results=results,
        output_path="figures/scattering_angle_histogram.png",
        bins=30,
    )

    energy_result = simulate_single_scattering(
        impact_parameter=2.0,
        store_trajectory=True,
    )

    plot_energy_conservation(
        result=energy_result,
        output_path="figures/energy_conservation.png",
    )

    plot_relative_energy_error(
        results=results,
        output_path="figures/relative_energy_error.png",
    )

    print("Running thermal extension...")

    thermal_dataframe = run_thermal_extension(
        temperatures=np.array([0.5, 1.0, 1.5, 2.0, 3.0]),
        n_particles_per_temperature=300,
        b_min=0.5,
        b_max=8.0,
        seed=123,
        output_path="data/thermal_extension.csv",
    )

    plot_mean_energy_vs_temperature(
        thermal_dataframe=thermal_dataframe,
        output_path="figures/mean_energy_vs_temperature.png",
    )

    plot_mean_scattering_angle_vs_temperature(
        thermal_dataframe=thermal_dataframe,
        output_path="figures/mean_scattering_angle_vs_temperature.png",
    )

    print("\nGenerated data files:")
    print("  data/results.csv")
    print("  data/summary.json")
    print("  data/thermal_extension.csv")

    print("\nGenerated figures:")
    print("  figures/initial_configuration.png")
    print("  figures/final_configuration.png")
    print("  figures/sample_trajectories.png")
    print("  figures/scattering_angle_vs_impact_parameter.png")
    print("  figures/scattering_angle_histogram.png")
    print("  figures/energy_conservation.png")
    print("  figures/relative_energy_error.png")
    print("  figures/mean_energy_vs_temperature.png")
    print("  figures/mean_scattering_angle_vs_temperature.png")

    print("\nSimulation completed successfully.")


if __name__ == "__main__":
    main()