"""
Trajectory visualization tools for the Rutherford scattering simulation.

This module generates the plots related to initial configurations, final
configurations and complete particle trajectories.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.domain.models import ScatteringResult, SimulationParameters
from src.simulation.integrator import (
    default_simulation_parameters,
    simulate_single_scattering,
)


def ensure_figure_directory(output_path: str | Path) -> Path:
    """
    Create the parent directory for a figure if it does not already exist.
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    return path


def generate_sample_trajectories(
    impact_parameters: list[float] | np.ndarray | None = None,
    parameters: SimulationParameters | None = None,
) -> list[ScatteringResult]:
    """
    Generate a small set of trajectories for visualization.

    These trajectories are not meant to replace the Monte Carlo ensemble.
    They are used only to illustrate how the impact parameter changes the
    particle path.
    """

    if parameters is None:
        parameters = default_simulation_parameters()

    if impact_parameters is None:
        impact_parameters = np.array([-5.0, -2.0, -1.0, 1.0, 2.0, 5.0])

    results = []

    for impact_parameter in impact_parameters:
        result = simulate_single_scattering(
            impact_parameter=float(impact_parameter),
            parameters=parameters,
            store_trajectory=True,
        )

        results.append(result)

    return results


def plot_initial_configuration(
    results: list[ScatteringResult],
    output_path: str | Path = "figures/initial_configuration.png",
) -> Path:
    """
    Plot the initial configuration of the incident particles.

    The particles start at x = -L with different impact parameters.
    """

    if len(results) == 0:
        raise ValueError("The results list cannot be empty.")

    output_path = ensure_figure_directory(output_path)

    initial_positions = []

    for result in results:
        if result.trajectory is None:
            raise ValueError("Each result must include a stored trajectory.")

        initial_positions.append(result.trajectory.position[0])

    initial_positions = np.array(initial_positions)

    plt.figure(figsize=(7, 5))
    plt.scatter(initial_positions[:, 0], initial_positions[:, 1], label="Incident particles")
    plt.scatter([0.0], [0.0], marker="x", s=100, label="Fixed nucleus")

    plt.axhline(0.0, linewidth=0.8)
    plt.axvline(0.0, linewidth=0.8)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Initial configuration")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def plot_final_configuration(
    results: list[ScatteringResult],
    output_path: str | Path = "figures/final_configuration.png",
) -> Path:
    """
    Plot the final configuration after scattering.

    The final particle positions and final velocity directions are shown.
    """

    if len(results) == 0:
        raise ValueError("The results list cannot be empty.")

    output_path = ensure_figure_directory(output_path)

    final_positions = np.array(
        [result.final_position for result in results],
        dtype=float,
    )

    final_velocities = np.array(
        [result.final_velocity for result in results],
        dtype=float,
    )

    velocity_norms = np.linalg.norm(final_velocities, axis=1)
    velocity_norms[velocity_norms == 0.0] = 1.0
    normalized_velocities = final_velocities / velocity_norms[:, None]

    plt.figure(figsize=(7, 5))
    plt.scatter(final_positions[:, 0], final_positions[:, 1], label="Final positions")
    plt.quiver(
        final_positions[:, 0],
        final_positions[:, 1],
        normalized_velocities[:, 0],
        normalized_velocities[:, 1],
        angles="xy",
        scale_units="xy",
        scale=0.15,
        width=0.004,
    )

    plt.scatter([0.0], [0.0], marker="x", s=100, label="Fixed nucleus")

    plt.axhline(0.0, linewidth=0.8)
    plt.axvline(0.0, linewidth=0.8)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Final configuration")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def plot_trajectories(
    results: list[ScatteringResult],
    output_path: str | Path = "figures/sample_trajectories.png",
) -> Path:
    """
    Plot the full trajectories of selected particles.
    """

    if len(results) == 0:
        raise ValueError("The results list cannot be empty.")

    output_path = ensure_figure_directory(output_path)

    plt.figure(figsize=(8, 6))

    for result in results:
        if result.trajectory is None:
            raise ValueError("Each result must include a stored trajectory.")

        positions = result.trajectory.position

        plt.plot(
            positions[:, 0],
            positions[:, 1],
            label=f"b = {result.impact_parameter:.2f}",
        )

    plt.scatter([0.0], [0.0], marker="x", s=100, label="Fixed nucleus")

    plt.axhline(0.0, linewidth=0.8)
    plt.axvline(0.0, linewidth=0.8)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Sample Rutherford scattering trajectories")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path