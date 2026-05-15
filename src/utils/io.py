"""
Input and output utilities for the Rutherford Monte Carlo simulation.

This module converts simulation results into files that can later be used
for plots, tables and report figures.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.domain.models import ScatteringResult


def ensure_directory(directory: str | Path) -> Path:
    """
    Create a directory if it does not already exist.
    """

    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)

    return path


def results_to_dataframe(results: list[ScatteringResult]) -> pd.DataFrame:
    """
    Convert a list of scattering results into a pandas DataFrame.
    """

    if len(results) == 0:
        raise ValueError("The results list cannot be empty.")

    data = []

    for particle_id, result in enumerate(results, start=1):
        data.append(
            {
                "particle_id": particle_id,
                "impact_parameter": result.impact_parameter,
                "scattering_angle_sim": result.scattering_angle_sim,
                "scattering_angle_theory": result.scattering_angle_theory,
                "initial_energy": result.initial_energy,
                "final_energy": result.final_energy,
                "relative_energy_error": result.relative_energy_error,
                "final_x": result.final_position[0],
                "final_y": result.final_position[1],
                "final_vx": result.final_velocity[0],
                "final_vy": result.final_velocity[1],
            }
        )

    return pd.DataFrame(data)


def save_results_csv(
    results: list[ScatteringResult],
    output_path: str | Path = "data/results.csv",
) -> Path:
    """
    Save scattering results to a CSV file.
    """

    output_path = Path(output_path)
    ensure_directory(output_path.parent)

    dataframe = results_to_dataframe(results)
    dataframe.to_csv(output_path, index=False)

    return output_path


def save_summary_json(
    summary: dict[str, float],
    output_path: str | Path = "data/summary.json",
) -> Path:
    """
    Save a statistical summary to a JSON file.
    """

    output_path = Path(output_path)
    ensure_directory(output_path.parent)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    return output_path


def load_results_csv(input_path: str | Path = "data/results.csv") -> pd.DataFrame:
    """
    Load scattering results from a CSV file.
    """

    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"The file does not exist: {input_path}")

    return pd.read_csv(input_path)