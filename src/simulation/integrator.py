"""
Numerical integration for one classical Rutherford scattering trajectory.

The trajectory is computed with the Velocity Verlet method, which is more
stable for conservative systems than the explicit Euler method.
"""

from __future__ import annotations

import numpy as np

from src.analysis.theoretical import (
    rutherford_scattering_angle,
    scattering_angle_from_velocity,
)
from src.domain.constants import (
    DEFAULT_COULOMB_COUPLING,
    DEFAULT_INITIAL_DISTANCE,
    DEFAULT_INITIAL_SPEED,
    DEFAULT_MAX_STEPS,
    DEFAULT_PARTICLE_MASS,
    DEFAULT_SOFTENING,
    DEFAULT_TIME_STEP,
)
from src.domain.models import (
    Nucleus,
    Particle,
    ScatteringResult,
    SimulationParameters,
    Trajectory,
)
from src.domain.physics import acceleration, kinetic_energy, total_energy


def default_simulation_parameters() -> SimulationParameters:
    """
    Return the default numerical parameters for the simulation.
    """

    return SimulationParameters(
        initial_distance=DEFAULT_INITIAL_DISTANCE,
        initial_speed=DEFAULT_INITIAL_SPEED,
        time_step=DEFAULT_TIME_STEP,
        max_steps=DEFAULT_MAX_STEPS,
        softening=DEFAULT_SOFTENING,
    )


def create_initial_particle(
    impact_parameter: float,
    parameters: SimulationParameters,
    mass: float = DEFAULT_PARTICLE_MASS,
    charge: float = 1.0,
) -> Particle:
    """
    Create an incident particle starting at x = -L with velocity along +x.

    The impact parameter is the initial y-coordinate.
    """

    position = np.array(
        [-parameters.initial_distance, impact_parameter],
        dtype=float,
    )

    velocity = np.array(
        [parameters.initial_speed, 0.0],
        dtype=float,
    )

    return Particle(
        mass=mass,
        charge=charge,
        position=position,
        velocity=velocity,
    )


def simulate_single_scattering(
    impact_parameter: float,
    parameters: SimulationParameters | None = None,
    coupling: float = DEFAULT_COULOMB_COUPLING,
    nucleus: Nucleus | None = None,
    store_trajectory: bool = True,
) -> ScatteringResult:
    """
    Simulate the classical scattering of one particle by a fixed nucleus.

    The particle starts at x = -L, y = b and moves initially along the
    positive x-axis. The simulation stops once the particle has interacted
    with the nucleus and has moved far away again.
    """

    if parameters is None:
        parameters = default_simulation_parameters()

    if nucleus is None:
        nucleus = Nucleus(
            charge=1.0,
            position=np.array([0.0, 0.0], dtype=float),
        )

    particle = create_initial_particle(
        impact_parameter=impact_parameter,
        parameters=parameters,
    )

    dt = parameters.time_step
    position = particle.position.copy()
    velocity = particle.velocity.copy()

    initial_radius = float(np.linalg.norm(position - nucleus.position))

    initial_kinetic_energy = kinetic_energy(
        particle_mass=particle.mass,
        velocity=velocity,
    )

    initial_total_energy = total_energy(
        position=position,
        velocity=velocity,
        nucleus_position=nucleus.position,
        particle_mass=particle.mass,
        coupling=coupling,
        softening=parameters.softening,
    )

    time_values: list[float] = []
    position_values: list[np.ndarray] = []
    velocity_values: list[np.ndarray] = []
    energy_values: list[float] = []

    current_acceleration = acceleration(
        position=position,
        nucleus_position=nucleus.position,
        particle_mass=particle.mass,
        coupling=coupling,
        softening=parameters.softening,
    )

    for step in range(parameters.max_steps):
        time = step * dt

        if store_trajectory:
            time_values.append(time)
            position_values.append(position.copy())
            velocity_values.append(velocity.copy())
            energy_values.append(
                total_energy(
                    position=position,
                    velocity=velocity,
                    nucleus_position=nucleus.position,
                    particle_mass=particle.mass,
                    coupling=coupling,
                    softening=parameters.softening,
                )
            )

        new_position = (
            position
            + velocity * dt
            + 0.5 * current_acceleration * dt**2
        )

        new_acceleration = acceleration(
            position=new_position,
            nucleus_position=nucleus.position,
            particle_mass=particle.mass,
            coupling=coupling,
            softening=parameters.softening,
        )

        new_velocity = velocity + 0.5 * (
            current_acceleration + new_acceleration
        ) * dt

        position = new_position
        velocity = new_velocity
        current_acceleration = new_acceleration

        radius = float(np.linalg.norm(position - nucleus.position))
        radial_velocity = float(np.dot(position - nucleus.position, velocity))

        has_moved_away = radial_velocity > 0.0
        is_far_again = radius >= initial_radius

        if step > 10 and has_moved_away and is_far_again:
            break

    final_total_energy = total_energy(
        position=position,
        velocity=velocity,
        nucleus_position=nucleus.position,
        particle_mass=particle.mass,
        coupling=coupling,
        softening=parameters.softening,
    )

    if initial_total_energy != 0.0:
        relative_energy_error = abs(
            (final_total_energy - initial_total_energy) / initial_total_energy
        )
    else:
        relative_energy_error = abs(final_total_energy - initial_total_energy)

    scattering_angle_sim = scattering_angle_from_velocity(velocity)

    scattering_angle_theory = rutherford_scattering_angle(
        impact_parameter=impact_parameter,
        kinetic_energy=initial_kinetic_energy,
        coupling=coupling,
        signed=True,
    )

    trajectory = None

    if store_trajectory:
        trajectory = Trajectory(
            time=np.array(time_values),
            position=np.array(position_values),
            velocity=np.array(velocity_values),
            energy=np.array(energy_values),
        )

    return ScatteringResult(
        impact_parameter=impact_parameter,
        scattering_angle_sim=scattering_angle_sim,
        scattering_angle_theory=scattering_angle_theory,
        initial_energy=initial_total_energy,
        final_energy=final_total_energy,
        relative_energy_error=relative_energy_error,
        final_position=position,
        final_velocity=velocity,
        trajectory=trajectory,
    )
    