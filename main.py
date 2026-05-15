from src.simulation.integrator import simulate_single_scattering


def main() -> None:
    impact_parameters = [0.5, 1.0, 2.0, 5.0]

    for b in impact_parameters:
        result = simulate_single_scattering(
            impact_parameter=b,
            store_trajectory=True,
        )

        print(f"b = {b}")
        print(f"theta_sim     = {result.scattering_angle_sim:.6f} rad")
        print(f"theta_theory  = {result.scattering_angle_theory:.6f} rad")
        print(f"energy_error  = {result.relative_energy_error:.3e}")
        print("-" * 40)


if __name__ == "__main__":
    main()