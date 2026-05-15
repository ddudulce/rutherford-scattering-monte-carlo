"""
Physical and numerical constants for the Rutherford scattering simulation.

The simulation is mainly performed in dimensionless units. This avoids
unnecessary numerical stiffness while preserving the structure of the
classical Coulomb scattering problem.
"""

# SI constants, kept for reference and for the written report.
COULOMB_CONSTANT = 8.9875517923e9          # N m^2 C^-2
ELEMENTARY_CHARGE = 1.602176634e-19       # C
VACUUM_PERMITTIVITY = 8.8541878128e-12    # F/m
ALPHA_PARTICLE_MASS = 6.6446573357e-27    # kg


# Dimensionless constants used in the numerical simulation.
DEFAULT_PARTICLE_MASS = 1.0
DEFAULT_COULOMB_COUPLING = 1.0

# Numerical safety parameter to avoid division by zero.
DEFAULT_SOFTENING = 1.0e-9

# Default simulation parameters.
DEFAULT_INITIAL_DISTANCE = 200.0
DEFAULT_INITIAL_SPEED = 1.0
DEFAULT_TIME_STEP = 1.0e-2
DEFAULT_MAX_STEPS = 500_000