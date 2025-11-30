from __future__ import annotations

from pathlib import Path

# Fundamental constants
G_SI = 6.67430e-11  # Gravitational constant [m^3 kg^-1 s^-2]

# Earth reference values
M_EARTH_KG = 5.9722e24
R_EARTH_M = 6.371e6
EARTH_DENSITY = 5.51  # g/cm^3
EARTH_ESCAPE_VELOCITY = 11.186  # km/s
EARTH_EQUILIBRIUM_TEMP = 255.0  # K (blackbody temperature)

# Solar references
M_SUN_KG = 1.98847e30
R_SUN_M = 6.957e8
T_SUN_K = 5778.0

# Paths - relative to project root
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = Path("reports")
