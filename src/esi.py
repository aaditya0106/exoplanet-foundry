"""
Earth Similarity Index (ESI) calculation based on Schulze-Makuch et al. (2011).

ESI is computed as a geometric mean of normalized parameter similarities:
ESI = (TT(1 - |x_i - x_earth|/|x_i + x_earth|))^(1/n)

Where x_i are the planet parameters and x_earth are Earth's reference values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import src.constants as consts


# Earth reference values for ESI calculation
EARTH_RADIUS = 1.0  # R_earth
EARTH_MASS = 1.0  # M_earth


def compute_esi_component(
    planet_value: pd.Series,
    earth_value: float,
) -> pd.Series:
    """
    Compute individual ESI component: 1 - |x - x_earth| / |x + x_earth|

    Args:
        planet_value: Series of planet parameter values
        earth_value: Earth's reference value for this parameter

    Returns:
        Series of ESI component values (0-1, where 1 is identical to Earth)
    """
    numerator = np.abs(planet_value - earth_value)
    denominator = np.abs(planet_value + earth_value)
    denominator = denominator.replace(0, np.nan)
    component = 1.0 - (numerator / denominator)
    return component.clip(lower=0.0, upper=1.0)


def compute_esi(
    df: pd.DataFrame,
    use_density: bool = True,
    use_escape_velocity: bool = True,
) -> pd.Series:
    """
    Compute Earth Similarity Index (ESI) for all planets.

    ESI uses 4-5 parameters:
    - Radius (R_earth)
    - Mass (M_earth)
    - Density (g/cm^3) - optional
    - Escape velocity (km/s) - optional
    - Equilibrium temperature (K)

    Returns:
        Series of ESI values (0-1, where 1 is identical to Earth)
    """
    components = []

    if "pl_rade" in df.columns:
        radius_comp = compute_esi_component(df["pl_rade"], EARTH_RADIUS)
        components.append(radius_comp)

    if "pl_masse" in df.columns:
        mass_comp = compute_esi_component(df["pl_masse"], EARTH_MASS)
        components.append(mass_comp)

    if use_density:
        if "pl_dens_calc" in df.columns:
            density_comp = compute_esi_component(
                df["pl_dens_calc"], consts.EARTH_DENSITY
            )
            components.append(density_comp)
        elif "pl_dens" in df.columns:
            density_comp = compute_esi_component(df["pl_dens"], consts.EARTH_DENSITY)
            components.append(density_comp)

    if use_escape_velocity and "pl_escvel_km_s" in df.columns:
        esc_vel_comp = compute_esi_component(
            df["pl_escvel_km_s"], consts.EARTH_ESCAPE_VELOCITY
        )
        components.append(esc_vel_comp)

    if "pl_eqt" in df.columns:
        temp_comp = compute_esi_component(df["pl_eqt"], consts.EARTH_EQUILIBRIUM_TEMP)
        components.append(temp_comp)

    if not components:
        raise ValueError("No valid ESI components found in dataframe")

    components_df = pd.DataFrame(components).T
    esi = components_df.prod(axis=1) ** (1.0 / len(components))

    return esi


def compute_esi_radius_mass_only(df: pd.DataFrame) -> pd.Series:
    """
    Compute simplified ESI using only radius and mass (2-parameter ESI).

    This is useful when other parameters are missing.
    """
    return compute_esi(df, use_density=False, use_escape_velocity=False)
