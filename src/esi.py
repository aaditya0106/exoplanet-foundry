"""
Earth Similarity Index (ESI) calculation based on Schulze-Makuch et al. (2011).

ESI = (TT(1 - |x_i - x_earth|/|x_i + x_earth|))^(1/n)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import src.constants as consts


def compute_esi_component(
    planet_value: pd.Series,
    earth_value: float,
) -> pd.Series:
    """
    Compute ESI component: 1 - |x - x_earth| / |x + x_earth|

    Returns:
        Series of component values (0-1, where 1 = identical to Earth)
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
    extra_features: list[str] | None = None,
    earth_idx: int | None = None,
) -> pd.Series:
    """
    Generic ESI calculator for standard (5 params) or extended (5+N params).

    Standard ESI: radius, mass, [density], [escape_vel], temperature

    Returns:
    Series of ESI values (0-1, Earth ≈ 1.0)
    """
    components = []

    # Standard ESI components (research-backed constants)
    if "pl_rade" in df.columns:
        components.append(compute_esi_component(df["pl_rade"], 1.0))

    if "pl_masse" in df.columns:
        components.append(compute_esi_component(df["pl_masse"], 1.0))

    if use_density:
        if "pl_dens_calc" in df.columns:
            components.append(
                compute_esi_component(df["pl_dens_calc"], consts.EARTH_DENSITY)
            )
        elif "pl_dens" in df.columns:
            components.append(
                compute_esi_component(df["pl_dens"], consts.EARTH_DENSITY)
            )

    if use_escape_velocity and "pl_escvel_km_s" in df.columns:
        components.append(
            compute_esi_component(df["pl_escvel_km_s"], consts.EARTH_ESCAPE_VELOCITY)
        )

    if "pl_eqt" in df.columns:
        components.append(
            compute_esi_component(df["pl_eqt"], consts.EARTH_EQUILIBRIUM_TEMP)
        )

    if extra_features:
        if earth_idx is None:
            earth_mask = df.index[df.get("pl_name", pd.Series()).str.lower() == "earth"]
            earth_idx = earth_mask[0] if len(earth_mask) > 0 else None

        for feat in extra_features:
            if feat in df.columns and earth_idx is not None:
                earth_val = df.loc[earth_idx, feat]
                if not pd.isna(earth_val) and earth_val != 0:
                    components.append(compute_esi_component(df[feat], earth_val))

    if not components:
        raise ValueError("No valid ESI components found")

    # Geometric mean of all components
    components_df = pd.DataFrame(components).T
    return components_df.prod(axis=1) ** (1.0 / len(components))


def compute_esi_radius_mass_only(df: pd.DataFrame) -> pd.Series:
    """Simplified 2-parameter ESI (radius + mass only)."""
    return compute_esi(df, use_density=False, use_escape_velocity=False)


def compute_extended_esi(
    df: pd.DataFrame,
    extra_features: list[str],
    earth_idx: int | None = None,
) -> pd.Series:
    """
    Alias for compute_esi with extra_features.
    Extended ESI = geometric_mean(5 standard + N extra features).
    """
    return compute_esi(df, extra_features=extra_features, earth_idx=earth_idx)
