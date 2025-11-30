"""
Helper functions for deterministic mass-radius reconstruction (Chen & Kipping, 2017).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MassRadiusRegime:
    exponent: float
    r_min: float
    r_max: float | None
    k: float


def _compute_regimes() -> Tuple[MassRadiusRegime, ...]:
    """
    Build piecewise power-law regimes ensuring continuity at boundaries.
    """
    regimes = []

    # Rocky planets: M = R^3.7 for R < 1.5 R_earth
    rocky_exp = 3.7
    rocky_k = 1.0  # passes through Earth (1,1)
    regimes.append(
        MassRadiusRegime(exponent=rocky_exp, r_min=0.0, r_max=1.5, k=rocky_k)
    )

    # Sub-Neptunes: calibrate constant to ensure continuity at R=1.5
    sub_exp = 2.6
    boundary_r = 1.5
    boundary_m = rocky_k * (boundary_r**rocky_exp)
    sub_k = boundary_m / (boundary_r**sub_exp)
    regimes.append(MassRadiusRegime(exponent=sub_exp, r_min=1.5, r_max=4.0, k=sub_k))

    # Gas giants: exponent 0.93, match at R=4
    giant_exp = 0.93
    boundary_r = 4.0
    boundary_m = sub_k * (boundary_r**sub_exp)
    giant_k = boundary_m / (boundary_r**giant_exp)
    regimes.append(
        MassRadiusRegime(exponent=giant_exp, r_min=4.0, r_max=None, k=giant_k)
    )

    return tuple(regimes)


REGIMES = _compute_regimes()


def _select_regime(radius: float) -> MassRadiusRegime:
    for regime in REGIMES:
        upper = regime.r_max if regime.r_max is not None else float("inf")
        if regime.r_min <= radius < upper:
            return regime
    return REGIMES[-1]


def predict_mass_from_radius(radius: pd.Series) -> pd.Series:
    """
    Predict mass (Earth masses) from radius (Earth radii) using the piecewise relations.
    """
    radius = radius.clip(lower=1e-6)

    def _predict_single(r: float) -> float:
        regime = _select_regime(r)
        return regime.k * (r**regime.exponent)

    return radius.apply(_predict_single)


def predict_radius_from_mass(mass: pd.Series) -> pd.Series:
    """
    Invert the mass-radius relation to estimate radius from mass.
    """
    mass = mass.clip(lower=1e-6)

    def _predict_single(m: float) -> float:
        for regime in REGIMES:
            upper = regime.r_max if regime.r_max is not None else float("inf")
            lower_mass = (
                regime.k * (regime.r_min**regime.exponent) if regime.r_min > 0 else 0.0
            )
            upper_mass = (
                regime.k * (upper**regime.exponent)
                if np.isfinite(upper)
                else float("inf")
            )
            if lower_mass <= m < upper_mass:
                return (m / regime.k) ** (1.0 / regime.exponent)
        return (m / REGIMES[-1].k) ** (1.0 / REGIMES[-1].exponent)

    return mass.apply(_predict_single)


def impute_mass_radius(
    df: pd.DataFrame,
    mass_col: str = "pl_masse",
    radius_col: str = "pl_rade",
) -> pd.DataFrame:
    """
    Fill missing mass or radius entries using the Chen & Kipping relations.
    """
    df = df.copy()

    missing_mass = df[mass_col].isna() & df[radius_col].notna()
    if missing_mass.any():
        df.loc[missing_mass, mass_col] = predict_mass_from_radius(
            df.loc[missing_mass, radius_col]
        )
        df[f"{mass_col}_imputed_flag"] = missing_mass.astype(int)
    else:
        df[f"{mass_col}_imputed_flag"] = 0

    missing_radius = df[radius_col].isna() & df[mass_col].notna()
    if missing_radius.any():
        df.loc[missing_radius, radius_col] = predict_radius_from_mass(
            df.loc[missing_radius, mass_col]
        )
        df[f"{radius_col}_imputed_flag"] = missing_radius.astype(int)
    else:
        df[f"{radius_col}_imputed_flag"] = 0

    return df
