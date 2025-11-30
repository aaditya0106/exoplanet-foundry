"""
Utilities for calculating research-backed derived features.

These helpers centralize the astrophysical formulas:
- Surface gravity & escape velocity (Schulze-Makuch et al., 2011)
- Bulk density sanity checks (Seager et al., 2007)
- Habitable zone bounds (Kasting et al., 1993)
- Eccentricity habitability modifier (Rodríguez-Mozos & Moya, 2025)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.constants import G_SI, M_EARTH_KG, R_EARTH_M, T_SUN_K


def surface_gravity_from_mass_radius(
    mass_earth: pd.Series,
    radius_earth: pd.Series,
) -> pd.Series:
    """
    Compute surface gravity (m/s^2) normalized to each planet.

    g = G * M / R^2, where M and R are converted from Earth units to SI.
    """
    mass_kg = mass_earth * M_EARTH_KG
    radius_m = radius_earth * R_EARTH_M
    gravity = G_SI * mass_kg / np.square(radius_m)
    return gravity


def escape_velocity_from_mass_radius(
    mass_earth: pd.Series,
    radius_earth: pd.Series,
) -> pd.Series:
    """
    Compute escape velocity (km/s) given mass and radius in Earth units.

    v_esc = sqrt(2 * G * M / R)
    """
    mass_kg = mass_earth * M_EARTH_KG
    radius_m = radius_earth * R_EARTH_M
    esc_vel = np.sqrt(2 * G_SI * mass_kg / radius_m)
    return esc_vel / 1000.0


def bulk_density_from_mass_radius(
    mass_earth: pd.Series,
    radius_earth: pd.Series,
) -> pd.Series:
    """
    Compute bulk density in g/cm^3 from mass and radius in Earth units.
    """
    mass_kg = mass_earth * M_EARTH_KG
    radius_m = radius_earth * R_EARTH_M
    volume_m3 = (4.0 / 3.0) * np.pi * np.power(radius_m, 3)
    density_kg_m3 = mass_kg / volume_m3
    return density_kg_m3 / 1000.0  # convert to g/cm^3


def stellar_luminosity_from_radius_temp(
    star_radius_solar: pd.Series,
    star_temp_k: pd.Series,
) -> pd.Series:
    """
    Estimate stellar luminosity relative to the Sun using the Stefan-Boltzmann law.

    L_star / L_sun = (R_star / R_sun)^2 * (T_star / T_sun)^4
    """
    radius_term = np.square(star_radius_solar)
    temp_term = np.power(star_temp_k / T_SUN_K, 4)
    return radius_term * temp_term


@dataclass
class HabitableZoneBounds:
    inner_au: pd.Series
    outer_au: pd.Series


def habitable_zone_bounds(
    stellar_luminosity: pd.Series,
) -> HabitableZoneBounds:
    """
    Compute conservative habitable zone bounds (Kasting et al., 1993).

    Inner = sqrt(L / 1.1), Outer = sqrt(L / 0.53) in AU.
    """
    inner = np.sqrt(stellar_luminosity / 1.1)
    outer = np.sqrt(stellar_luminosity / 0.53)
    return HabitableZoneBounds(inner_au=inner, outer_au=outer)


def eccentricity_habitability_score(eccentricity: pd.Series) -> pd.Series:
    """
    Simple monotonic penalty for high eccentricity (Rodríguez-Mozos & Moya, 2025).

    score = 1 / (1 + e); yields 1.0 for circular orbits, <1 for higher e.
    """
    return 1.0 / (1.0 + eccentricity.clip(lower=0))


def annotate_in_habitable_zone(
    semi_major_axis_au: pd.Series,
    hz_bounds: HabitableZoneBounds,
) -> pd.Series:
    """
    Flag planets whose semi-major axis falls inside the computed HZ bounds.
    """
    return (
        (semi_major_axis_au >= hz_bounds.inner_au)
        & (semi_major_axis_au <= hz_bounds.outer_au)
    ).astype(int)
