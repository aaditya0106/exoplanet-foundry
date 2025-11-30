"""
Generate the processed dataset with research-backed derived features.
"""

from __future__ import annotations

import pandas as pd

from src.derived_features import (
    annotate_in_habitable_zone,
    bulk_density_from_mass_radius,
    eccentricity_habitability_score,
    habitable_zone_bounds,
    surface_gravity_from_mass_radius,
    escape_velocity_from_mass_radius,
    stellar_luminosity_from_radius_temp,
)


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Append the derived feature columns to the input dataframe."""
    mass = df.pl_masse
    radius = df.pl_rade

    df["pl_surfgrav_m_s2"] = surface_gravity_from_mass_radius(mass, radius)
    df["pl_escvel_km_s"] = escape_velocity_from_mass_radius(mass, radius)
    df["pl_dens_calc"] = bulk_density_from_mass_radius(mass, radius)

    df["st_luminosity_solar"] = stellar_luminosity_from_radius_temp(
        df.st_rad, df.st_teff
    )

    hz_bounds = habitable_zone_bounds(df.st_luminosity_solar)
    df["hz_inner_au"] = hz_bounds.inner_au
    df["hz_outer_au"] = hz_bounds.outer_au
    df["in_habitable_zone"] = annotate_in_habitable_zone(df.pl_orbsmax, hz_bounds)

    df["ecc_hab_score"] = eccentricity_habitability_score(df.pl_orbeccen)

    return df
