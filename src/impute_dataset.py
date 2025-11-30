"""
Impute missing values using the research-backed strategy outlined in the plan.
"""

from __future__ import annotations

import pandas as pd
from sklearn.impute import IterativeImputer

from src.mass_radius_imputation import impute_mass_radius

MICE_COLS = [
    "pl_eqt",
    "pl_insol",
    "pl_orbsmax",
    "pl_orbper",
    "pl_dens",
    "st_teff",
    "st_met",
    "st_mass",
    "st_rad",
    "pl_surfgrav_m_s2",
    "pl_escvel_km_s",
]


def iterative_imputation(df: pd.DataFrame) -> pd.DataFrame:
    imputer = IterativeImputer(
        random_state=42,
        sample_posterior=True,
        max_iter=15,
        initial_strategy="median",
        n_jobs=-1,
        verbose=1,
        tol=1e-4,
    )
    solar_mask = df["hostname"].str.lower().eq("sun")
    imputer.fit(df.loc[~solar_mask, MICE_COLS])
    imputed_values = imputer.transform(df[MICE_COLS])
    before = df[MICE_COLS].copy()
    df[MICE_COLS] = imputed_values
    for col in MICE_COLS:
        df[f"{col}_mice_flag"] = before[col].isna().astype(int)

    return df


def impute(df: pd.DataFrame) -> pd.DataFrame:
    """Apply two-stage imputation to the dataframe."""
    df = df.copy()
    # Step 1: deterministic reconstruction for mass/radius pairs
    df = impute_mass_radius(df, mass_col="pl_masse", radius_col="pl_rade")
    # Step 2: Iterative imputation for broader feature set (excluding solar rows during fit)
    df = iterative_imputation(df)
    return df
