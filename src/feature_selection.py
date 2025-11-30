"""
Train a Random Forest model and compute SHAP-based feature importances.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


ESI_FEATURES = [
    "pl_rade",
    "pl_masse",
    "pl_dens_calc",
    "pl_escvel_km_s",
    "pl_eqt",
]

HABITABILITY_FEATURES = [
    "pl_insol",
    "in_habitable_zone",
    "pl_orbeccen",
    "ecc_hab_score",
    "pl_orbsmax",
    "pl_orbper",
    "st_teff",
    "st_met",
    "st_mass",
    "st_rad",
    "pl_surfgrav_m_s2",
]

FEATURE_COLUMNS = list(dict.fromkeys(ESI_FEATURES + HABITABILITY_FEATURES))


def compute_earth_distance(X: np.ndarray, earth_vector: np.ndarray) -> np.ndarray:
    """Euclidean distance from each row to Earth's vector in standardized space."""
    diffs = X - earth_vector
    return np.linalg.norm(diffs, axis=1)


def prepare_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare feature matrix from dataframe."""
    feature_df = df[FEATURE_COLUMNS].copy()
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    feature_df = feature_df.fillna(feature_df.median(numeric_only=True))
    return feature_df


def standardize_features(
    feature_df: pd.DataFrame, df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Standardize features and extract Earth's vector."""
    scaler = StandardScaler()
    X = scaler.fit_transform(feature_df)

    earth_mask = df["pl_name"].str.lower() == "earth"
    earth_row = feature_df.loc[earth_mask]
    if earth_row.empty:
        raise ValueError("Earth row not found in dataset.")
    earth_vector = scaler.transform(earth_row)[0]

    return X, earth_vector, scaler


def train_earth_distance_model(
    feature_df: pd.DataFrame, distances: np.ndarray
) -> RandomForestRegressor:
    """Train Random Forest model to predict Earth distance."""
    model = RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
        verbose=1,
        max_depth=5,
    )
    model.fit(feature_df, distances)
    return model


def compute_shap_importance(
    model: RandomForestRegressor, feature_df: pd.DataFrame
) -> pd.DataFrame:
    """Compute SHAP values and create importance dataframe."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(feature_df)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    importance_df = (
        pd.DataFrame({"feature": FEATURE_COLUMNS, "mean_abs_shap": mean_abs_shap})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    return importance_df


def save_feature_importance_results(
    df: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    importance_df: pd.DataFrame,
    output_dir,
) -> None:
    """Save feature importance results to files."""
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "dataset_with_earth_distance.csv", index=False)
    feature_matrix.to_csv(output_dir / "feature_matrix.csv", index=False)
    importance_df.to_csv(output_dir / "feature_importance_shap.csv", index=False)


def compute_feature_importance(
    df: pd.DataFrame,
    output_dir=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute SHAP-based feature importance for Earth distance prediction."""
    feature_df = prepare_feature_matrix(df)
    X, earth_vector, scaler = standardize_features(feature_df, df)

    distances = compute_earth_distance(X, earth_vector)
    df["earth_distance"] = distances

    model = train_earth_distance_model(feature_df, distances)
    importance_df = compute_shap_importance(model, feature_df)
    feature_matrix = feature_df.copy()
    feature_matrix["earth_distance"] = distances

    if output_dir:
        save_feature_importance_results(df, feature_matrix, importance_df, output_dir)

    return importance_df, feature_matrix


# def main() -> None:
#     """CLI entry point - reads from default paths."""
#     from pathlib import Path
#     from src.constants import PROCESSED_DIR, REPORTS_DIR

#     imputed_path = PROCESSED_DIR / "processed_exoplanets_imputed.csv"
#     df = pd.read_csv(imputed_path)

#     importance_df, feature_matrix = compute_feature_importance(
#         df, output_dir=REPORTS_DIR
#     )

#     print("Top features by SHAP importance:")
#     print(importance_df.head(10))


# if __name__ == "__main__":
#     main()
