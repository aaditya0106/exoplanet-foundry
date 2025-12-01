"""
Train a Random Forest model and compute SHAP-based feature importances.
"""

from __future__ import annotations

from dataclasses import dataclass

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

DERIVED_FEATURES = [
    "ecc_hab_score",
    "pl_dens_calc",
    "pl_escvel_km_s",
    "pl_surfgrav_m_s2",
]


@dataclass
class FeatureImportanceResults:
    importance_df: pd.DataFrame
    feature_matrix: pd.DataFrame
    X: np.ndarray
    earth_vector: np.ndarray
    model: RandomForestRegressor
    shap_values: np.ndarray
    df_filtered: pd.DataFrame


def identify_correlated_features(
    feature_df: pd.DataFrame, threshold: float = 0.95
) -> list[tuple[str, str, float]]:
    """
    Identify highly correlated feature pairs using vectorized operations.

    Returns list of (feature1, feature2, correlation) tuples sorted by correlation.
    """
    corr_matrix = feature_df.corr().abs()

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    corr_stacked = corr_matrix.where(mask).stack()

    high_corr_pairs = [
        (feat1, feat2, corr_val)
        for (feat1, feat2), corr_val in corr_stacked.items()
        if corr_val >= threshold
    ]

    return sorted(high_corr_pairs, key=lambda x: x[2], reverse=True)


def remove_correlated_features(
    feature_df: pd.DataFrame,
    features_to_remove: list[str],
) -> pd.DataFrame:
    """Remove specified features from feature matrix."""
    return feature_df.drop(columns=features_to_remove, errors="ignore")


def get_features_to_remove(
    high_corr_pairs: list[tuple[str, str, float]],
) -> list[str]:
    """
    Determine which features to remove based on correlation pairs.
    KEEPS derived features and removes raw features they're derived from.
    """
    features_to_remove = set()

    for feat1, feat2, _ in high_corr_pairs:
        if feat1 in DERIVED_FEATURES and feat2 not in DERIVED_FEATURES:
            features_to_remove.add(feat2)
        elif feat2 in DERIVED_FEATURES and feat1 not in DERIVED_FEATURES:
            features_to_remove.add(feat1)
        else:
            features_to_remove.add(feat2)

    return list(features_to_remove)


def compute_earth_distance(X: np.ndarray, earth_vector: np.ndarray) -> np.ndarray:
    """Euclidean distance from each row to Earth's vector in standardized space."""
    diffs = X - earth_vector
    return np.linalg.norm(diffs, axis=1)


def prepare_feature_matrix(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    min_features_required: float = 0.5,
    required_core_features: list[str] | None = None,
) -> pd.DataFrame:
    """
    Clean and prepare feature matrix from dataframe.
    """
    if feature_columns is None:
        feature_columns = FEATURE_COLUMNS

    if required_core_features is None:
        # Core ESI features: radius and mass are minimum requirements
        required_core_features = ["pl_rade", "pl_masse"]

    feature_df = df[feature_columns].copy()
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)

    # Step 1: Filter rows that have required core features
    core_features_present = [
        f for f in required_core_features if f in feature_df.columns
    ]
    if core_features_present:
        core_mask = feature_df[core_features_present].notna().all(axis=1)
        feature_df = feature_df[core_mask].copy()
        print(
            f"Filtered to {len(feature_df)} planets with required core features: {core_features_present}"
        )

    # Step 2: Filter rows with too many missing values
    if min_features_required > 0:
        n_features = len(feature_columns)
        min_features_count = int(n_features * min_features_required)
        non_null_counts = feature_df.notna().sum(axis=1)
        sufficient_data_mask = non_null_counts >= min_features_count
        n_dropped = (~sufficient_data_mask).sum()
        if n_dropped > 0:
            print(
                f"Dropped {n_dropped} planets with <{min_features_required:.0%} of features present"
            )
        feature_df = feature_df[sufficient_data_mask].copy()

    feature_df = feature_df.fillna(feature_df.median(numeric_only=True))
    return feature_df


def standardize_features(
    feature_df: pd.DataFrame, df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Standardize features and extract Earth's vector."""
    scaler = StandardScaler()
    X = scaler.fit_transform(feature_df)

    earth_mask = df["pl_name"].str.lower() == "earth"
    earth_row = feature_df.loc[earth_mask]
    if earth_row.empty:
        raise ValueError("Earth row not found in dataset.")
    earth_vector = scaler.transform(earth_row)[0]

    return X, earth_vector


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
    model: RandomForestRegressor,
    feature_df: pd.DataFrame,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute SHAP values and create importance dataframe.
    """
    if feature_names is None:
        feature_names = list(feature_df.columns)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(feature_df)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    importance_df = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
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
    correlation_threshold: float = 0.90,
    min_features_required: float = 0.5,
    required_core_features: list[str] | None = None,
) -> FeatureImportanceResults:
    """
    Compute SHAP-based feature importance for Earth distance prediction.

    Returns:
        FeatureImportanceResults: Dataclass containing all results
    """
    feature_df = prepare_feature_matrix(
        df,
        min_features_required=min_features_required,
        required_core_features=required_core_features,
    )

    if correlation_threshold < 1.0:
        high_corr = identify_correlated_features(feature_df, correlation_threshold)
        if high_corr:
            print(
                f"\nFound {len(high_corr)} highly correlated feature pairs (>={correlation_threshold}):"
            )
            for feat1, feat2, corr in high_corr:
                print(f"\t{feat1} <-> {feat2}: {corr:.3f}")

            features_to_remove = get_features_to_remove(high_corr)
            if features_to_remove:
                print(
                    f"\nRemoving {len(features_to_remove)} highly correlated features: {features_to_remove}"
                )
                feature_df = remove_correlated_features(feature_df, features_to_remove)

    df_filtered = df.loc[feature_df.index].copy()
    X, earth_vector = standardize_features(feature_df, df_filtered)

    distances = compute_earth_distance(X, earth_vector)
    df_filtered["earth_distance"] = distances

    model = train_earth_distance_model(feature_df, distances)
    feature_names = list(feature_df.columns)
    importance_df = compute_shap_importance(model, feature_df, feature_names)

    feature_matrix = feature_df.copy()
    feature_matrix["earth_distance"] = distances

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(feature_df)

    if output_dir:
        save_feature_importance_results(
            df_filtered, feature_matrix, importance_df, output_dir
        )

    return FeatureImportanceResults(
        importance_df=importance_df,
        feature_matrix=feature_matrix,
        X=X,
        earth_vector=earth_vector,
        model=model,
        shap_values=shap_values,
        df_filtered=df_filtered,
    )
