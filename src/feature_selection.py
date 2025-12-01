"""
Train a Random Forest model to predict ESI and compute SHAP-based feature importances.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
    model: RandomForestRegressor
    shap_values: np.ndarray
    df_filtered: pd.DataFrame


@dataclass
class ExtendedESIResults:
    model: RandomForestRegressor
    extended_features: list[str]
    importance_df: pd.DataFrame
    shap_values: np.ndarray
    r2_score: float


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


def filter_rows_by_completeness(
    df: pd.DataFrame,
    feature_columns: list[str],
    min_completeness: float = 0.5,
) -> pd.DataFrame:
    """Filter rows based on feature completeness before imputation."""
    feature_df = df[feature_columns].copy()
    n_features = len(feature_columns)
    min_features_count = int(n_features * min_completeness)
    non_null_counts = feature_df.notna().sum(axis=1)
    orig_len = len(df)
    df = df[non_null_counts < min_features_count]
    if orig_len > len(df):
        print(
            f"Filtered {orig_len - len(df)} rows with <{min_completeness:.0%} feature completeness"
        )
    return df


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


def standardize_features(feature_df: pd.DataFrame) -> np.ndarray:
    """Standardize features for model training."""
    scaler = StandardScaler()
    X = scaler.fit_transform(feature_df)
    return X


def train_rf_model(
    X: pd.DataFrame, y: np.ndarray, verbose: int = 0
) -> RandomForestRegressor:
    """Train Random Forest with standard hyperparameters."""
    model = RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
        verbose=verbose,
        max_depth=10,
        min_samples_leaf=5,
    )
    model.fit(X, y)
    return model


def train_esi_model(
    feature_df: pd.DataFrame, esi_values: np.ndarray
) -> RandomForestRegressor:
    """Train RF to predict ESI using NON-ESI features."""
    non_esi_features = [col for col in feature_df.columns if col not in ESI_FEATURES]
    if not non_esi_features:
        raise ValueError("No non-ESI features available!")

    X_train = feature_df[non_esi_features]
    model = train_rf_model(X_train, esi_values, verbose=1)

    r2 = model.score(X_train, esi_values)
    print(f"\nR² (non-ESI → ESI): {r2:.4f}")
    return model


def train_extended_esi_model(
    feature_df: pd.DataFrame,
    esi_values: np.ndarray,
    top_non_esi_features: list[str],
    n_features: int = 10,
) -> tuple[RandomForestRegressor, list[str]]:
    """Train RF on ESI features + top N non-ESI features."""
    extended_features = ESI_FEATURES + top_non_esi_features[:n_features]
    X_extended = feature_df[extended_features]

    model = train_rf_model(X_extended, esi_values, verbose=1)
    r2 = model.score(X_extended, esi_values)

    print(f"\nExtended ESI model ({len(extended_features)} features):")
    print(f"ESI features: {len(ESI_FEATURES)}")
    print(f"  Non-ESI features: {n_features}")
    print(f"  R² (extended → ESI): {r2:.4f}")

    return model, extended_features


def compute_shap_importance(
    model: RandomForestRegressor,
    feature_df: pd.DataFrame,
    feature_names: list[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
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
    return importance_df, shap_values


def save_feature_importance_results(
    df: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    importance_df: pd.DataFrame,
    output_dir,
) -> None:
    """Save feature importance results to files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "dataset_with_esi.csv", index=False)
    feature_matrix.to_csv(output_dir / "feature_matrix.csv", index=False)
    importance_df.to_csv(output_dir / "feature_importance_shap.csv", index=False)


def compute_feature_importance(
    df: pd.DataFrame,
    output_dir=None,
    correlation_threshold: float = 0.90,
    min_features_required: float = 0.5,
    required_core_features: list[str] | None = None,
) -> FeatureImportanceResults:
    """Compute SHAP-based feature importance for ESI prediction."""
    if "esi" not in df.columns:
        raise ValueError("ESI must be calculated before feature importance analysis")

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
    esi_values = df_filtered["esi"].values

    model = train_esi_model(feature_df, esi_values)
    non_esi_features = [col for col in feature_df.columns if col not in ESI_FEATURES]
    feature_df_trained = feature_df[non_esi_features].copy()

    importance_df, shap_values = compute_shap_importance(
        model, feature_df_trained, list(feature_df_trained.columns)
    )

    print("\nTop 10 features predicting ESI (excluding ESI-defining features):")
    print(importance_df.head(10).to_string(index=False))

    X = standardize_features(feature_df)

    feature_matrix = feature_df.copy()
    feature_matrix["esi"] = esi_values

    if output_dir:
        save_feature_importance_results(
            df_filtered, feature_matrix, importance_df, output_dir
        )

    return FeatureImportanceResults(
        importance_df=importance_df,
        feature_matrix=feature_matrix,
        X=X,
        model=model,
        shap_values=shap_values,
        df_filtered=df_filtered,
    )


def compute_extended_esi(
    feature_matrix: pd.DataFrame,
    top_non_esi_features: list[str],
    n_features: int = 10,
) -> ExtendedESIResults:
    """
    Train Extended ESI model using ESI + top N non-ESI features.
    Returns predictions as proposed ESI scores.
    """
    esi_values = feature_matrix["esi"].values
    feature_df = feature_matrix.drop(columns=["esi"])

    model, extended_features = train_extended_esi_model(
        feature_df, esi_values, top_non_esi_features, n_features
    )

    X_extended = feature_df[extended_features]
    importance_df, shap_values = compute_shap_importance(
        model, X_extended, extended_features
    )
    r2 = model.score(X_extended, esi_values)

    return ExtendedESIResults(
        model=model,
        extended_features=extended_features,
        importance_df=importance_df,
        shap_values=shap_values,
        r2_score=r2,
    )
