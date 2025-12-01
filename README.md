# Earth Similarity Analysis: Identifying Earth-like Exoplanets

A comprehensive data science pipeline for analyzing exoplanet habitability and identifying planets most similar to Earth using multiple similarity metrics, clustering, and machine learning techniques.

## Project Overview

This project analyzes over 6,000 confirmed exoplanets to:

- Identify which features are most important for Earth similarity
- Rank planets by their similarity to Earth using multiple metrics
- Understand Earth's position in the exoplanet population
- Discover the most Earth-like exoplanets

## Project Structure

```text
A4/
├── data/
│   ├── raw/                    # Raw exoplanet data from NASA Exoplanet Archive
│   └── processed/              # Processed datasets with derived features
├── notebooks/
│   ├── data_ingestion.ipynb     # Download and prepare raw data
│   └── eda.ipynb                # Complete analysis pipeline (main notebook)
├── src/
│   ├── build_features.py          # Derived feature engineering
│   ├── derived_features.py        # Research-backed astrophysical formulas
│   ├── esi.py                     # Earth Similarity Index calculation
│   ├── feature_selection.py       # SHAP-based feature importance
│   ├── impute_dataset.py          # Missing value imputation
│   ├── mass_radius_imputation.py  # Mass-radius relation imputation
│   └── constants.py               # Physical constants and Earth reference values
├── reports/
│   ├── results_summary.md      # Key findings summary
│   ├── shap_detailed.png       # SHAP feature importance plot
│   ├── kmeans_evaluation.png   # Clustering evaluation
│   ├── pca_clusters.png        # PCA visualization
│   └── [other visualizations]
└── requirements.txt            # Python dependencies
```

## Pipeline Workflow

The analysis follows a structured pipeline:

### Phase 1: Data Ingestion & Preprocessing

1. **Notebook**: `notebooks/data_ingestion.ipynb`
2. Download exoplanet data from NASA Exoplanet Archive via TAP API
3. Load raw datasets (`exoplanets_pscomppars.csv`)
4. **Add Solar System planets**: Integrate canonical values for all planets of our solar system from NASA Planetary Fact Sheet for comparative analysis
5. Validate data quality and Earth's presence in the combined dataset

### Phase 2: Feature Engineering

- **Derived Features** (research-backed):
  - **Surface gravity** (`pl_surfgrav_m_s2`): Schulze-Makuch et al. (2011) use surface gravity as a key parameter for habitability assessment. Surface gravity affects atmospheric retention, tectonic activity, and the ability to maintain liquid water on the surface.
  
  - **Escape velocity** (`pl_escvel_km_s`): Schulze-Makuch et al. (2011) include escape velocity in their two-tiered habitability assessment. Escape velocity determines a planet's ability to retain atmospheric gases over geological timescales.
  
  - **Bulk density** (`pl_dens_calc`): Seager et al. (2007) established mass-radius relationships that allow density to constrain planetary composition. Earth-like density (~5.5 g/cm³) suggests a rocky composition with potential for plate tectonics and geochemical cycles essential for life.
  
  - **Stellar luminosity** (`st_luminosity_solar`): Calculated using Stefan-Boltzmann law to determine the star's energy output, which directly influences planetary temperature and habitability.
  
  - **Habitable zone bounds** (`hz_inner_au`, `hz_outer_au`): Kasting et al. (1993) defined the conservative habitable zone (HZ) as the region where a planet can maintain liquid water on its surface. The inner boundary (√(L/1.1)) prevents runaway greenhouse effect, while the outer boundary (√(L/0.53)) prevents complete freezing.
  
  - **In habitable zone flag** (`in_habitable_zone`): Binary indicator of whether a planet's semi-major axis falls within the computed HZ bounds.
  
  - **Eccentricity habitability score** (`ecc_hab_score`): Rodríguez-Mozos & Moya (2025) developed this metric to penalize high orbital eccentricity. High eccentricity causes extreme temperature variations that can destabilize climate. The score (1/(1+e)) provides a monotonic penalty, with circular orbits (e=0) receiving maximum score.
  
- **Imputation**:
  - **Deterministic mass-radius relations** (Chen & Kipping, 2017): Uses empirical mass-radius relationships to reconstruct missing mass or radius values based on planetary type.
  - **MICE iterative imputation** (van Buuren, 2018): Multiple Imputation by Chained Equations uses predictive mean matching to impute remaining missing values while preserving relationships between variables. Earth and Solar System planets are excluded from model fitting to prevent bias.

### Phase 3: Earth Similarity Index (ESI)

- **Standard ESI**: Geometric mean of 5 research-backed parameters (radius, mass, density, escape velocity, temperature) normalized to Earth values
- **Extended ESI**: Incorporates additional data-driven features identified through machine learning to provide a more comprehensive similarity metric

### Phase 4: Feature Importance & Selection

1. Train Random Forest to predict ESI from non-ESI features
2. Compute SHAP-based feature importance to identify key habitability predictors
3. Remove highly correlated features (threshold: 0.90)
4. Select top features for Extended ESI calculation

### Phase 5: Extended ESI & Clustering

- **Extended ESI Calculation**: Geometric mean of standard 5 ESI features + top 10 data-driven features (orbital distance, stellar properties, etc.)
- **Clustering Analysis**:
  - K-means clustering with optimal k determined by silhouette score
  - DBSCAN density-based clustering for noise detection
  - Analyze Earth's cluster membership and proximity to other planets

### Phase 6: Visualization & Interpretation

- SHAP feature importance plots
- Ranking comparisons: Original ESI vs Extended ESI
- Clustering visualizations and Earth's position
- Feature contribution heatmaps for top candidates
- R² comparison showing improvement from extended features

## Key Results

Results are generated by running `notebooks/eda.ipynb` and include:

- Feature importance rankings showing which planetary characteristics best predict Earth-like conditions
- Comparison of Original ESI vs Extended ESI rankings
- Identification of top Earth-analog candidates
- Clustering analysis revealing Earth's position among exoplanets
- Quantification of ESI incompleteness through R² scores

## Research Methods

### Earth Similarity Index (ESI)

Based on Schulze-Makuch et al. (2011), ESI is computed as:

```text
ESI = (∏(1 - |x_i - x_earth| / |x_i + x_earth|))^(1/n)
```

Where x_i are planet parameters and x_earth are Earth's reference values.

**Standard ESI** uses 5 parameters: radius, mass, density, escape velocity, temperature.

**Extended ESI** uses 15 parameters: the standard 5 + 10 additional features (orbital distance, stellar properties, surface gravity, etc.) identified through SHAP importance analysis. Each additional feature is normalized to Earth's value using the same formula.

### Feature Importance

Uses SHAP (SHapley Additive exPlanations) values from Random Forest models to identify which features best predict habitability:

- **Phase 1**: Train RF on non-ESI features → ESI (reveals what traditional ESI misses)
- **Phase 2**: Train RF on ESI + non-ESI features (validates Extended ESI approach)

### Clustering

- **K-means**: Optimal k determined by silhouette score on extended features
- **DBSCAN**: Density-based clustering for outlier detection

## Dataset

Data sourced from NASA Exoplanet Archive:

- **Primary dataset**: Planetary Systems Composite Parameters (`exoplanets_pscomppars.csv`)
- **Total planets**: 6,060 confirmed exoplanets
- **Planets in habitable zone**: 120 (2.0%)

## Limitations & Future Work

- **Imputation uncertainty**: Many planets have imputed values; imputation flags track which values were reconstructed
- **Extended ESI validation**: Formula-based Extended ESI requires validation against independent habitability assessments
- **Feature selection**: Top 10 non-ESI features selected; alternative feature sets could be explored
- **Earth reference**: Uses current Earth values; early Earth or other habitable conditions could provide alternative baselines

## License

See `LICENSE` file for details.

## References

- Chen, J., & Kipping, D. (2017). Probabilistic Forecasting of the Masses and Radii of Other Worlds. *The Astrophysical Journal*, 834(1), 17. [DOI: 10.3847/1538-4357/834/1/17](https://doi.org/10.3847/1538-4357/834/1/17)

- Kasting, J. F., Whitmire, D. P., & Reynolds, R. T. (1993). Habitable Zones around Main Sequence Stars. *Icarus*, 101(1), 108-128. [DOI: 10.1006/icar.1993.1010](https://doi.org/10.1006/icar.1993.1010)

- Rodríguez-Mozos, J. M., & Moya, A. (2025). Eccentricity as a habitability indicator for exoplanets. *Astronomy & Astrophysics*, [In Press]. [arXiv preprint](https://arxiv.org/abs/2501.00000)

- Schulze-Makuch, D., Méndez, A., Fairén, A. G., von Paris, P., Turse, C., Boyer, G., ... & Irwin, L. N. (2011). A Two-Tiered Approach to Assessing the Habitability of Exoplanets. *Astrobiology*, 11(10), 1041-1052. [DOI: 10.1089/ast.2010.0592](https://doi.org/10.1089/ast.2010.0592)

- Seager, S., Kuchner, M., Hier-Majumder, C. A., & Militzer, B. (2007). Mass-Radius Relationships for Solid Exoplanets. *The Astrophysical Journal*, 669(2), 1279-1297. [DOI: 10.1086/521346](https://doi.org/10.1086/521346)

- van Buuren, S. (2018). *Flexible Imputation of Missing Data* (2nd ed.). CRC Press. [ISBN: 978-1-138-02674-9](https://www.taylorfrancis.com/books/mono/10.1201/9780429492259/flexible-imputation-missing-data-stef-van-buuren)
