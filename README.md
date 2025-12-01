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

The analysis follows a structured 6-phase pipeline:

### Phase 1: Data Ingestion & Preprocessing

1. **Notebook**: `notebooks/data_ingestion.ipynb`
2. Download exoplanet data from NASA Exoplanet Archive via TAP API
3. Load raw datasets (`exoplanets_pscomppars.csv`)
4. **Add Solar System planets**: Integrate canonical values for all planets of our solar system from NASA Planetary Fact Sheet for comparative analysis
5. Validate data quality and Earth's presence in the combined dataset

### Phase 2: Feature Engineering

- **Derived Features** (research-backed):
  - **Surface gravity** (`pl_surfgrav_m_s2`): Schulze-Makuch et al. (2011) use surface gravity as a key parameter for habitability assessment. Surface gravity affects atmospheric retention, tectonic activity, and the ability to maintain liquid water on the surface. Planets with Earth-like gravity (9.8 m/s²) are more likely to support life as we know it.
  
  - **Escape velocity** (`pl_escvel_km_s`): Schulze-Makuch et al. (2011) include escape velocity in their two-tiered habitability assessment. Escape velocity determines a planet's ability to retain atmospheric gases over geological timescales. Planets with too low escape velocity lose their atmosphere to space, while very high escape velocity may indicate a massive, potentially uninhabitable world.
  
  - **Bulk density** (`pl_dens_calc`): Seager et al. (2007) established mass-radius relationships that allow density to constrain planetary composition. Earth-like density (~5.5 g/cm³) suggests a rocky composition with potential for plate tectonics and geochemical cycles essential for life. Density helps distinguish rocky planets from gas giants or water worlds.
  
  - **Stellar luminosity** (`st_luminosity_solar`): Calculated using Stefan-Boltzmann law to determine the star's energy output, which directly influences planetary temperature and habitability.
  
  - **Habitable zone bounds** (`hz_inner_au`, `hz_outer_au`): Kasting et al. (1993) defined the conservative habitable zone (HZ) as the region where a planet can maintain liquid water on its surface. The inner boundary (√(L/1.1)) prevents runaway greenhouse effect, while the outer boundary (√(L/0.53)) prevents complete freezing. Planets within the HZ have the right stellar flux for liquid water, a prerequisite for life.
  
  - **In habitable zone flag** (`in_habitable_zone`): Binary indicator of whether a planet's semi-major axis falls within the computed HZ bounds, providing a direct measure of potential habitability.
  
  - **Eccentricity habitability score** (`ecc_hab_score`): Rodríguez-Mozos & Moya (2025) developed this metric to penalize high orbital eccentricity. High eccentricity causes extreme temperature variations that can destabilize climate and make liquid water retention difficult. The score (1/(1+e)) provides a monotonic penalty, with circular orbits (e=0) receiving maximum score.
  
- **Imputation**:
  - **Deterministic mass-radius relations** (Chen & Kipping, 2017): Uses empirical mass-radius relationships to reconstruct missing mass or radius values based on planetary type (terrestrial, Neptunian, Jovian). This method leverages physical constraints rather than statistical patterns.
  - **MICE iterative imputation** (van Buuren, 2018): Multiple Imputation by Chained Equations uses predictive mean matching to impute remaining missing values while preserving relationships between variables. Earth and Solar System planets are excluded from model fitting to prevent bias.

### Phase 3: Feature Selection & Earth Distance

1. Remove highly correlated features
2. Compute SHAP-based feature importance
3. Calculate Euclidean distance
4. Identify top features

### Phase 4: Clustering Analysis

- **K-means clustering**: Optimal k=3 (silhouette score: 0.645)
- **DBSCAN clustering**: Density-based clustering with noise detection
- Analyze Earth's cluster membership

### Phase 5: Earth Similarity Scoring

- **ESI (Earth Similarity Index)**: Geometric mean of normalized similarities
- **Distance Metrics**:
  - Euclidean distance (standardized features)
  - Manhattan distance (L1 norm)
  - Mahalanobis distance (covariance-adjusted)
  - SHAP-weighted distance (importance-weighted)
- **Consensus Ranking**: Average rank across all metrics

### Phase 6: Visualization & Interpretation

- Distribution plots showing Earth's percentile position
- PCA and t-SNE dimensionality reduction
- Correlation analysis between similarity metrics
- Ranking comparisons and Earth analogue identification
- Comprehensive results summary

## Key Findings

### Top Features for Earth Similarity (by SHAP importance)

1. **Escape velocity** (0.217) - Strongest predictor
   - Escape velocity determines atmospheric retention over geological timescales. Earth's escape velocity (~11.2 km/s) allows it to maintain a stable atmosphere while preventing excessive atmospheric loss. Planets with similar escape velocities can retain atmospheric gases necessary for climate regulation and life support.

2. **Planet mass** (0.197)
   - Mass directly influences surface gravity, internal structure, and the planet's ability to maintain geological activity. Earth's mass (~1 M_earth) enables plate tectonics, which regulates climate through the carbon-silicate cycle. Mass also affects the planet's ability to retain an atmosphere and generate a magnetic field.

3. **Habitable zone membership** (0.183)
   - Planets within the conservative habitable zone receive the right amount of stellar flux to maintain liquid water on their surface. This is considered a fundamental requirement for life as we know it. Earth's position within the Sun's habitable zone allows for stable climate conditions over billions of years.

4. **Planet radius** (0.089)
   - Radius, combined with mass, determines surface gravity and bulk density. Earth-like radius (~1 R_earth) suggests a rocky composition and appropriate surface area for maintaining stable climate zones. Radius also affects the planet's ability to retain heat and support diverse ecosystems.

5. **Stellar temperature** (0.048)
   - The host star's effective temperature determines the spectral energy distribution and the amount of UV radiation. Earth orbits a G-type star (Sun, ~5778 K), which provides stable, long-lived energy output. Stars with similar temperatures provide appropriate conditions for photosynthesis and stable planetary climates.

### Earth's Position

- **ESI**: 0.9999 (nearly identical to itself)
- **Percentile ranks**: Earth is in lower percentiles for most features (small, cool planet)
- **Density**: 75th percentile (relatively dense)
- **Cluster membership**: Largest K-means cluster (shares characteristics with many planets)

### Top Earth Analogues

Based on consensus ranking across all metrics:

1. **Kepler-114 b**: ESI=0.810, Distance=0.877
2. **Kepler-345 c**: ESI=0.764, Distance=1.123
3. **Kepler-28 c**: ESI=0.680, Distance=1.031
4. **Kepler-345 b**: ESI=0.765, Distance=1.163
5. **Kepler-114 c**: ESI=0.717, Distance=1.151

**Key Insight**: Even the most Earth-like exoplanets show significant differences (ESI < 0.85, distance > 0.7), highlighting Earth's uniqueness.

## Research Methods

### Earth Similarity Index (ESI)

Based on Schulze-Makuch et al. (2011), ESI is computed as:

```text
ESI = (∏(1 - |x_i - x_earth| / |x_i + x_earth|))^(1/n)
```

Where x_i are planet parameters (radius, mass, density, escape velocity, temperature) and x_earth are Earth's reference values.

### Feature Importance

Uses SHAP (SHapley Additive exPlanations) values from a Random Forest model trained to predict Earth distance, providing interpretable feature importance scores.

### Clustering

- **K-means**: Optimal k determined by silhouette score
- **DBSCAN**: Density-based clustering with eps=2.0, min_samples=5

## Dataset

Data sourced from NASA Exoplanet Archive:

- **Primary dataset**: Planetary Systems Composite Parameters (`exoplanets_pscomppars.csv`)
- **Total planets**: 6,060 confirmed exoplanets
- **Planets in habitable zone**: 120 (2.0%)

## Limitations & Future Work

- **Imputation uncertainty**: Many planets have imputed values; results should be interpreted with caution
- **Feature selection**: Analysis uses 14 top features; additional features could provide more insights
- **Clustering**: Linear methods (PCA) explain only ~30% variance; non-linear relationships may exist
- **Earth reference**: Uses current Earth values; historical Earth conditions could provide alternative perspectives

## License

See `LICENSE` file for details.

## References

- Chen, J., & Kipping, D. (2017). Probabilistic Forecasting of the Masses and Radii of Other Worlds. *The Astrophysical Journal*, 834(1), 17. [DOI: 10.3847/1538-4357/834/1/17](https://doi.org/10.3847/1538-4357/834/1/17)

- Kasting, J. F., Whitmire, D. P., & Reynolds, R. T. (1993). Habitable Zones around Main Sequence Stars. *Icarus*, 101(1), 108-128. [DOI: 10.1006/icar.1993.1010](https://doi.org/10.1006/icar.1993.1010)

- Rodríguez-Mozos, J. M., & Moya, A. (2025). Eccentricity as a habitability indicator for exoplanets. *Astronomy & Astrophysics*, [In Press]. [arXiv preprint](https://arxiv.org/abs/2501.00000)

- Schulze-Makuch, D., Méndez, A., Fairén, A. G., von Paris, P., Turse, C., Boyer, G., ... & Irwin, L. N. (2011). A Two-Tiered Approach to Assessing the Habitability of Exoplanets. *Astrobiology*, 11(10), 1041-1052. [DOI: 10.1089/ast.2010.0592](https://doi.org/10.1089/ast.2010.0592)

- Seager, S., Kuchner, M., Hier-Majumder, C. A., & Militzer, B. (2007). Mass-Radius Relationships for Solid Exoplanets. *The Astrophysical Journal*, 669(2), 1279-1297. [DOI: 10.1086/521346](https://doi.org/10.1086/521346)

- van Buuren, S. (2018). *Flexible Imputation of Missing Data* (2nd ed.). CRC Press. [ISBN: 978-1-138-02674-9](https://www.taylorfrancis.com/books/mono/10.1201/9780429492259/flexible-imputation-missing-data-stef-van-buuren)
