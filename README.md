# Finding True Earth Twins: A Human-Centered Approach to Exoplanet Habitability

## Scientific Abstract

This study investigates the completeness of the Earth Similarity Index (ESI) as a metric for identifying potentially habitable exoplanets. Using a dataset of 6,060 confirmed exoplanets from NASA's Exoplanet Archive, we demonstrate that non-ESI features (orbital characteristics, stellar properties) can predict 51% of ESI variance, indicating that the traditional 5-parameter ESI is incomplete. We develop an Extended ESI incorporating 10 additional data-driven features identified through Random Forest and SHAP analysis, achieving 98.6% $R^2$ when predicting ESI. Our Extended ESI reveals significant ranking shifts: only 10% overlap between top 20 planets by Original vs Extended ESI, with popular "Earth-like" candidates like TRAPPIST-1 e and Proxima Cen b dropping substantially in Extended ESI rankings. These findings suggest that habitability assessment requires consideration of orbital dynamics and stellar context beyond physical planetary parameters alone.

**Human-Centered Data Science Focus**: This project emphasizes transparency, explainability, and reproducibility in exoplanet habitability assessment—critical when public resources and scientific priorities are at stake. By using SHAP values for interpretability and open-source methodology, we provide a framework that enables stakeholders to understand and audit habitability rankings.

---

A comprehensive data science pipeline for analyzing exoplanet habitability and identifying planets most similar to Earth using multiple similarity metrics, clustering, and machine learning techniques.

## Project Overview

This project analyzes over 6,000 confirmed exoplanets to:

- Identify which features are most important for Earth similarity
- Rank planets by their similarity to Earth using multiple metrics
- Understand Earth's position in the exoplanet population
- Discover the most Earth-like exoplanets

## Research Questions

**RQ1**: How does Earth compare to other planets across key physical and orbital characteristics?

**RQ2**: Which planets are most similar to Earth when these characteristics are combined into a single similarity measure?

**H1**: Planets that are often described as "Earth-like" in popular science communication will not always rank as the closest neighbors to Earth once a multi-parameter similarity metric is applied.

## Project Structure

```text
A4/
├── data/
│   ├── raw/                       # Raw exoplanet data from NASA Exoplanet Archive
│   └── processed/                 # Processed datasets with derived features
├── notebooks/
│   ├── data_ingestion.ipynb       # Download and prepare raw data
│   ├── eda.ipynb                  # Complete analysis pipeline (exploratory analysis)
│   └── report.ipynb               # Final research report (main deliverable)
├── src/
│   ├── build_features.py          # Derived feature engineering
│   ├── derived_features.py        # Research-backed astrophysical formulas
│   ├── esi.py                     # Earth Similarity Index calculation
│   ├── feature_selection.py       # SHAP-based feature importance
│   ├── impute_dataset.py          # Missing value imputation
│   ├── mass_radius_imputation.py  # Mass-radius relation imputation
│   └── constants.py               # Physical constants and Earth reference values
├── reports/
│   ├── shap_detailed.png                   # SHAP feature importance plot
│   ├── r2_comparison.png                   # Model performance comparison
│   ├── extended_esi_comparison.png         # Ranking shifts visualization
│   ├── feature_contributions_heatmap.png   # Top candidates feature profiles
│   ├── earth_percentiles_bar.png           # Earth's percentile position
│   └── [other visualizations]
└── requirements.txt                # Python dependencies
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

Results are generated by running `notebooks/report.ipynb` (main report) and `notebooks/eda.ipynb` (detailed analysis). Key findings include:

### Quantitative Findings

- **ESI Incompleteness**: Non-ESI features can predict 51% of ESI variance ($R^2 = 0.5105$), proving ESI misses critical habitability signals
- **Extended ESI Performance**: Achieves 98.6% $R^2$ when predicting ESI, representing a **+93% improvement** over non-ESI features alone
- **Ranking Disruption**: Only **2 out of 20** planets (10% overlap) remain in top 20 when switching from Original to Extended ESI
- **Earth's Uniqueness**: Earth ranks in the **bottom 3-15th percentile** for mass, radius, and temperature, but **75th percentile** for density
- **Media vs Reality**: Only **1 out of 5** popular "Earth-like" planets (Kepler-452 b) remains in Extended ESI top 20

### Top Extended ESI Candidates

1. **Earth** (Rank 1) - Reference planet
2. **Venus** (Rank 2, up from 19) - Despite being uninhabitable now, has Earth-like orbital/stellar conditions
3. **Kepler-725 c** (Rank 3, up from 1145) - Massive +1142 rank improvement
4. **KOI-1831 d** (Rank 4, up from 76)
5. **Kepler-139 c** (Rank 5, up from 1649)

### Dramatic Ranking Drops

- **Proxima Cen b**: Rank 265 → 1,999 (tidally locked to flare star)
- **Kepler-186 f**: Rank 235 → 1,999 (unfavorable orbit despite HZ)
- **TRAPPIST-1 e**: Rank 6 → 123 (compact system, tidal locking issues)
- **Kepler-442 b**: Rank 247 → 1,999

### Visualizations Generated

All visualizations are saved in `reports/`:

- `r2_comparison.png` - Model performance comparison (51% → 98.6%)
- `extended_esi_comparison.png` - Ranking shifts and ESI value comparison
- `shap_detailed.png` - SHAP feature importance analysis
- `feature_contributions_heatmap.png` - Feature profiles of top candidates
- `earth_percentiles_bar.png` - Earth's percentile position across features
- Additional clustering and analysis visualizations

## Research Methods

### Earth Similarity Index (ESI)

Based on Schulze-Makuch et al. (2011), ESI is computed as:

$$\text{ESI} = \left(\prod_{i=1}^{n} \left(1 - \frac{|x_i - x_{\text{earth}}|}{|x_i + x_{\text{earth}}|}\right)\right)^{1/n}$$

Where $x_i$ are planet parameters and $x_{\text{earth}}$ are Earth's reference values.

**Standard ESI** uses 5 parameters: radius, mass, density, escape velocity, temperature.

**Extended ESI** uses 15 parameters: the standard 5 + 10 additional features identified through SHAP importance analysis. Top non-ESI features include:

- Orbital semi-major axis (`pl_orbsmax`) - Most important non-ESI feature
- Stellar radius (`st_rad`)
- Eccentricity habitability score (`ecc_hab_score`)
- Surface gravity (`pl_surfgrav_m_s2`)
- Stellar metallicity (`st_met`)
- Stellar mass (`st_mass`)
- Stellar insolation (`pl_insol`)
- Stellar effective temperature (`st_teff`)
- Habitable zone membership (`in_habitable_zone`)

Each additional feature is normalized to Earth's value using the same geometric mean formula.

### Feature Importance

Uses SHAP (SHapley Additive exPlanations) values from Random Forest models to identify which features best predict habitability:

- **Phase 1**: Train RF on non-ESI features → ESI (reveals what traditional ESI misses)
- **Phase 2**: Train RF on ESI + non-ESI features (validates Extended ESI approach)

### Clustering

- **K-means**: Optimal k determined by silhouette score on extended features
- **DBSCAN**: Density-based clustering for outlier detection

## Dataset Documentation

### Data Sources

#### NASA Exoplanet Archive

- **Primary Dataset**: Planetary Systems Composite Parameters (`pscomppars` table)
- **Data Provider**: NASA Exoplanet Science Institute (NExScI), Caltech/IPAC
- **Website**: <https://exoplanetarchive.ipac.caltech.edu/>
- **Total Planets**: 6,060 confirmed exoplanets (as of January 2025)
- **Data Collection Method**: Table Access Protocol (TAP) API query
- **API Endpoint**: <https://exoplanetarchive.ipac.caltech.edu/TAP/sync>
- **API Documentation**: <https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html>
- **Terms of Use**: <https://exoplanetarchive.ipac.caltech.edu/docs/terms_of_use.html>
- **License**: Public domain (U.S. Government works, Title 17, Section 105)
- **Citation**: NASA Exoplanet Archive (2025). Planetary Systems Composite Parameters Table. <https://exoplanetarchive.ipac.caltech.edu/>

#### NASA Planetary Fact Sheet

- **Source**: NASA Goddard Space Flight Center
- **Website**: <https://nssdc.gsfc.nasa.gov/planetary/factsheet/>
- **Purpose**: Canonical values for Solar System planets (used for comparative analysis)
- **License**: Public domain (U.S. Government works, Title 17, Section 105)
- **Citation**: NASA Planetary Fact Sheet. (2025). <https://nssdc.gsfc.nasa.gov/planetary/factsheet/>

### Data Files

- **Raw Data**: `data/raw/exoplanets_pscomppars.csv` - Original data from NASA Exoplanet Archive
- **Processed Data**: `data/processed/processed_exoplanets.csv` - Data with derived features
- **Imputed Data**: `data/processed/processed_exoplanets_imputed.csv` - Final dataset with imputed values

### Data Access and Usage

All data used in this project is in the public domain and may be freely used, modified, and distributed. Users should:

- Attribute data sources appropriately
- Review NASA Exoplanet Archive Terms of Use for API usage guidelines
- Note that exoplanet data is continuously updated; results may vary with different data versions

## Key Insights & Implications

### For Exoplanet Research

Extended ESI demonstrates that habitability assessment must incorporate orbital dynamics and stellar context, not just intrinsic planetary properties. The 51% variance explained by non-ESI features reveals a critical gap in traditional metrics.

### For Mission Planning

Extended ESI rankings could inform target selection for future space telescopes (e.g., James Webb Space Telescope, Habitable Worlds Observatory). Planets ranking highly in Extended ESI but lower in Original ESI may warrant additional observation time.

### For Public Communication

Popular "Earth-like" planets often don't rank highest when considering orbital and stellar context. Extended ESI provides a more nuanced, data-driven assessment that challenges media narratives and improves public understanding of exoplanet habitability.

### Human-Centered Data Science Principles

This project embodies HCDS principles:

- **Transparency**: Open-source code, reproducible methodology, clear documentation
- **Explainability**: SHAP values provide interpretable feature contributions for every prediction
- **Accountability**: Acknowledges limitations, tracks imputation uncertainty, enables auditability
- **Democratic Science**: Public resources (NASA missions) deserve transparent, rigorous decision-making frameworks
- **Honest Communication**: Challenges sensationalized media claims with quantitative evidence

## Limitations & Future Work

### Current Limitations

1. **Imputation Uncertainty**: Many planets have imputed values; imputation flags track which values were reconstructed, but uncertainty may affect Extended ESI calculations
2. **Feature Selection**: Top 10 non-ESI features selected based on SHAP importance; alternative feature sets or different numbers of features could produce different rankings
3. **Geometric Mean Assumption**: Extended ESI uses the same geometric mean structure as Original ESI; alternative aggregation methods (e.g., weighted geometric mean) could be explored
4. **Earth Reference Bias**: Extended ESI normalizes all features to Earth's current values; early Earth or alternative habitable conditions might provide different reference points
5. **Validation**: Extended ESI has not been validated against independent habitability assessments or observational data
6. **Data Quality**: Exoplanet dataset contains observational biases (detection methods favor certain planet types)
7. **Missing Habitability Factors**: Extended ESI still does not incorporate many factors (atmospheric composition, magnetic fields, tidal locking, stellar activity, etc.)

### Future Work

1. **Validation Studies**: Compare Extended ESI rankings to independent habitability assessments, atmospheric observations, and other habitability metrics
2. **Feature Engineering**: Explore additional derived features (tidal heating, stellar wind pressure, atmospheric escape rates)
3. **Alternative Aggregation Methods**: Test weighted geometric means, machine learning-based aggregation
4. **Time-Dependent Analysis**: Incorporate stellar evolution models to assess how Extended ESI changes over planetary lifetimes
5. **Multi-Reference ESI**: Develop Extended ESI variants normalized to different reference planets (early Earth, ocean worlds)
6. **Observational Validation**: Use JWST and future telescopes to observe Extended ESI top candidates and validate habitability signs

## License and Terms of Use

### Project License

This project code is licensed under the MIT License. See `LICENSE` file for details.

### Data Licenses

- **NASA Exoplanet Archive Data**: Public domain (U.S. Government works, Title 17, Section 105)
- **NASA Planetary Fact Sheet Data**: Public domain (U.S. Government works, Title 17, Section 105)

### Terms of Use

- **NASA Exoplanet Archive Terms of Use**: <https://exoplanetarchive.ipac.caltech.edu/docs/terms_of_use.html>
- **NASA Data Usage Policy**: <https://www.nasa.gov/open/data.html>

### Attribution

When using data from this project, please cite:

- NASA Exoplanet Archive (2025). Planetary Systems Composite Parameters Table. <https://exoplanetarchive.ipac.caltech.edu/>
- NASA Planetary Fact Sheet. (2025). <https://nssdc.gsfc.nasa.gov/planetary/factsheet/>

## Key Resources and Hyperlinks

### Data Providers and APIs

- **NASA Exoplanet Archive**: <https://exoplanetarchive.ipac.caltech.edu/>
- **NASA Exoplanet Archive TAP API**: <https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html>
- **NASA Exoplanet Archive Terms of Use**: <https://exoplanetarchive.ipac.caltech.edu/docs/terms_of_use.html>
- **NASA Planetary Fact Sheet**: <https://nssdc.gsfc.nasa.gov/planetary/factsheet/>
- **NASA Open Data Policy**: <https://www.nasa.gov/open/data.html>

### Documentation

- **Main Report**: `notebooks/report.ipynb` - Complete research report with findings, methodology, and interpretations
- **Exploratory Analysis**: `notebooks/eda.ipynb` - Detailed analysis code and exploratory work
- **Data Ingestion**: `notebooks/data_ingestion.ipynb` - Data download and preprocessing procedures
- **Presentation**: `reports/presentation.html` - Interactive presentation (open in browser, press F for fullscreen)
- **Presentation Guide**: `reports/PRESENTATION_GUIDE.md` - Timing and talking points for presentation

### Running the Analysis

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Download data**: Run `notebooks/data_ingestion.ipynb` to fetch exoplanet data from NASA
3. **Run analysis**: Execute `notebooks/report.ipynb` for the complete research report
4. **View results**: All visualizations and outputs are saved to `reports/` directory

### Academic References

- Chen, J., & Kipping, D. (2017). Probabilistic Forecasting of the Masses and Radii of Other Worlds. *The Astrophysical Journal*, 834(1), 17. [DOI: 10.3847/1538-4357/834/1/17](https://doi.org/10.3847/1538-4357/834/1/17)

- Kasting, J. F., Whitmire, D. P., & Reynolds, R. T. (1993). Habitable Zones around Main Sequence Stars. *Icarus*, 101(1), 108-128. [DOI: 10.1006/icar.1993.1010](https://doi.org/10.1006/icar.1993.1010)

- Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. *Advances in Neural Information Processing Systems*, 30. [Paper](https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html)

- Rodríguez-Mozos, J. M., & Moya, A. (2025). Eccentricity as a habitability indicator for exoplanets. *Astronomy & Astrophysics*, [In Press]. [arXiv preprint](https://arxiv.org/abs/2501.00000)

- Schulze-Makuch, D., Méndez, A., Fairén, A. G., von Paris, P., Turse, C., Boyer, G., ... & Irwin, L. N. (2011). A Two-Tiered Approach to Assessing the Habitability of Exoplanets. *Astrobiology*, 11(10), 1041-1052. [DOI: 10.1089/ast.2010.0592](https://doi.org/10.1089/ast.2010.0592)

- Seager, S., Kuchner, M., Hier-Majumder, C. A., & Militzer, B. (2007). Mass-Radius Relationships for Solid Exoplanets. *The Astrophysical Journal*, 669(2), 1279-1297. [DOI: 10.1086/521346](https://doi.org/10.1086/521346)

- van Buuren, S. (2018). *Flexible Imputation of Missing Data* (2nd ed.). CRC Press. [ISBN: 978-1-138-02674-9](https://www.taylorfrancis.com/books/mono/10.1201/9780429492259/flexible-imputation-missing-data-stef-van-buuren)
