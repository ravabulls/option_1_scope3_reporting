# Implementation Plan v2 — Scope 3 Disclosure Quality Analysis
## "How Companies Report Scope 3 Emissions: Completeness, Omission, and Method Quality"

> **Status:** All 7 improvement modules implemented. Run `python run_pipeline.py` to regenerate all outputs.

---

## Research Questions

| RQ | Question | Addressed by |
|----|----------|-------------|
| RQ1 | Which Scope 3 categories do companies systematically classify as irrelevant, and does this suggest strategic omission rather than genuine non-materiality? | Modules 1 & 7 |
| RQ2 | What proportion of disclosures rely on primary vs secondary data, and how does this vary across sectors and time? | Modules 5 & 6 |
| RQ3 | Does third-party verification and boundary choice predict higher disclosure quality? | Modules 2, 3, & 4 |

---

## SDQI Formula

### SDQI_basic (main specification)
$$\text{SDQI}_{\text{basic},it} = 0.7 \cdot \text{CS}_{it} + 0.3 \cdot \text{PDR}_{it}$$

- **CS** (Completeness Score): proportion of sector-material categories that are addressed (either Relevant+emissions or actively dismissed as Not relevant). Denominator is sector-specific — categories material for ≥30 % of sector peers.
- **PDR** (Primary Data Ratio): average primary-data percentage across all relevant-reported categories.

### SDQI_extended (robustness)
$$\text{SDQI}_{\text{extended},it} = 0.6 \cdot \text{CS}_{it} + 0.2 \cdot \text{PDR}_{it} + 0.2 \cdot \text{MDR}_{it}$$

- **MDR** (Method Disclosure Rate): proportion of relevant-reported categories where the company also described its calculation methodology. New in v2.

### SDQI_panel (temporal)
$$\text{SDQI}_{\text{panel},it} = 0.8 \cdot \text{SDQI}_{\text{basic},it} + 0.2 \cdot \text{RetentionRate}_{it}$$

- **RetentionRate** (fixed Consistency): proportion of prior-year relevant categories that remain relevant. **Does NOT penalise expansion** (companies adding new categories are rewarded, not penalised). This corrects a flaw in v1 which used symmetric Jaccard similarity.

---

## Four-State Relevancy (Critical — v2 fix)

CDP disclosures have four distinct states for each category:

| State | Meaning | Omission tier |
|-------|---------|---------------|
| `Relevant` | Category assessed as material; emissions reported | None |
| `Not relevant` | Category actively assessed and dismissed | Active dismissal |
| `Not evaluated` | Company explicitly acknowledged non-evaluation | Tier 2 (suspicious) |
| `Silent` (was NaN) | Company gave no response at all | Tier 1 (most suspicious) |

**v1 bug:** `fillna("Not evaluated")` merged `Silent` and `Not evaluated` into one label.  
**v2 fix:** NaN → `"Silent"`, preserving all four states separately.

---

## Improvement Modules (v2)

### Module 1 — Four-State Relevancy Bug Fix
**File:** `scope3_thesis/data_processor.py`  
Preserves NaN as "Silent" instead of merging it with "Not evaluated". Enables tier-based omission analysis.

### Module 2 — Within-Company Switcher Event Study & DiD
**File:** `scope3_thesis/switcher_analysis.py` (new)  
Uses 1,917 companies that switch from unverified to verified during the panel.  
Addresses the endogeneity concern in RQ3.  
Produces:
- `outputs/switcher_event_study.csv` — SDQI at T-2 through T+2 relative to first verification
- `outputs/switcher_did_results.csv` — DiD estimate with company + year fixed effects

### Module 3 — Propensity Score Matching
**File:** `scope3_thesis/regression_modeler.py` (`run_psm_analysis`)  
Matches verified to unverified company-years on sector × year × region × size.  
Produces:
- `outputs/psm_results.csv` — ATT by year + overall weighted ATT

### Module 4 — Clustered Standard Errors
**File:** `scope3_thesis/regression_modeler.py`  
All four regression models now use `cov_type='cluster'` clustered at `nz_id` level.  
Corrects for heteroskedasticity and within-company serial correlation.

### Module 5 — Method Disclosure Rate (MDR)
**Files:** `scope3_thesis/data_processor.py`, `scope3_thesis/sdqi_calculator.py`  
Reads `disclose_s3_ghgp_c{i}_emissions_method_bool` (15 columns, previously unused).  
MDR = share of relevant-reported categories where company described methodology.  
Produces:
- `outputs/mdr_matrix_sector.csv` + `outputs/heatmap_mdr_sector.png`

### Module 6 — Consistency Fix + SDQI Sensitivity
**File:** `scope3_thesis/sdqi_calculator.py`, `scope3_thesis/regression_modeler.py`  
- Fixes Jaccard-based consistency to use retention rate (non-penalising of expansion).
- Adds 5 SDQI weight configurations (sensitivity analysis).  
Produces:
- `outputs/sdqi_sensitivity_table.csv`

### Module 7 — Persistence Analysis
**File:** `scope3_thesis/descriptive_analyzer.py`  
For each company × high-materiality category, computes max consecutive years of "Silent" or "Not evaluated".  
Companies with 3+ consecutive years are classified as "systematic omitters".  
Produces:
- `outputs/persistence_scores_sector.csv`
- `outputs/persistence_scores_category.csv`
- `outputs/persistence_distribution.png`

---

## Regression Specifications

$$\text{SDQI}_{it} = \beta_0 + \beta_1 \text{Verified}_{it} + \beta_2 \text{BoundaryApproach}_{it} + \beta_3 \log(\text{Size}_{it}) + \gamma_s + \delta_t + \lambda_r + \varepsilon_{it}$$

where:
- $\gamma_s$ = sector fixed effects
- $\delta_t$ = year fixed effects  
- $\lambda_r$ = region fixed effects (EU-27, US, Japan, China, Rest of World)
- Standard errors clustered at company ($nz_id$) level

**Four model specifications:**
1. Global Basic — full 33,681 sample
2. Panel Cohort — restricted to firms with ≥2 years (consistency available)
3. EU Subsample — 6,577 EU-only disclosures
4. Known Sectors — 11,953 disclosures with identified sector

---

## Data Dictionary (key columns)

| Column | Description |
|--------|-------------|
| `sdqi_basic` | Main SDQI (0–1) |
| `sdqi_extended` | SDQI including MDR (0–1) |
| `sdqi_panel` | SDQI with retention-based consistency (0–1) |
| `sdqi_w50_50` / `w60_40` / `w80_20` | Sensitivity variants |
| `completeness_score` | CS component (0–1) |
| `primary_data_ratio` | PDR component (0–1) |
| `method_disclosure_rate` | MDR component (0–1) |
| `retention_rate` | Year-on-year category retention (0–1) |
| `expansion_rate` | New categories added as fraction of current set |
| `consistency_score` | Legacy Jaccard (kept for backward compat.) |
| `is_switcher` | Company switches unverified→verified during panel |
| `event_time` | Years relative to first verification year |

---

## Size Proxy Justification

No revenue or headcount data is available in the CDU Global export. We use  
$\text{SizeProxy}_{it} = \log_{10}(\text{Scope1}_{it} + \text{Scope2}_{it} + 1)$  
which is highly correlated with operational scale in carbon accounting studies and is  
available for all 33,681 rows. Its coefficient is positive and significant in all models,  
confirming that larger (by emissions scale) companies disclose more completely.

---

## Output File Reference

| File | Description |
|------|-------------|
| `heatmap_relevancy_sector.png` | Category relevance rates by sector |
| `heatmap_omission_opaque.png` | Silent+NotEval omission rates |
| `heatmap_primary_data_sector.png` | PDR by sector & category |
| `heatmap_mdr_sector.png` | MDR by sector & category |
| `persistence_distribution.png` | Histogram of omission persistence |
| `sdqi_trends_time.png` | SDQI trends 2018–2023 by region |
| `relevancy_matrix_sector.csv` | Relevance rate matrix |
| `omission_opaque_matrix_sector.csv` | Opaque omission rate matrix |
| `primary_data_matrix_sector.csv` | PDR matrix |
| `mdr_matrix_sector.csv` | MDR matrix |
| `strategic_omissions_tiered_sector.csv` | Tiered omission summary by sector |
| `persistence_scores_sector.csv` | Systematic omitter rates by sector |
| `persistence_scores_category.csv` | Systematic omitter rates by category |
| `sdqi_trends_region.csv` | Mean SDQI by region & year |
| `policy_shift_comparison.csv` | Pre/post-2022 quality comparison |
| `switcher_event_study.csv` | SDQI event study for verification switchers |
| `switcher_did_results.csv` | DiD coefficient (company+year FE) |
| `psm_results.csv` | PSM ATT by year |
| `sdqi_sensitivity_table.csv` | Verification coefficient across weight specs |
| `regression_summary_*.txt` | Full model summaries (4 models) |
| `regression_publication_table.csv` | Side-by-side table (CSV) |
| `regression_publication_table.tex` | Side-by-side table (LaTeX) |
