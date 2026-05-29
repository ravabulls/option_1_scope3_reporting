# Setup Guide — Scope 3 Disclosure Quality Analysis

> **"How Companies Report Scope 3 Emissions: A Large-Scale Analysis of Completeness, Category Relevance, and Method Quality in Corporate Carbon Disclosures"**

---

## Requirements

- Python 3.10 or higher
- ~1 GB free RAM for dataset processing
- Streamlit >= 1.31 (required for ℹ️ popover buttons in the dashboard)

## Installation

```bash
pip install pandas>=2.0 numpy>=1.24 matplotlib>=3.7 seaborn>=0.12 \
            statsmodels>=0.14 scikit-learn>=1.3 plotly>=5.15 streamlit>=1.31
```

## Data File

Place `cdu_global_all_data.csv` (66.8 MB, 33,681 rows × 273 columns) inside `option_1_scope3_reporting/`. This is the CDP CDU Global dataset and is not included in the repository.

---

## Running the Pipeline

From inside `option_1_scope3_reporting/`:

```bash
python run_pipeline.py
```

This runs all 7 steps in sequence (~5–10 minutes):

| Step | What it does |
|------|-------------|
| 1 | Data ingestion — reads, cleans, preserves 4-state relevancy |
| 2 | SDQI calculation — basic, extended, panel, and 3 sensitivity variants |
| 3 | Descriptive analysis — heatmaps, omission tiers, persistence scores |
| 4 | Switcher event study + DiD — addresses endogeneity in RQ3 |
| 5 | Panel OLS regressions — 4 models with clustered standard errors |
| 6 | Propensity Score Matching — ATT comparison vs OLS |
| 7 | SDQI sensitivity table — verification coefficient across 5 weight configs |

All outputs are saved to `outputs/`.

---

## Launching the Dashboard

```bash
streamlit run dashboard.py
```

Open the URL shown in the terminal (usually http://localhost:8501).

- The dashboard loads raw data and computes SDQI on first run (~30 sec, then cached).
- Run `python run_pipeline.py` first so all output files exist.
- Every chart has an **ℹ️ button** at the top-right with plain-language explanations.

---

## SDQI Index — Mathematical Summary

$$\text{SDQI}_{\text{basic},it} = 0.7 \cdot \text{CS}_{it} + 0.3 \cdot \text{PDR}_{it}$$

$$\text{SDQI}_{\text{extended},it} = 0.6 \cdot \text{CS}_{it} + 0.2 \cdot \text{PDR}_{it} + 0.2 \cdot \text{MDR}_{it}$$

$$\text{SDQI}_{\text{panel},it} = 0.8 \cdot \text{SDQI}_{\text{basic},it} + 0.2 \cdot \text{RetentionRate}_{it}$$

| Component | Definition |
|-----------|-----------|
| **CS** (Completeness Score) | Proportion of sector-material categories that are addressed (Relevant + emissions, or active Not relevant dismissal). Denominator = categories material for ≥30 % of sector peers. |
| **PDR** (Primary Data Ratio) | Average primary-data % across all Relevant-reported categories. |
| **MDR** (Method Disclosure Rate) | Proportion of Relevant-reported categories where company described calculation methodology. |
| **RetentionRate** | Proportion of prior-year Relevant categories still Relevant. Does **not** penalise expansion (adding new categories). |

---

## Regression Specification

$$\text{SDQI}_{it} = \beta_0 + \beta_1\,\text{Verified}_{it} + \beta_2\,\text{BoundaryApproach}_{it} + \beta_3\,\log\text{Size}_{it} + \gamma_s + \delta_t + \lambda_r + \varepsilon_{it}$$

- $\gamma_s$ = sector fixed effects
- $\delta_t$ = year fixed effects
- $\lambda_r$ = region fixed effects (EU-27, US, Japan, China, Rest of World)
- Standard errors **clustered at company (nz_id) level** in all models

**Why clustered SE?** The same company appears in multiple years. Observations within a company are correlated. Ordinary standard errors assume independence across all rows, which is violated in panel data. Clustering corrects this.

---

## Four-State Relevancy System

Each CDP response has one of four states per Scope 3 category:

| State | Meaning | Strategic concern |
|-------|---------|------------------|
| `Relevant` | Material; emissions reported | None |
| `Not relevant` | Actively assessed as immaterial | Low (active assessment) |
| `Not evaluated` | Company said it did not evaluate | Medium (Tier 2 omission) |
| `Silent` | No response at all | High (Tier 1 omission) |

Tier 1 (Silent) and Tier 2 (Not evaluated) combined = **opaque omission rate**.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `FileNotFoundError: cdu_global_all_data.csv` | Place the data file in `option_1_scope3_reporting/` |
| `ModuleNotFoundError: statsmodels` | Run `pip install statsmodels scikit-learn` |
| `AttributeError: 'streamlit' has no attribute 'popover'` | Upgrade: `pip install --upgrade streamlit` |
| Dashboard shows "missing file" warnings | Run `python run_pipeline.py` first |
| Pipeline slow on PSM step | Expected — sklearn nearest-neighbour on 30k rows takes ~2 min |
