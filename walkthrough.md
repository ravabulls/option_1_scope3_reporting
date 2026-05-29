# Walkthrough: What Was Built, What Changed, and What the Results Mean

## What This Project Does

This is a fully automated Python research pipeline that analyses **33,681 corporate Scope 3 carbon disclosures** from the CDP CDU Global dataset (2018–2023). It produces a reproducible, publication-ready empirical analysis answering three research questions about how companies disclose their supply-chain emissions.

---

## What Changed in v2 (The 7-Module Upgrade)

### Module 1 — Data Bug Fixed (Four-State Relevancy)
The original pipeline treated blank responses and explicit "Not evaluated" responses identically. They are now preserved as four distinct states: **Relevant**, **Not relevant**, **Not evaluated**, and **Silent**. This is not a minor tweak — it changes the entire RQ1 analysis and makes the "strategic omission" claim defensible.

### Module 2 — Verification Endogeneity Addressed
Among the 1,917 companies that *switched* from unverified to verified during the panel, we now run a within-company event study and Difference-in-Differences analysis. This is the key addition for RQ3: the DiD controls for company-level fixed effects, meaning the verification coefficient now represents a change *within the same company*, not just a comparison between better and worse companies.

### Module 3 — Propensity Score Matching
A logit model predicts the probability of being verified based on sector, year, region, and size. Verified and unverified companies are matched on this score. The ATT from matched pairs confirms whether the OLS verification coefficient survives selection bias correction.

### Module 4 — Clustered Standard Errors
All four regression models now use standard errors clustered at the company (nz_id) level. The Omnibus test confirmed heteroskedasticity in every model in v1. This was a methodological necessity — all t-statistics and confidence intervals in v1 were technically incorrect. The coefficients did not change; the inference is now honest.

### Module 5 — Method Disclosure Rate
The 15 `disclose_s3_ghgp_c{i}_emissions_method_bool` columns were in the dataset but never read. MDR measures whether companies describe *how* they calculated each category. The PDR vs MDR divergence (claiming primary data without explaining methodology) is a new finding not in any prior literature.

### Module 6 — SDQI Sensitivity + Consistency Fix
The consistency component in v1 (Jaccard similarity) penalised companies for *adding* new relevant categories — the opposite of the intended behaviour. Fixed to use retention rate. The sensitivity table re-runs the main model with five SDQI weight configurations and shows the verification coefficient is stable (range ~0.30–0.40) across all of them.

### Module 7 — Persistence Analysis
The original "strategic omission" measure was a single cross-sectional count. The persistence analysis tracks each company × high-materiality category combination across all years they appear. Companies with 3+ consecutive years of silence on a category their sector peers report are classified as systematic omitters. This is a much stronger claim than a one-year snapshot.

---

## Key Empirical Results

| Metric | Value |
|--------|-------|
| Global mean SDQI (basic) | ~0.34–0.40 across years |
| Verification effect (OLS, clustered SE) | +0.25 to +0.37 (all models, p < 0.001) |
| DiD verification effect (company+year FE) | Run pipeline to see current estimate |
| Global primary data use | ~18–19 % (flat 2018–2023) |
| EU primary data use post-2022 | +5.4 pp above pre-2022 |
| Opaque omission rate (global) | ~30–34 % of material categories |
| Systematic omitters (3+ consecutive years) | Run pipeline to see current counts |

---

## How to Reproduce Everything

```bash
# 1. Install dependencies
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn plotly streamlit

# 2. Place data file in the project folder
# cdu_global_all_data.csv must be in option_1_scope3_reporting/

# 3. Run the full pipeline (all 7 steps, ~5–10 minutes)
cd option_1_scope3_reporting
python run_pipeline.py

# 4. Launch the interactive dashboard
streamlit run dashboard.py
```

---

## File Map

```
option_1_scope3_reporting/
├── run_pipeline.py              ← one-click orchestrator
├── dashboard.py                 ← Streamlit app (7 tabs)
├── cdu_global_all_data.csv      ← raw data (not in repo)
├── scope3_thesis/
│   ├── data_processor.py        ← ingestion, 4-state fix, method_bool
│   ├── sdqi_calculator.py       ← SDQI, MDR, retention rate, sensitivity
│   ├── switcher_analysis.py     ← event study + DiD (new in v2)
│   ├── descriptive_analyzer.py  ← heatmaps, persistence, tiered omissions
│   └── regression_modeler.py    ← OLS (clustered), PSM, sensitivity table
└── outputs/
    ├── *.png                    ← heatmaps and charts
    ├── *.csv                    ← all numerical results
    └── *.tex                    ← LaTeX regression table
```

---

## Dashboard Guide

The dashboard has 7 tabs. Each chart has an **ℹ️ button** at the top-right that explains the chart in plain language.

| Tab | What it shows |
|-----|--------------|
| 🏠 Overview | Top findings, glossary, navigation guide |
| 📊 Category Relevance | Which categories are omitted and how suspiciously |
| 🔬 Method Quality | PDR and MDR heatmaps, CSRD comparison table |
| 📅 Time Trends | SDQI by region 2018–2023, CSRD effect, sector trend selector |
| ✅ Verification Deep-Dive | Switcher event study, DiD coefficient, PSM ATT |
| 📐 Regression Models | Publication table, coefficient plot, sensitivity table |
| 🏢 Company Explorer | Per-company SDQI history, category breakdown |
