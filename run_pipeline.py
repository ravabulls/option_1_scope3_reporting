"""
run_pipeline.py — One-click research pipeline orchestrator.

Steps:
  1. Ingest & preprocess CDU Global dataset
  2. Calculate SDQI scores (all variants)
  3. Descriptive analysis & visualisations
  4. Switcher event study + DiD (Module 2)
  5. Panel OLS regressions with clustered SE (Modules 3 & 4)
  6. Propensity Score Matching (Module 3)
  7. SDQI weight sensitivity (Module 6)
  8. Export slim dashboard cache (data_slim.csv — safe to commit, no raw data)
"""

import os
import sys
import time

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from scope3_thesis.data_processor    import load_and_preprocess_data
from scope3_thesis.sdqi_calculator   import calculate_sdqi_scores
from scope3_thesis.descriptive_analyzer import run_descriptive_analysis
from scope3_thesis.switcher_analysis import run_switcher_analysis
from scope3_thesis.regression_modeler import (
    run_regression_analysis,
    run_psm_analysis,
    run_sdqi_sensitivity,
)


def main():
    t0 = time.time()
    csv_file   = "cdu_global_all_data.csv"
    output_dir = "outputs"

    print("=" * 70)
    print("  SCOPE 3 CARBON ACCOUNTING — DISCLOSURE QUALITY RESEARCH PIPELINE")
    print("=" * 70)

    if not os.path.exists(csv_file):
        print(f"\n[ERROR] {csv_file} not found in the current directory.")
        print("Please place the CDP CDU Global data file here and retry.")
        sys.exit(1)

    # ── Step 1: Ingest ──────────────────────────────────────────────────
    print(f"\nStep 1/7  Ingesting & preprocessing '{csv_file}' ...")
    df = load_and_preprocess_data(csv_file)
    print(f"          {len(df):,} disclosures  |  "
          f"{df['reporting_year'].nunique()} years  |  "
          f"{df['is_eu'].sum():,} EU disclosures")
    print(f"          C1 relevancy states: "
          + "  ".join(f"{k}={v}" for k, v in
                      df["s3_ghgp_c1_emissions_relevancy"].value_counts().items()))

    # ── Step 2: SDQI ────────────────────────────────────────────────────
    print("\nStep 2/7  Calculating SDQI scores (basic, extended, panel, sensitivity) ...")
    df = calculate_sdqi_scores(df)
    print(f"          SDQI basic mean:    {df['sdqi_basic'].mean():.4f}")
    print(f"          SDQI extended mean: {df['sdqi_extended'].mean():.4f}")
    print(f"          MDR mean:           {df['method_disclosure_rate'].mean():.4f}")
    print(f"          Retention rate (consistency-fixed) non-null: "
          f"{df['retention_rate'].notna().sum():,}")

    # ── Step 3: Descriptive ─────────────────────────────────────────────
    print("\nStep 3/7  Descriptive analysis & visualisations ...")
    run_descriptive_analysis(df, output_dir=output_dir)

    # ── Step 4: Switcher analysis ────────────────────────────────────────
    print("\nStep 4/7  Switcher event study & Difference-in-Differences ...")
    df, ev_df, ev_stats, did_df, did_stats = run_switcher_analysis(df, output_dir=output_dir)
    if did_stats.get("did_coef") is not None:
        import numpy as np
        if not np.isnan(did_stats["did_coef"]):
            print(f"          DiD coefficient (verification effect, company+year FE): "
                  f"{did_stats['did_coef']:+.4f}  p={did_stats['did_pval']:.4f}")

    # ── Step 5: Regressions ─────────────────────────────────────────────
    print("\nStep 5/7  Panel OLS regressions (clustered standard errors) ...")
    m1, m2, m3, m4 = run_regression_analysis(df, output_dir=output_dir)
    for name, m in [("M1 Global", m1), ("M2 Panel", m2),
                    ("M3 EU", m3),    ("M4 Known", m4)]:
        print(f"          {name}: R²={m.rsquared:.4f}  N={int(m.nobs):,}")

    # ── Step 6: PSM ─────────────────────────────────────────────────────
    print("\nStep 6/7  Propensity Score Matching (verification ATT) ...")
    psm_df, psm_stats = run_psm_analysis(df, output_dir=output_dir)

    # ── Step 7: Sensitivity ─────────────────────────────────────────────
    print("\nStep 7/7  SDQI weight sensitivity analysis ...")
    sens_df = run_sdqi_sensitivity(df, output_dir=output_dir)
    if not sens_df.empty:
        print("          Verification coefficient across SDQI specifications:")
        for _, row in sens_df.iterrows():
            print(f"            {row['SDQI Specification']:<38}  "
                  f"coef={row['Verification Coefficient']:+.4f}  "
                  f"p={row['P-value']:.4f}")

    # ── Step 8: Slim cache for Streamlit Cloud ───────────────────────────
    print("\nStep 8/8  Exporting slim dashboard cache (data_slim.csv) ...")
    _export_slim_cache(df, output_dir)

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"  PIPELINE COMPLETE  ({elapsed:.1f}s)")
    print(f"  All outputs saved to '{output_dir}/'")
    print("  data_slim.csv ready for Streamlit Cloud deployment")
    print("  Launch dashboard:  streamlit run dashboard.py")
    print("=" * 70)


def _export_slim_cache(df, output_dir):
    """
    Saves a compact pre-processed snapshot for the Streamlit dashboard.

    The raw CSV (66.8 MB, 273 cols) stays local.  This file (~15 MB, ~85 cols)
    contains only computed indices and dashboard-required columns and is safe
    to commit to GitHub for Streamlit Cloud deployment.
    """
    import numpy as np

    S3_CATS = [f"c{i}" for i in range(1, 16)]

    keep = [
        # Identity & metadata
        "nz_id", "company_name", "reporting_year",
        "sics_sector", "sics_industry",
        "jurisdiction_clean", "region", "is_eu",
        "boundary_approach_clean", "emissions_verified",
        # SDQI scores
        "sdqi_basic", "sdqi_extended", "sdqi_panel",
        "sdqi_w50_50", "sdqi_w60_40", "sdqi_w80_20",
        # Sub-components
        "completeness_score", "primary_data_ratio",
        "method_disclosure_rate", "consistency_score",
        "retention_rate", "expansion_rate",
        # Switcher flags
        "is_switcher", "never_verified", "ever_verified",
        "first_verified_year", "event_time",
        # Controls & emissions totals
        "company_size_proxy", "post_csrd_announcement",
        "total_s3_emissions_best",
        "total_s1_emissions_ghg", "total_s2_emissions_best",
        "s3_math_discrepancy",
    ]
    # Per-category columns (4 × 15 = 60 columns)
    for c in S3_CATS:
        keep += [
            f"s3_ghgp_{c}_emissions_relevancy",
            f"s3_ghgp_{c}_emissions_primary_data",
            f"disclose_s3_ghgp_{c}_emissions_method_bool",
            f"total_s3_ghgp_{c}_emissions_ghg",
        ]

    # Keep only columns that exist in the dataframe
    keep = [c for c in keep if c in df.columns]
    slim = df[keep].copy()

    # Downcast floats to float32 to save space
    float_cols = slim.select_dtypes("float64").columns
    slim[float_cols] = slim[float_cols].astype("float32")

    out_path = os.path.join(output_dir, "data_slim.csv")
    slim.to_csv(out_path, index=False)
    size_mb = os.path.getsize(out_path) / 1_048_576
    print(f"          Saved {len(slim):,} rows x {len(keep)} cols -> "
          f"{size_mb:.1f} MB  ({out_path})")


if __name__ == "__main__":
    main()
