"""
descriptive_analyzer.py — Descriptive statistics, visualisations, and
omission analyses (Modules 1 & 7).

Key additions over v1:
  - Four-state relevancy heatmap (Relevant / Not relevant / Not evaluated / Silent)
  - Tier-ranked strategic omission analysis (Silent > Not evaluated > Not relevant)
  - Persistence analysis: companies with 3+ consecutive years of silence on a
    high-materiality category (Module 7)
  - MDR (Method Disclosure Rate) heatmap alongside PDR
  - Pre/post-CSRD comparison updated to include MDR
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scope3_thesis.data_processor import S3_CATEGORIES, CATEGORY_LABELS

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 11,
    'axes.labelsize': 12, 'axes.titlesize': 14,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'figure.dpi': 150
})

CAT_SHORT = {c: f"C{c[1:]}" for c in S3_CATEGORIES}   # c1 → C1


def run_descriptive_analysis(df, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    print("Running descriptive analysis...")

    sector_df = df[df["sics_sector"] != "Information Not Available"].copy()

    _heatmap_relevancy(sector_df, output_dir)
    _heatmap_4state(sector_df, output_dir)
    _heatmap_primary_data(sector_df, output_dir)
    _heatmap_mdr(sector_df, output_dir)
    _sdqi_trends(df, output_dir)
    _policy_shift(df, output_dir)
    _math_discrepancies(df, output_dir)
    _strategic_omissions_tiered(df, output_dir)
    _persistence_analysis(df, output_dir)

    print("Descriptive analysis complete — outputs saved to", output_dir)


# -----------------------------------------------------------------------
# Existing heatmap: share marked Relevant (backward compatible)
# -----------------------------------------------------------------------
def _heatmap_relevancy(sector_df, output_dir):
    rel_matrix = pd.DataFrame(index=sector_df["sics_sector"].unique(),
                               columns=[CAT_SHORT[c] for c in S3_CATEGORIES])
    for i, c in enumerate(S3_CATEGORIES):
        col = f"s3_ghgp_{c}_emissions_relevancy"
        share = sector_df.groupby("sics_sector")[col].apply(
            lambda x: (x == "Relevant").mean())
        rel_matrix[CAT_SHORT[c]] = share

    rel_matrix = rel_matrix.astype(float)
    plt.figure(figsize=(14, 8))
    sns.heatmap(rel_matrix * 100, annot=True, fmt=".0f", cmap="YlGnBu",
                cbar_kws={'label': 'Relevance Rate (%)'}, linewidths=0.4)
    plt.title("Scope 3 Category Relevance Rate by Sector (%)", pad=20, weight='bold')
    plt.xlabel("Scope 3 Category (GHG Protocol)")
    plt.ylabel("SICS Sector")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "heatmap_relevancy_sector.png"), dpi=300)
    plt.close()
    rel_matrix.to_csv(os.path.join(output_dir, "relevancy_matrix_sector.csv"))


# -----------------------------------------------------------------------
# NEW: 4-state relevancy breakdown by sector × category
# Shows shares for each of Relevant / Not relevant / Not evaluated / Silent
# -----------------------------------------------------------------------
def _heatmap_4state(sector_df, output_dir):
    states = ["Relevant", "Not relevant", "Not evaluated", "Silent"]
    for state in states:
        mat = pd.DataFrame(index=sector_df["sics_sector"].unique(),
                           columns=[CAT_SHORT[c] for c in S3_CATEGORIES])
        for c in S3_CATEGORIES:
            col = f"s3_ghgp_{c}_emissions_relevancy"
            share = sector_df.groupby("sics_sector")[col].apply(
                lambda x, s=state: (x == s).mean())
            mat[CAT_SHORT[c]] = share
        mat = mat.astype(float)
        mat.to_csv(os.path.join(output_dir,
                                f"relevancy_4state_{state.lower().replace(' ', '_')}_sector.csv"))

    # Save combined summary: for each sector×category, the dominant non-Relevant state
    omission_mat = pd.DataFrame(index=sector_df["sics_sector"].unique(),
                                columns=[CAT_SHORT[c] for c in S3_CATEGORIES])
    for c in S3_CATEGORIES:
        col = f"s3_ghgp_{c}_emissions_relevancy"
        for sector in sector_df["sics_sector"].unique():
            sub = sector_df[sector_df["sics_sector"] == sector][col]
            silent_rate   = (sub == "Silent").mean()
            noteval_rate  = (sub == "Not evaluated").mean()
            notrel_rate   = (sub == "Not relevant").mean()
            relevant_rate = (sub == "Relevant").mean()
            omission_mat.loc[sector, CAT_SHORT[c]] = round(
                (silent_rate + noteval_rate) * 100, 1)

    omission_mat = omission_mat.astype(float)
    plt.figure(figsize=(14, 8))
    sns.heatmap(omission_mat, annot=True, fmt=".0f", cmap="Reds",
                cbar_kws={'label': 'Silent + Not evaluated Rate (%)'},
                linewidths=0.4, vmin=0, vmax=70)
    plt.title("Opaque Omission Rate by Sector & Category\n"
              "(% companies giving Silent or 'Not evaluated' response)", pad=15, weight='bold')
    plt.xlabel("Scope 3 Category")
    plt.ylabel("Sector")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "heatmap_omission_opaque.png"), dpi=300)
    plt.close()
    omission_mat.to_csv(os.path.join(output_dir, "omission_opaque_matrix_sector.csv"))


# -----------------------------------------------------------------------
# Primary data heatmap (unchanged)
# -----------------------------------------------------------------------
def _heatmap_primary_data(sector_df, output_dir):
    pd_matrix = pd.DataFrame(index=sector_df["sics_sector"].unique(),
                              columns=[CAT_SHORT[c] for c in S3_CATEGORIES])
    for c in S3_CATEGORIES:
        rel_col = f"s3_ghgp_{c}_emissions_relevancy"
        pd_col  = f"s3_ghgp_{c}_emissions_primary_data"
        mean_pd = sector_df[sector_df[rel_col] == "Relevant"].groupby("sics_sector")[pd_col].mean()
        pd_matrix[CAT_SHORT[c]] = mean_pd

    pd_matrix = pd_matrix.astype(float).fillna(0.0)
    plt.figure(figsize=(14, 8))
    sns.heatmap(pd_matrix, annot=True, fmt=".0f", cmap="Purples",
                cbar_kws={'label': 'Avg Primary Data (%)'}, linewidths=0.4)
    plt.title("Average Primary Data Share in Reported Scope 3 Emissions (%)", pad=20, weight='bold')
    plt.xlabel("Scope 3 Category")
    plt.ylabel("Sector")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "heatmap_primary_data_sector.png"), dpi=300)
    plt.close()
    pd_matrix.to_csv(os.path.join(output_dir, "primary_data_matrix_sector.csv"))


# -----------------------------------------------------------------------
# NEW: Method Disclosure Rate heatmap (Module 5)
# -----------------------------------------------------------------------
def _heatmap_mdr(sector_df, output_dir):
    mdr_matrix = pd.DataFrame(index=sector_df["sics_sector"].unique(),
                               columns=[CAT_SHORT[c] for c in S3_CATEGORIES])
    for c in S3_CATEGORIES:
        rel_col  = f"s3_ghgp_{c}_emissions_relevancy"
        meth_col = f"disclose_s3_ghgp_{c}_emissions_method_bool"
        if meth_col not in sector_df.columns:
            mdr_matrix[CAT_SHORT[c]] = 0.0
            continue
        relevant_sub = sector_df[sector_df[rel_col] == "Relevant"]
        mdr = relevant_sub.groupby("sics_sector")[meth_col].apply(
            lambda x: x.astype(bool).mean() * 100)
        mdr_matrix[CAT_SHORT[c]] = mdr

    mdr_matrix = mdr_matrix.astype(float).fillna(0.0)
    plt.figure(figsize=(14, 8))
    sns.heatmap(mdr_matrix, annot=True, fmt=".0f", cmap="Blues",
                cbar_kws={'label': 'Method Disclosure Rate (%)'}, linewidths=0.4)
    plt.title("Method Disclosure Rate by Sector & Category\n"
              "(% of relevant disclosures that describe their calculation methodology)",
              pad=15, weight='bold')
    plt.xlabel("Scope 3 Category")
    plt.ylabel("Sector")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "heatmap_mdr_sector.png"), dpi=300)
    plt.close()
    mdr_matrix.to_csv(os.path.join(output_dir, "mdr_matrix_sector.csv"))


# -----------------------------------------------------------------------
# SDQI trends by region
# -----------------------------------------------------------------------
def _sdqi_trends(df, output_dir):
    trends = df.groupby(["reporting_year", "region"])["sdqi_basic"].mean().unstack()
    plt.figure(figsize=(10, 6))
    for region in trends.columns:
        plt.plot(trends.index, trends[region], marker='o', linewidth=2, label=region)
    plt.title("Scope 3 SDQI Trends by Region (2018–2023)", pad=15, weight='bold')
    plt.xlabel("Reporting Year")
    plt.ylabel("Mean SDQI (Basic)")
    plt.legend(title="Region")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sdqi_trends_time.png"), dpi=300)
    plt.close()
    trends.to_csv(os.path.join(output_dir, "sdqi_trends_region.csv"))


# -----------------------------------------------------------------------
# Pre/post 2022 (CSRD) policy comparison — now includes MDR
# -----------------------------------------------------------------------
def _policy_shift(df, output_dir):
    rows = []
    groups = [
        ("Global",          df),
        ("EU-27 Only",      df[df["is_eu"]]),
        ("Non-EU Only",     df[~df["is_eu"]]),
        ("United States",   df[df["jurisdiction_clean"] == "United States"]),
    ]
    for name, sub in groups:
        pre  = sub[sub["reporting_year"] < 2022]
        post = sub[sub["reporting_year"] >= 2022]
        rows.append({
            "Group":                  name,
            "Pre-2022 Count":         len(pre),
            "Post-2022 Count":        len(post),
            "Pre-2022 SDQI":          pre["sdqi_basic"].mean(),
            "Post-2022 SDQI":         post["sdqi_basic"].mean(),
            "SDQI Change":            post["sdqi_basic"].mean() - pre["sdqi_basic"].mean(),
            "Pre-2022 Completeness":  pre["completeness_score"].mean(),
            "Post-2022 Completeness": post["completeness_score"].mean(),
            "Completeness Change":    post["completeness_score"].mean() - pre["completeness_score"].mean(),
            "Pre-2022 Primary Data":  pre["primary_data_ratio"].mean(),
            "Post-2022 Primary Data": post["primary_data_ratio"].mean(),
            "Primary Data Change":    post["primary_data_ratio"].mean() - pre["primary_data_ratio"].mean(),
            "Pre-2022 MDR":           pre["method_disclosure_rate"].mean() if "method_disclosure_rate" in pre.columns else np.nan,
            "Post-2022 MDR":          post["method_disclosure_rate"].mean() if "method_disclosure_rate" in post.columns else np.nan,
            "MDR Change":             (post["method_disclosure_rate"].mean() - pre["method_disclosure_rate"].mean())
                                       if "method_disclosure_rate" in df.columns else np.nan,
        })
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, "policy_shift_comparison.csv"), index=False)


# -----------------------------------------------------------------------
# Mathematical discrepancy mapping
# -----------------------------------------------------------------------
def _math_discrepancies(df, output_dir):
    df = df.copy()
    df["has_discrepancy"] = (df["s3_math_discrepancy"] >= 1.0).astype(int)

    disc_sector = df.groupby("sics_sector").agg(
        total_disclosures   =("nz_id", "count"),
        discrepancy_count   =("has_discrepancy", "sum"),
        mean_abs_discrepancy=("s3_math_discrepancy", "mean"),
        median_abs_discrepancy=("s3_math_discrepancy", "median"),
        max_abs_discrepancy =("s3_math_discrepancy", "max"),
    )
    disc_sector["discrepancy_rate (%)"] = (
        disc_sector["discrepancy_count"] / disc_sector["total_disclosures"] * 100)
    disc_sector.to_csv(os.path.join(output_dir, "math_discrepancies_sector.csv"))

    disc_juris = df.groupby("jurisdiction_clean").agg(
        total_disclosures   =("nz_id", "count"),
        discrepancy_count   =("has_discrepancy", "sum"),
        mean_abs_discrepancy=("s3_math_discrepancy", "mean"),
        max_abs_discrepancy =("s3_math_discrepancy", "max"),
    )
    disc_juris["discrepancy_rate (%)"] = (
        disc_juris["discrepancy_count"] / disc_juris["total_disclosures"] * 100)
    disc_juris[disc_juris["total_disclosures"] >= 100].sort_values(
        "discrepancy_rate (%)", ascending=False
    ).to_csv(os.path.join(output_dir, "math_discrepancies_jurisdiction.csv"))


# -----------------------------------------------------------------------
# UPDATED strategic omissions — tier-based using 4 relevancy states
# -----------------------------------------------------------------------
def _strategic_omissions_tiered(df, output_dir):
    """
    Builds a tiered omission profile:
      Tier-1 (Silent)        — gave NO response to a high-materiality category
      Tier-2 (Not evaluated) — explicitly said they did not evaluate it
      Active dismissal        — said 'Not relevant' (active assessment, debatable)

    High-materiality threshold: category is relevant for >= 50 % of sector peers.
    """
    rows = []
    for sector, sub in df.groupby("sics_sector"):
        if sector == "Information Not Available":
            continue
        high_rel_cats = [
            c for c in S3_CATEGORIES
            if (sub[f"s3_ghgp_{c}_emissions_relevancy"] == "Relevant").mean() >= 0.50
        ]
        if not high_rel_cats:
            continue
        for _, row in sub.iterrows():
            t1 = t2 = active_dismiss = 0
            for c in high_rel_cats:
                val = row[f"s3_ghgp_{c}_emissions_relevancy"]
                if val == "Silent":
                    t1 += 1
                elif val == "Not evaluated":
                    t2 += 1
                elif val == "Not relevant":
                    active_dismiss += 1
            n = len(high_rel_cats)
            rows.append({
                "nz_id":                    row["nz_id"],
                "company_name":             row["company_name"],
                "sics_sector":              sector,
                "reporting_year":           row["reporting_year"],
                "high_materiality_cats":    n,
                "tier1_silent_count":       t1,
                "tier2_not_evaluated_count":t2,
                "active_dismissal_count":   active_dismiss,
                "opaque_omission_rate_pct": (t1 + t2) / n * 100,
            })

    if not rows:
        return

    omissions_df = pd.DataFrame(rows)
    summary = omissions_df.groupby("sics_sector").agg(
        mean_tier1_silent         =("tier1_silent_count",        "mean"),
        mean_tier2_not_evaluated  =("tier2_not_evaluated_count", "mean"),
        mean_opaque_omission_rate =("opaque_omission_rate_pct",  "mean"),
        high_omission_firms       =("opaque_omission_rate_pct",  lambda x: (x > 30).sum()),
        total_firms               =("nz_id",                     "nunique"),
    ).round(2)
    summary.to_csv(os.path.join(output_dir, "strategic_omissions_tiered_sector.csv"))

    # Also keep legacy output for backward compat
    legacy = omissions_df.groupby("sics_sector").agg(
        mean_omitted_count   =("tier1_silent_count", lambda x: x.mean() + omissions_df.loc[x.index, "tier2_not_evaluated_count"].mean()),
        mean_omission_rate   =("opaque_omission_rate_pct", "mean"),
        high_omission_firms_count=("opaque_omission_rate_pct", lambda x: (x > 30).sum()),
    ).round(2)
    legacy.to_csv(os.path.join(output_dir, "strategic_omissions_sector.csv"))


# -----------------------------------------------------------------------
# NEW: Persistence analysis (Module 7)
# -----------------------------------------------------------------------
def _persistence_analysis(df, output_dir):
    """
    For each company × high-materiality category, count consecutive years
    of 'Silent' or 'Not evaluated' response.

    A company with 3+ consecutive years of silence on a material category is
    classified as a 'systematic omitter' — this is hard to explain as a one-off
    oversight and strongly suggests deliberate non-disclosure.
    """
    # Identify high-materiality categories per sector (>= 50 % relevant rate)
    high_mat = {}
    for sector, sub in df.groupby("sics_sector"):
        if sector == "Information Not Available":
            continue
        high_mat[sector] = [
            c for c in S3_CATEGORIES
            if (sub[f"s3_ghgp_{c}_emissions_relevancy"] == "Relevant").mean() >= 0.50
        ]

    rows = []
    for sector, mat_cats in high_mat.items():
        if not mat_cats:
            continue
        sec_df = df[df["sics_sector"] == sector].copy()

        for company, comp_df in sec_df.groupby("nz_id"):
            comp_df = comp_df.sort_values("reporting_year")
            n_years = len(comp_df)
            if n_years < 2:
                continue

            for c in mat_cats:
                col = f"s3_ghgp_{c}_emissions_relevancy"
                vals = comp_df[col].tolist()
                years = comp_df["reporting_year"].tolist()

                # Count max consecutive years of opaque omission (Silent or Not evaluated)
                is_opaque = [v in ("Silent", "Not evaluated") for v in vals]
                max_consec = _max_consecutive_true(is_opaque)
                total_opaque = sum(is_opaque)

                if total_opaque > 0:
                    rows.append({
                        "nz_id":            company,
                        "sics_sector":      sector,
                        "category":         c,
                        "n_years_in_panel": n_years,
                        "n_opaque_years":   total_opaque,
                        "max_consecutive_opaque": max_consec,
                        "is_systematic_omitter": max_consec >= 3,
                        "first_year":       years[0],
                        "last_year":        years[-1],
                    })

    if not rows:
        return

    persist_df = pd.DataFrame(rows)

    # Sector summary
    sector_summary = persist_df.groupby("sics_sector").agg(
        total_company_category_pairs   =("nz_id",                    "count"),
        systematic_omitters            =("is_systematic_omitter",    "sum"),
        pct_systematic                 =("is_systematic_omitter",    lambda x: x.mean() * 100),
        avg_max_consecutive_opaque_yrs =("max_consecutive_opaque",   "mean"),
        avg_opaque_years               =("n_opaque_years",           "mean"),
    ).round(2)
    sector_summary.to_csv(os.path.join(output_dir, "persistence_scores_sector.csv"))

    # Category summary across all sectors
    cat_summary = persist_df.groupby("category").agg(
        total_pairs        =("nz_id",                    "count"),
        systematic_omitters=("is_systematic_omitter",    "sum"),
        pct_systematic     =("is_systematic_omitter",    lambda x: x.mean() * 100),
    ).round(2)
    cat_summary.index = [CATEGORY_LABELS.get(c, c) for c in cat_summary.index]
    cat_summary.to_csv(os.path.join(output_dir, "persistence_scores_category.csv"))

    # Distribution plot: histogram of max consecutive opaque years
    plt.figure(figsize=(9, 5))
    plt.hist(persist_df["max_consecutive_opaque"], bins=range(0, 8),
             color="#1e3d59", edgecolor="white", alpha=0.85, align='left')
    plt.axvline(x=3, color="#ff6f61", linestyle="--", linewidth=2,
                label="3-year threshold (systematic omitter)")
    plt.title("Distribution of Maximum Consecutive Opaque Omission Years\n"
              "(per company × high-materiality category)", pad=12, weight='bold')
    plt.xlabel("Max consecutive years with Silent / Not evaluated response")
    plt.ylabel("Number of company–category pairs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "persistence_distribution.png"), dpi=300)
    plt.close()


def _max_consecutive_true(bool_list):
    """Return the maximum run-length of True values in a boolean list."""
    max_run = current_run = 0
    for v in bool_list:
        if v:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    return max_run


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from scope3_thesis.data_processor import load_and_preprocess_data
    from scope3_thesis.sdqi_calculator import calculate_sdqi_scores
    csv_file = "cdu_global_all_data.csv"
    if os.path.exists(csv_file):
        df = load_and_preprocess_data(csv_file)
        df = calculate_sdqi_scores(df)
        run_descriptive_analysis(df)
