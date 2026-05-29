import pandas as pd
import numpy as np
from scope3_thesis.data_processor import S3_CATEGORIES


def compute_sector_materiality(df, threshold=0.30):
    """
    Returns a dict mapping sector → frozenset of material category codes.

    A category is material for a sector when >= threshold of firms in that
    sector mark it as 'Relevant'. The threshold of 30 % is the empirically
    conservative choice: it counts a category as sector-material only if
    nearly a third of peers actively report it.
    """
    global_rates = {
        c: (df[f"s3_ghgp_{c}_emissions_relevancy"] == "Relevant").mean()
        for c in S3_CATEGORIES
    }
    global_material = frozenset(c for c, r in global_rates.items() if r >= threshold)
    if len(global_material) < 3:
        global_material = frozenset(
            c for c, _ in sorted(global_rates.items(), key=lambda x: -x[1])[:5]
        )

    sector_materiality = {}
    for sector in df["sics_sector"].unique():
        if sector == "Information Not Available":
            sector_materiality[sector] = global_material
            continue
        sub = df[df["sics_sector"] == sector]
        mat = frozenset(
            c for c in S3_CATEGORIES
            if (sub[f"s3_ghgp_{c}_emissions_relevancy"] == "Relevant").mean() >= threshold
        )
        if len(mat) < 3:
            rates = {c: (sub[f"s3_ghgp_{c}_emissions_relevancy"] == "Relevant").mean()
                     for c in S3_CATEGORIES}
            mat = frozenset(c for c, _ in sorted(rates.items(), key=lambda x: -x[1])[:5])
        sector_materiality[sector] = mat

    return sector_materiality


def calculate_sdqi_scores(df, threshold=0.30, w_completeness=0.7, w_primary=0.3):
    """
    Calculates all SDQI variants and sub-components for every disclosure row.

    Components
    ----------
    completeness_score  : share of sector-material categories that are addressed
                          (either Relevant+emissions, or actively dismissed as Not relevant)
    primary_data_ratio  : average primary-data % across relevant-reported categories
    method_disclosure_rate : share of relevant-reported categories where the company
                             also described its calculation methodology
    consistency_fixed   : retention rate — share of last year's relevant categories
                          that remain relevant this year (does NOT penalise expansion)
    expansion_rate      : share of this year's relevant categories that are newly added

    SDQI variants
    -------------
    sdqi_basic    : 0.7 * completeness + 0.3 * PDR   (main specification, w_c/w_p args)
    sdqi_extended : 0.6 * completeness + 0.2 * PDR + 0.2 * MDR
    sdqi_w50_50   : 0.5 * completeness + 0.5 * PDR   (sensitivity)
    sdqi_w60_40   : 0.6 * completeness + 0.4 * PDR   (sensitivity)
    sdqi_w80_20   : 0.8 * completeness + 0.2 * PDR   (sensitivity)
    sdqi_panel    : 0.8 * sdqi_basic   + 0.2 * consistency_fixed
                    (falls back to sdqi_basic when no prior-year observation)
    """
    df = df.copy()
    sector_mat = compute_sector_materiality(df, threshold=threshold)

    # ------------------------------------------------------------------
    # Step 1 — Vectorised per-category binary flags
    # ------------------------------------------------------------------
    for c in S3_CATEGORIES:
        rel_col  = f"s3_ghgp_{c}_emissions_relevancy"
        em_col   = f"total_s3_ghgp_{c}_emissions_ghg"
        pd_col   = f"s3_ghgp_{c}_emissions_primary_data"
        meth_col = f"disclose_s3_ghgp_{c}_emissions_method_bool"

        is_relevant_reported = (df[rel_col] == "Relevant") & df[em_col].notna()

        # Completeness: Relevant+emissions OR active Not relevant dismissal
        df[f"_complete_{c}"] = is_relevant_reported | (df[rel_col] == "Not relevant")

        # Primary data value (NaN unless relevant+reported)
        df[f"_pd_val_{c}"] = np.where(is_relevant_reported, df[pd_col] / 100.0, np.nan)

        # Relevant flag for denominators
        df[f"_relevant_{c}"] = is_relevant_reported

        # Method bool masked to relevant-reported only
        if meth_col in df.columns:
            df[f"_method_{c}"] = df[meth_col].astype(bool) & is_relevant_reported
        else:
            df[f"_method_{c}"] = False

    # ------------------------------------------------------------------
    # Step 2 — Completeness score (sector-specific denominator)
    # ------------------------------------------------------------------
    completeness = pd.Series(0.0, index=df.index)
    for sector, mat_cats in sector_mat.items():
        mask = df["sics_sector"] == sector
        if not mask.any() or not mat_cats:
            continue
        comp_cols = [f"_complete_{c}" for c in mat_cats]
        completeness[mask] = df.loc[mask, comp_cols].sum(axis=1) / len(mat_cats)

    df["completeness_score"] = completeness

    # ------------------------------------------------------------------
    # Step 3 — Primary Data Ratio (PDR)
    # ------------------------------------------------------------------
    pd_cols  = [f"_pd_val_{c}" for c in S3_CATEGORIES]
    pd_sum   = df[pd_cols].sum(axis=1, min_count=1)
    pd_count = df[pd_cols].notna().sum(axis=1)
    df["primary_data_ratio"] = (pd_sum / pd_count.clip(lower=1)).where(pd_count > 0, 0.0)

    # ------------------------------------------------------------------
    # Step 4 — Method Disclosure Rate (MDR)
    # ------------------------------------------------------------------
    meth_cols = [f"_method_{c}" for c in S3_CATEGORIES]
    rel_cols  = [f"_relevant_{c}" for c in S3_CATEGORIES]
    mdr_num = df[meth_cols].sum(axis=1)
    mdr_den = df[rel_cols].sum(axis=1)
    df["method_disclosure_rate"] = (mdr_num / mdr_den.clip(lower=1)).where(mdr_den > 0, 0.0)

    # ------------------------------------------------------------------
    # Step 5 — SDQI variants (all bounded [0, 1])
    # ------------------------------------------------------------------
    cs  = df["completeness_score"]
    pdr = df["primary_data_ratio"]
    mdr = df["method_disclosure_rate"]

    df["sdqi_basic"]    = w_completeness * cs + w_primary * pdr
    df["sdqi_extended"] = 0.6 * cs + 0.2 * pdr + 0.2 * mdr
    df["sdqi_w50_50"]   = 0.5 * cs + 0.5 * pdr
    df["sdqi_w60_40"]   = 0.6 * cs + 0.4 * pdr
    df["sdqi_w80_20"]   = 0.8 * cs + 0.2 * pdr

    # ------------------------------------------------------------------
    # Step 6 — Year-over-year consistency (retention + expansion)
    # ------------------------------------------------------------------
    df_sorted = df.sort_values(["nz_id", "reporting_year"]).copy()

    # Build relevant-category frozensets per row
    rel_sets = [
        frozenset(c for c in S3_CATEGORIES
                  if row[f"s3_ghgp_{c}_emissions_relevancy"] == "Relevant")
        for _, row in df_sorted.iterrows()
    ]
    df_sorted["_rel_set"]     = rel_sets
    df_sorted["_prev_nz_id"]  = df_sorted["nz_id"].shift(1)
    df_sorted["_prev_year"]   = df_sorted["reporting_year"].shift(1)
    df_sorted["_prev_rel_set"] = df_sorted["_rel_set"].shift(1)

    consistency_vals = []
    retention_vals   = []
    expansion_vals   = []

    for _, row in df_sorted.iterrows():
        if (row["nz_id"] == row["_prev_nz_id"]
                and row["reporting_year"] == row["_prev_year"] + 1):

            s_t   = row["_rel_set"]
            s_t1  = row["_prev_rel_set"]
            union = s_t | s_t1

            jaccard   = len(s_t & s_t1) / len(union) if union else 1.0
            # Retention: % of prior-year relevant categories still relevant
            retention = len(s_t & s_t1) / len(s_t1) if s_t1 else 1.0
            # Expansion: newly added relative to this year's set
            expansion = len(s_t - s_t1) / max(len(s_t), 1)

            consistency_vals.append(jaccard)
            retention_vals.append(retention)
            expansion_vals.append(expansion)
        else:
            consistency_vals.append(np.nan)
            retention_vals.append(np.nan)
            expansion_vals.append(np.nan)

    df_sorted["consistency_score"]  = consistency_vals   # legacy Jaccard (kept for compat.)
    df_sorted["retention_rate"]     = retention_vals     # fixed: does not penalise expansion
    df_sorted["expansion_rate"]     = expansion_vals

    # SDQI_panel uses retention (not Jaccard) so expanding companies are not penalised
    df_sorted["consistency_fixed"]  = df_sorted["retention_rate"]
    df_sorted["sdqi_panel"] = np.where(
        df_sorted["consistency_fixed"].notna(),
        0.8 * df_sorted["sdqi_basic"] + 0.2 * df_sorted["consistency_fixed"],
        df_sorted["sdqi_basic"]
    )

    # Drop temp columns
    df_sorted.drop(columns=["_rel_set", "_prev_nz_id", "_prev_year", "_prev_rel_set"],
                   inplace=True)

    # Drop per-category helper columns (keep df clean)
    helper_prefixes = ("_complete_", "_pd_val_", "_relevant_", "_method_")
    drop_cols = [c for c in df_sorted.columns
                 if any(c.startswith(p) for p in helper_prefixes)]
    df_sorted.drop(columns=drop_cols, inplace=True)

    return df_sorted.loc[df.index]


if __name__ == "__main__":
    from scope3_thesis.data_processor import load_and_preprocess_data
    csv_file = "cdu_global_all_data.csv"
    import os
    if os.path.exists(csv_file):
        df = load_and_preprocess_data(csv_file)
        df = calculate_sdqi_scores(df)
        print("SDQI basic:    ", df["sdqi_basic"].describe().round(3))
        print("SDQI extended: ", df["sdqi_extended"].describe().round(3))
        print("MDR:           ", df["method_disclosure_rate"].describe().round(3))
        print("Retention:     ", df["retention_rate"].describe().round(3))
        print("Expansion:     ", df["expansion_rate"].describe().round(3))
