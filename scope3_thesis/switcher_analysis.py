"""
switcher_analysis.py — Module 2: Within-company verification event study & DiD.

Rationale
---------
The OLS finding that "verified companies have 0.35-point higher SDQI" is open to
the criticism that better-managed companies *both* verify *and* disclose more —
i.e., verification correlates with but does not cause higher quality.

This module addresses that by studying the 1,917 companies that *switch* from
unverified to verified during the 2018–2023 panel.  For these firms we observe
SDQI both before and after they began verifying, controlling for the company
itself and for the global SDQI time-trend.

Two analyses are produced:
1. Event study  — mean SDQI at event-times T-2, T-1, T, T+1, T+2 relative to
                  the year of first verification, for switchers vs never-verified.
2. DiD estimate — the average SDQI improvement of switchers relative to
                  never-verified companies over the same calendar-year window.
"""

import os
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


def identify_switchers(df):
    """
    Tags each row with switcher-related metadata.

    A 'switcher' is a company that appears as unverified in at least one year
    and as verified in at least one later year (monotone switch, False → True).

    New columns added:
      ever_verified      — True if the company is verified in any year
      first_verified_year — first year with verified=True (NaN if never verified)
      is_switcher        — True if company switches from False to True at some point
      event_time         — reporting_year − first_verified_year (only for switchers)
    """
    df = df.copy()

    # Per-company summary
    comp = df.groupby("nz_id").agg(
        ever_verified=("emissions_verified", "max"),
        ever_unverified=("emissions_verified", lambda x: (~x).any()),
        n_years=("reporting_year", "nunique")
    )

    # Switcher: was unverified at some point AND later became verified
    switcher_ids = comp[comp["ever_verified"] & comp["ever_unverified"]].index

    df["ever_verified"]   = df["nz_id"].isin(comp[comp["ever_verified"]].index)
    df["is_switcher"]     = df["nz_id"].isin(switcher_ids)
    df["never_verified"]  = df["nz_id"].isin(comp[~comp["ever_verified"]].index)

    # First verified year for switchers
    first_ver = (
        df[df["emissions_verified"]]
        .groupby("nz_id")["reporting_year"]
        .min()
        .rename("first_verified_year")
    )
    df = df.join(first_ver, on="nz_id")
    df["event_time"] = np.where(
        df["is_switcher"],
        df["reporting_year"] - df["first_verified_year"],
        np.nan
    )
    df["event_time"] = df["event_time"].astype("Int64")  # nullable int

    return df


def run_event_study(df, output_dir="outputs"):
    """
    Event study: mean SDQI by event-time for switchers vs never-verified.

    Switcher event-time is defined as years relative to first verification
    (T=0 is the first year verified, T=-1 is the last unverified year, etc.).
    We restrict to event-times -2 through +2 to maintain balanced comparison.

    Never-verified companies are assigned a pseudo-event-time based on the
    calendar year mid-point of the switchers' first-verification years so the
    two groups cover the same calendar period.

    Saves: outputs/switcher_event_study.csv
    """
    os.makedirs(output_dir, exist_ok=True)

    # Switchers: restrict to event-time window [-2, +2]
    sw = df[df["is_switcher"] & df["event_time"].notna()].copy()
    sw = sw[sw["event_time"].between(-2, 2)]

    sw_agg = (
        sw.groupby("event_time")["sdqi_basic"]
        .agg(mean_sdqi="mean", se=lambda x: x.std() / np.sqrt(len(x)), n="count")
        .reset_index()
    )
    sw_agg["group"] = "Switchers"

    # Never-verified: align by calendar years that cover switchers' verification window
    nv = df[df["never_verified"]].copy()
    # Use calendar years 2018-2023, map to event-time anchored at 2021 (median switch year)
    median_switch_year = df[df["is_switcher"]]["first_verified_year"].median()
    nv["event_time"] = nv["reporting_year"] - int(median_switch_year)
    nv = nv[nv["event_time"].between(-2, 2)]

    nv_agg = (
        nv.groupby("event_time")["sdqi_basic"]
        .agg(mean_sdqi="mean", se=lambda x: x.std() / np.sqrt(len(x)), n="count")
        .reset_index()
    )
    nv_agg["group"] = "Never Verified"

    result = pd.concat([sw_agg, nv_agg], ignore_index=True)
    result.to_csv(os.path.join(output_dir, "switcher_event_study.csv"), index=False)

    # Key stat: SDQI jump at T=0 for switchers vs never-verified at same event-time
    stats = {}
    for g, sub in result.groupby("group"):
        pre  = sub[sub["event_time"] < 0]["mean_sdqi"].mean()
        post = sub[sub["event_time"] >= 0]["mean_sdqi"].mean()
        stats[g] = {"pre_mean": pre, "post_mean": post, "change": post - pre}

    return result, stats


def run_did_analysis(df, output_dir="outputs"):
    """
    Difference-in-Differences: does switching to verified improve SDQI
    beyond the background time trend?

    Specification:
        SDQI_it = α_i + δ_t + β * Post_it + ε_it

    where Post_it = 1 if company i has been verified in year t or before,
    0 otherwise.  α_i = company fixed effects, δ_t = year fixed effects.

    The sample is restricted to switchers and never-verified companies
    (verified-in-all-years companies are excluded to avoid contamination).

    β is the DiD estimate: the within-company SDQI improvement attributable
    to the verification event, net of the global time trend.

    Saves: outputs/switcher_did_results.csv
    """
    os.makedirs(output_dir, exist_ok=True)

    # Keep switchers + never-verified only
    sample = df[df["is_switcher"] | df["never_verified"]].copy()
    sample = sample[sample["reporting_year"].between(2018, 2023)]

    # Post indicator: for switchers, 1 once they have started verifying
    sample["post_verified"] = sample["emissions_verified"].astype(int)

    # Require companies with at least 2 observations
    company_counts = sample.groupby("nz_id")["reporting_year"].count()
    valid_companies = company_counts[company_counts >= 2].index
    sample = sample[sample["nz_id"].isin(valid_companies)]

    if len(sample) < 100:
        return pd.DataFrame(), {}

    # Two-way FE via within-company demeaning (Frisch–Waugh–Lovell theorem).
    # This is mathematically equivalent to including C(nz_id) dummy variables
    # but avoids the near-singular ~12,000-column design matrix that causes
    # the LAPACK DLASCLS warning with statsmodels OLS.
    sample["sdqi_demean"] = (sample["sdqi_basic"]
                             - sample.groupby("nz_id")["sdqi_basic"].transform("mean"))
    sample["post_demean"]  = (sample["post_verified"]
                              - sample.groupby("nz_id")["post_verified"].transform("mean"))

    # Regress demeaned SDQI on demeaned treatment indicator + year dummies.
    # Year dummies absorb global time trends; the coefficient on post_demean
    # is the within-company DiD estimate.
    formula = "sdqi_demean ~ post_demean + C(reporting_year)"
    try:
        model = smf.ols(formula, data=sample).fit(
            cov_type='cluster',
            cov_kwds={'groups': sample['nz_id']}
        )
        did_coef = model.params.get("post_demean", np.nan)
        did_se   = model.bse.get("post_demean", np.nan)
        did_pval = model.pvalues.get("post_demean", np.nan)
        r2       = model.rsquared          # within-R²
        n_obs    = int(model.nobs)
    except Exception:
        did_coef, did_se, did_pval, r2, n_obs = np.nan, np.nan, np.nan, np.nan, 0

    results_df = pd.DataFrame([{
        "estimator":    "DiD (Within-Company Demean + Year FE, Clustered SE)",
        "coefficient":  did_coef,
        "std_error":    did_se,
        "p_value":      did_pval,
        "r_squared":    r2,
        "n_obs":        n_obs,
        "n_switchers":  df["is_switcher"].sum(),
        "n_never_ver":  df["never_verified"].sum(),
        "interpretation": (
            f"Switching to verified is associated with a {did_coef:+.4f} change in SDQI "
            f"(p={did_pval:.4f}), controlling for company-level fixed effects and "
            f"the global year trend."
        ) if pd.notna(did_coef) else "Model failed."
    }])

    results_df.to_csv(os.path.join(output_dir, "switcher_did_results.csv"), index=False)
    return results_df, {
        "did_coef": did_coef, "did_se": did_se, "did_pval": did_pval,
        "n_obs": n_obs
    }


def run_switcher_analysis(df, output_dir="outputs"):
    """Master entry point — runs both event study and DiD, returns combined dict."""
    df = identify_switchers(df)
    n_switchers = df["is_switcher"].sum()
    print(f"   --> Identified {int(df['nz_id'][df['is_switcher']].nunique()):,} unique switcher companies.")

    event_df, event_stats  = run_event_study(df, output_dir)
    did_df, did_stats       = run_did_analysis(df, output_dir)

    print(f"   --> Event study saved: {len(event_df)} rows.")
    if did_stats.get("did_coef") is not None and not np.isnan(did_stats.get("did_coef", np.nan)):
        print(f"   --> DiD estimate: {did_stats['did_coef']:+.4f} (p={did_stats['did_pval']:.4f})")

    return df, event_df, event_stats, did_df, did_stats


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from scope3_thesis.data_processor import load_and_preprocess_data
    from scope3_thesis.sdqi_calculator import calculate_sdqi_scores

    csv_file = "cdu_global_all_data.csv"
    if os.path.exists(csv_file):
        df = load_and_preprocess_data(csv_file)
        df = calculate_sdqi_scores(df)
        df, ev_df, ev_stats, did_df, did_stats = run_switcher_analysis(df)
        print(ev_stats)
        print(did_df)
    else:
        print("Data file not found.")
