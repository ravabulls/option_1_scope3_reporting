import pandas as pd
import numpy as np
import os
import re

S3_CATEGORIES = [f"c{i}" for i in range(1, 16)]

EU_COUNTRIES = {
    "Austria", "Belgium", "Bulgaria", "Croatia", "Republic of Cyprus", "Cyprus",
    "Czechia", "Czech Republic", "Denmark", "Estonia", "Finland", "France",
    "Germany", "Greece", "Hungary", "Ireland", "Italy", "Latvia", "Lithuania",
    "Luxembourg", "Malta", "Netherlands", "Poland", "Portugal", "Romania",
    "Slovakia", "Slovenia", "Spain", "Sweden"
}

EUROPE_COUNTRIES = EU_COUNTRIES.union({
    "United Kingdom of Great Britain and Northern Ireland", "United Kingdom",
    "Switzerland", "Norway", "Iceland", "Liechtenstein"
})

# GHG Protocol Scope 3 category labels for display
CATEGORY_LABELS = {
    "c1":  "C1: Purchased Goods & Services",
    "c2":  "C2: Capital Goods",
    "c3":  "C3: Fuel & Energy Activities",
    "c4":  "C4: Upstream Transport & Distribution",
    "c5":  "C5: Waste Generated in Operations",
    "c6":  "C6: Business Travel",
    "c7":  "C7: Employee Commuting",
    "c8":  "C8: Upstream Leased Assets",
    "c9":  "C9: Downstream Transport & Distribution",
    "c10": "C10: Processing of Sold Products",
    "c11": "C11: Use of Sold Products",
    "c12": "C12: End-of-Life Treatment",
    "c13": "C13: Downstream Leased Assets",
    "c14": "C14: Franchises",
    "c15": "C15: Investments",
}

# Four relevancy states (ordered from most transparent to most opaque)
RELEVANCY_STATES = ["Relevant", "Not relevant", "Not evaluated", "Silent"]


def clean_jurisdiction_name(name):
    if not isinstance(name, str):
        return "Unknown"
    name = name.strip()
    name = re.sub(r'T.rkiye', 'Türkiye', name)
    name = re.sub(r'T�rkiye', 'Türkiye', name)
    if name == "United States of America":
        return "United States"
    if name == "United Kingdom of Great Britain and Northern Ireland":
        return "United Kingdom"
    return name


def get_required_columns():
    cols = [
        "nz_id", "company_name", "reporting_year", "jurisdiction",
        "org_boundary_approach", "disclose_verif_emissions_bool",
        "sics_sector", "sics_industry", "sics_sub_sector",
        "total_s3_emissions_ghg", "total_s1_emissions_ghg",
        "total_s2_lb_emissions_ghg", "total_s2_mb_emissions_ghg",
        "total_s3_other_emissions_ghg", "s3_other_emissions_relevancy",
        "s3_other_emissions_primary_data"
    ]
    for c in S3_CATEGORIES:
        cols.append(f"s3_ghgp_{c}_emissions_relevancy")
        cols.append(f"s3_ghgp_{c}_emissions_primary_data")
        cols.append(f"total_s3_ghgp_{c}_emissions_ghg")
        cols.append(f"disclose_s3_ghgp_{c}_emissions_method_bool")
    return cols


def load_and_preprocess_data(csv_path):
    """
    Loads and standardizes the CDP Scope 3 dataset.

    Key design decision — four-state relevancy:
    CDP disclosures have four distinct relevancy states for each category.
    We preserve all four rather than collapsing NaN into 'Not evaluated',
    because the distinction carries meaningful analytical signal:
      'Relevant'      — category assessed as material; emissions reported
      'Not relevant'  — category actively assessed and dismissed
      'Not evaluated' — company explicitly acknowledged it did not evaluate
      'Silent'        — company gave no response at all (strongest omission signal)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found at {csv_path}")

    required_cols = get_required_columns()
    # Only keep columns that exist in the file (method_bool may be missing on older exports)
    try:
        header = pd.read_csv(csv_path, nrows=0, encoding='utf-8').columns.tolist()
    except UnicodeDecodeError:
        header = pd.read_csv(csv_path, nrows=0, encoding='latin1').columns.tolist()
    usecols = [c for c in required_cols if c in header]

    try:
        df = pd.read_csv(csv_path, usecols=usecols, encoding='utf-8', low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, usecols=usecols, encoding='latin1', low_memory=False)

    # --- Jurisdiction & region flags ---
    df["jurisdiction_clean"] = df["jurisdiction"].apply(clean_jurisdiction_name)
    df["is_eu"] = df["jurisdiction_clean"].isin(EU_COUNTRIES)
    df["is_europe"] = df["jurisdiction_clean"].isin(EUROPE_COUNTRIES)

    # --- Organisational boundary ---
    boundary_mapping = {
        "Operational control": "Operational Control",
        "Financial control":   "Financial Control",
        "Equity share":        "Equity Share",
        "Other":               "Other / Hybrid",
        "Approach not disclosed": "Not Disclosed",
    }
    df["boundary_approach_clean"] = (
        df["org_boundary_approach"]
        .fillna("Not Disclosed")
        .map(boundary_mapping)
        .fillna("Other / Hybrid")
    )

    # --- Verification flag ---
    df["emissions_verified"] = df["disclose_verif_emissions_bool"].fillna(False).astype(bool)

    # --- Reporting year ---
    df["reporting_year"] = pd.to_numeric(df["reporting_year"], errors='coerce').fillna(2022).astype(int)

    # --- Per-category fields ---
    for c in S3_CATEGORIES + ["other"]:
        prefix = f"s3_ghgp_{c}" if c != "other" else "s3_other"

        # Primary data percentage
        pdata_col = f"{prefix}_emissions_primary_data"
        if pdata_col in df.columns:
            df[pdata_col] = pd.to_numeric(df[pdata_col], errors='coerce').clip(0.0, 100.0)

        # Category emissions
        em_col = f"total_s3_ghgp_{c}_emissions_ghg" if c != "other" else "total_s3_other_emissions_ghg"
        if em_col in df.columns:
            df[em_col] = pd.to_numeric(df[em_col], errors='coerce').clip(lower=0.0)

        # Relevancy — FOUR-STATE PRESERVATION (critical for omission analysis)
        # NaN in the raw data means the company never responded → labelled "Silent"
        # "Not evaluated" means company explicitly acknowledged non-evaluation
        # These two states must NOT be merged.
        rel_col = f"{prefix}_emissions_relevancy"
        if rel_col in df.columns:
            df[rel_col] = (
                df[rel_col]
                .astype(str)
                .str.strip()
                .replace({"nan": np.nan, "None": np.nan})
            )
            # Map NaN → "Silent" (distinct from explicit "Not evaluated")
            df[rel_col] = df[rel_col].fillna("Silent")

        # Method disclosure boolean (whether company described calculation methodology)
        if c != "other":
            meth_col = f"disclose_s3_ghgp_{c}_emissions_method_bool"
            if meth_col in df.columns:
                # Map to int (1/0) first to avoid pandas FutureWarning about
                # object-dtype downcasting on .fillna, then cast to bool.
                df[meth_col] = (
                    df[meth_col]
                    .map({True: 1, False: 0, "True": 1, "False": 0})
                    .fillna(0)
                    .astype(int)
                    .astype(bool)
                )
            else:
                # Column absent in export — default to False
                df[meth_col] = False

    # --- Scope 1 / 2 / 3 totals ---
    df["total_s1_emissions_ghg"] = pd.to_numeric(df["total_s1_emissions_ghg"], errors='coerce').clip(lower=0.0).fillna(0.0)
    df["total_s2_lb_emissions_ghg"] = pd.to_numeric(df["total_s2_lb_emissions_ghg"], errors='coerce').clip(lower=0.0).fillna(0.0)
    df["total_s2_mb_emissions_ghg"] = pd.to_numeric(df["total_s2_mb_emissions_ghg"], errors='coerce').clip(lower=0.0).fillna(0.0)
    df["total_s2_emissions_best"] = df[["total_s2_lb_emissions_ghg", "total_s2_mb_emissions_ghg"]].max(axis=1)
    df["total_s3_emissions_ghg"] = pd.to_numeric(df["total_s3_emissions_ghg"], errors='coerce').clip(lower=0.0)

    # --- Calculated sum of categories (for discrepancy check) ---
    cat_em_cols = [f"total_s3_ghgp_c{i}_emissions_ghg" for i in range(1, 16)] + ["total_s3_other_emissions_ghg"]
    df["total_s3_calculated_sum_ghg"] = df[cat_em_cols].sum(axis=1, min_count=1).fillna(0.0)
    df["total_s3_emissions_best"] = df["total_s3_emissions_ghg"].fillna(df["total_s3_calculated_sum_ghg"])

    # --- Mathematical discrepancy ---
    df["s3_math_discrepancy"] = (
        df["total_s3_calculated_sum_ghg"] - df["total_s3_emissions_ghg"].fillna(0.0)
    ).abs()

    # --- Company size proxy: log10(Scope1 + Scope2 + 1) ---
    df["company_size_proxy"] = np.log10(
        df["total_s1_emissions_ghg"] + df["total_s2_emissions_best"] + 1.0
    )

    # --- Policy dummy: post-CSRD announcement (year >= 2022) ---
    df["post_csrd_announcement"] = (df["reporting_year"] >= 2022).astype(int)

    # --- Region assignment (used in regression) ---
    def _assign_region(row):
        if row["is_eu"]:
            return "EU-27"
        jc = row["jurisdiction_clean"]
        if jc == "United States":
            return "United States"
        if jc == "China":
            return "China"
        if jc == "Japan":
            return "Japan"
        return "Rest_of_World"

    df["region"] = df.apply(_assign_region, axis=1)

    return df


if __name__ == "__main__":
    import sys
    csv_file = "cdu_global_all_data.csv"
    if os.path.exists(csv_file):
        df = load_and_preprocess_data(csv_file)
        print("Shape:", df.shape)
        print("EU count:", df["is_eu"].sum())
        print("Verification rate:", df["emissions_verified"].mean().round(3))
        print("\nC1 relevancy distribution:")
        print(df["s3_ghgp_c1_emissions_relevancy"].value_counts(dropna=False))
    else:
        print("cdu_global_all_data.csv not found.")
