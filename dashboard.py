"""
dashboard.py — Interactive Streamlit dashboard for the Scope 3 Disclosure
Quality Index (SDQI) research project.

Navigation:
  Tab 1 — Overview & Key Findings
  Tab 2 — Category Relevance & Omissions
  Tab 3 — Method Quality (PDR + MDR)
  Tab 4 — Time Trends & CSRD Policy Effect
  Tab 5 — Verification Deep-Dive (Switcher DiD + PSM)
  Tab 6 — Regression Models
  Tab 7 — Company Explorer

Every chart has an ℹ️ popover button (top-right of the chart header) that
explains in plain language what the chart shows, how to read it, and what the
key takeaway is.  The app is designed to be fully understandable by a reader
with no prior knowledge of carbon accounting.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Scope 3 Disclosure Quality Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Hide Streamlit Cloud toolbar: fork / view-source / GitHub links ── */
#MainMenu                        { visibility: hidden !important; }
header[data-testid="stHeader"]   { visibility: hidden !important; height: 0 !important; }
[data-testid="stToolbar"]        { display: none !important; }
[data-testid="stDecoration"]     { display: none !important; }
[data-testid="stDeployButton"]   { display: none !important; }
footer                           { visibility: hidden !important; }

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.hero {
    background: linear-gradient(135deg, #1e3d59 0%, #17b978 100%);
    padding: 2.2rem 2.5rem; border-radius: 16px; color: white;
    margin-bottom: 1.5rem; box-shadow: 0 8px 32px rgba(31,38,135,0.15);
}
.hero h1 { font-size: 2.2rem; font-weight: 700; margin: 0; }
.hero p  { font-size: 1.05rem; opacity: 0.9; margin: 0.4rem 0 0; }
.kpi {
    background: white; border: 1px solid #e8ecef; border-radius: 12px;
    padding: 1.2rem; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.kpi-num   { font-size: 2rem; font-weight: 700; color: #1e3d59; }
.kpi-label { font-size: 0.82rem; color: #6c757d; text-transform: uppercase;
             letter-spacing: 0.06em; margin-top: 0.2rem; }
.finding-card {
    background: #f0faf5; border-left: 4px solid #17b978;
    padding: 1rem 1.2rem; border-radius: 8px; margin-bottom: 0.8rem;
}
.finding-card strong { color: #1e3d59; }
.section-intro {
    background: #f8f9fa; border-radius: 8px; padding: 0.9rem 1.1rem;
    font-size: 0.95rem; color: #444; margin-bottom: 1rem;
    border-left: 3px solid #1e3d59;
}
</style>
""", unsafe_allow_html=True)

OUTPUT_DIR = "outputs"
CAT_NAMES = {
    "C1": "C1: Purchased Goods & Services",  "C2": "C2: Capital Goods",
    "C3": "C3: Fuel & Energy Activities",     "C4": "C4: Upstream Transport",
    "C5": "C5: Waste in Operations",          "C6": "C6: Business Travel",
    "C7": "C7: Employee Commuting",           "C8": "C8: Upstream Leased Assets",
    "C9": "C9: Downstream Transport",         "C10": "C10: Processing of Sold Products",
    "C11": "C11: Use of Sold Products",       "C12": "C12: End-of-Life Treatment",
    "C13": "C13: Downstream Leased Assets",   "C14": "C14: Franchises",
    "C15": "C15: Investments",
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _load_csv(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    return pd.read_csv(path) if os.path.exists(path) else None


def _missing(name):
    st.info(f"📂 **{name}** not found. Run `python run_pipeline.py` first to generate all outputs.", icon="ℹ️")


def chart_header(title, popover_title, popover_body):
    """Renders a chart title with an ℹ️ info popover at the right."""
    col_title, col_info = st.columns([11, 1])
    with col_title:
        st.markdown(f"#### {title}")
    with col_info:
        with st.popover("ℹ️"):
            st.markdown(f"**{popover_title}**")
            st.markdown(popover_body)


def kpi(num, label):
    st.markdown(f"""
    <div class='kpi'>
        <div class='kpi-num'>{num}</div>
        <div class='kpi-label'>{label}</div>
    </div>""", unsafe_allow_html=True)


def section_intro(text):
    st.markdown(f"<div class='section-intro'>{text}</div>", unsafe_allow_html=True)


def finding(text):
    st.markdown(f"<div class='finding-card'>{text}</div>", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────
SLIM_CACHE = os.path.join(OUTPUT_DIR, "data_slim.csv")
RAW_CSV    = "cdu_global_all_data.csv"


@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    """
    Load strategy (in order of preference):
      1. data_slim.csv  — pre-computed cache (~15 MB), committed to GitHub.
                          Works on Streamlit Cloud without the raw file.
      2. cdu_global_all_data.csv — full raw file (local only, 66.8 MB).
                          Used when running locally after pipeline regeneration.
    """
    if os.path.exists(SLIM_CACHE):
        df = pd.read_csv(SLIM_CACHE, low_memory=False)
        # Ensure bool columns are bool (CSV reads them as string True/False)
        bool_cols = [c for c in df.columns
                     if "emissions_method_bool" in c or c in
                     ("emissions_verified", "is_eu", "is_europe",
                      "is_switcher", "never_verified", "ever_verified")]
        for c in bool_cols:
            if c in df.columns:
                df[c] = df[c].map({"True": True, "False": False,
                                   True: True, False: False}).fillna(False).astype(bool)
        return df

    if os.path.exists(RAW_CSV):
        from scope3_thesis.data_processor   import load_and_preprocess_data
        from scope3_thesis.sdqi_calculator  import calculate_sdqi_scores
        from scope3_thesis.switcher_analysis import identify_switchers
        df = load_and_preprocess_data(RAW_CSV)
        df = calculate_sdqi_scores(df)
        df = identify_switchers(df)
        return df

    raise FileNotFoundError(
        "Neither outputs/data_slim.csv nor cdu_global_all_data.csv found. "
        "Run python run_pipeline.py first."
    )


try:
    df = load_data()
    data_ok = True
except Exception as err:
    data_ok = False
    df = pd.DataFrame()
    st.error(f"Could not load dataset: {err}")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌱 Scope 3 SDQI Explorer")
    st.markdown(
        "This dashboard presents research findings from a large-scale analysis of "
        "**33,681 corporate Scope 3 carbon disclosures** (2018–2023) across "
        "**11 sectors** and **40+ countries**."
    )
    st.divider()
    st.markdown("### 📖 Key Terms")
    with st.expander("What is Scope 3?"):
        st.write(
            "Scope 3 emissions are the **indirect emissions** in a company's "
            "value chain — from suppliers (upstream) and customers (downstream). "
            "They typically represent **70–90 % of a company's total carbon footprint** "
            "but are the least regulated and most under-reported."
        )
    with st.expander("What is the SDQI?"):
        st.write(
            "The **Scope 3 Disclosure Quality Index** (SDQI) is an original index "
            "created for this research. It ranges from 0 (worst) to 1 (best) and "
            "combines:\n"
            "- **Completeness** (70 %): how many material categories are addressed\n"
            "- **Primary Data Ratio** (30 %): how much first-hand supplier data is used\n\n"
            "A score of 0.5 means a company covers 50 % of what it should."
        )
    with st.expander("What are the 15 Scope 3 categories?"):
        for k, v in CAT_NAMES.items():
            st.write(f"**{k}** — {v[4:]}")
    with st.expander("What is third-party verification?"):
        st.write(
            "Verification (or assurance) means an independent auditor has reviewed "
            "and confirmed the emissions data. It is similar to a financial audit. "
            "Without it, a company's emissions numbers are self-reported and unconfirmed."
        )
    with st.expander("What is CSRD?"):
        st.write(
            "The **Corporate Sustainability Reporting Directive** (CSRD) is a 2022 EU law "
            "requiring ~50,000 companies to report detailed sustainability data including "
            "Scope 3 emissions. It is the most significant carbon disclosure regulation "
            "ever enacted."
        )
    st.divider()
    if data_ok:
        st.metric("Disclosures loaded", f"{len(df):,}")
        st.metric("Unique companies",   f"{df['nz_id'].nunique():,}")
        st.metric("Reporting years",    f"{df['reporting_year'].nunique()}")

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
  <h1>Scope 3 Disclosure Quality Research Dashboard</h1>
  <p>How transparent are companies about their supply-chain emissions?
     33,681 disclosures · 2018–2023 · 40+ countries · 11 sectors</p>
</div>
""", unsafe_allow_html=True)

if not data_ok:
    st.warning("Place **cdu_global_all_data.csv** in the same folder as dashboard.py and reload.")
    st.stop()

# ── KPI row ──────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1: kpi(f"{len(df):,}", "Total Disclosures")
with c2: kpi(f"{df['nz_id'].nunique():,}", "Unique Companies")
with c3: kpi(f"{df['sdqi_basic'].mean()*100:.1f}%", "Global Mean SDQI")
with c4: kpi(f"{df['emissions_verified'].mean()*100:.0f}%", "Verified Disclosures")
with c5: kpi(f"{df['primary_data_ratio'].mean()*100:.0f}%", "Avg Primary Data Use")
with c6: kpi(f"{df['method_disclosure_rate'].mean()*100:.0f}%", "Avg Method Disclosed")
st.divider()

# ════════════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "🏠 Overview",
    "📊 Category Relevance",
    "🔬 Method Quality",
    "📅 Time Trends & CSRD",
    "✅ Verification Deep-Dive",
    "📐 Regression Models",
    "🏢 Company Explorer",
])

# ────────────────────────────────────────────────────────────────────────────
# TAB 1 — OVERVIEW
# ────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown("### What this dashboard covers")
    section_intro(
        "This research analyses how completely and accurately companies disclose their "
        "Scope 3 supply-chain emissions.  Use the tabs above to explore different angles: "
        "which categories companies report, how much real supplier data they use, whether "
        "verification drives better reporting, and how EU regulation is shifting behaviour."
    )

    st.markdown("#### Key Findings at a Glance")
    finding("<strong>Finding 1 — Verification is the strongest quality driver.</strong> "
            "Companies that get their emissions externally verified score 0.25–0.37 SDQI "
            "points higher. This effect survives company fixed effects and propensity score "
            "matching — it is not merely a selection effect.")
    finding("<strong>Finding 2 — Primary data use is stuck at ~18–19 % globally</strong> "
            "and has not improved in six years. Most companies rely on generic spend-based "
            "estimates, not real supplier data. The EU is the exception: primary data use "
            "rose +5.4 pp after the 2022 CSRD announcement.")
    finding("<strong>Finding 3 — ~30 % of material categories are systematically omitted.</strong> "
            "Companies give 'Silent' or 'Not evaluated' responses to roughly one-third of "
            "categories their sector peers report as material. Extractives & Minerals "
            "Processing shows the highest omission rate (34 %) despite being most exposed.")
    finding("<strong>Finding 4 — Persistent omitters are common.</strong> "
            "Many companies omit the same high-materiality categories for 3+ consecutive years, "
            "which is difficult to explain as oversight and suggests deliberate non-disclosure.")
    finding("<strong>Finding 5 — The EU is pulling ahead.</strong> "
            "EU companies show higher SDQI than any other region and are accelerating — "
            "consistent with firms pre-emptively preparing for CSRD requirements.")

    st.divider()
    st.markdown("#### How to use this dashboard")
    col_h1, col_h2 = st.columns(2)
    with col_h1:
        st.markdown("""
| Tab | What it answers |
|-----|----------------|
| 📊 Category Relevance | *Which categories are omitted — and how suspiciously?* |
| 🔬 Method Quality | *How much real supplier data backs the numbers?* |
| 📅 Time Trends & CSRD | *Is disclosure quality improving over time?* |
""")
    with col_h2:
        st.markdown("""
| Tab | What it answers |
|-----|----------------|
| ✅ Verification Deep-Dive | *Does getting audited actually improve reporting?* |
| 📐 Regression Models | *What statistically predicts better disclosure?* |
| 🏢 Company Explorer | *How does one specific company compare to peers?* |
""")
    st.info(
        "💡 Every chart has an **ℹ️ button** at the top-right. Click it for a plain-language "
        "explanation of what the chart shows and how to read it.",
        icon="💡"
    )

# ────────────────────────────────────────────────────────────────────────────
# TAB 2 — CATEGORY RELEVANCE & OMISSIONS
# ────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    section_intro(
        "The GHG Protocol defines 15 Scope 3 categories. Companies must assess each one "
        "and either report emissions or explain why it is not material. This tab shows "
        "which categories companies actually report — and which they quietly ignore."
    )

    # ── Sector relevancy heatmap ──
    rel_mat = _load_csv("relevancy_matrix_sector.csv")
    chart_header(
        "Category Relevance Rate by Sector (%)",
        "How to read this heatmap",
        "Each cell shows the **percentage of companies in that sector** who marked the "
        "category as 'Relevant' (i.e., material to their business). "
        "Darker blue = more companies report it. "
        "Categories with consistently low rates across all sectors (C13, C14) "
        "may be genuinely non-material. Categories with low rates in sectors where "
        "peers do report them are potential strategic omissions. "
        "\n\n**Key pattern:** C1 (Purchased Goods) is always highest — every sector "
        "has large upstream procurement. C14 (Franchises) is near-zero everywhere except "
        "restaurant and retail companies."
    )
    if rel_mat is not None:
        rel_mat = rel_mat.set_index(rel_mat.columns[0]).apply(pd.to_numeric, errors='coerce') * 100
        fig = px.imshow(rel_mat, color_continuous_scale="YlGnBu",
                        labels=dict(color="Relevance %"), aspect="auto",
                        zmin=0, zmax=100,
                        text_auto=".0f")
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=420,
                          coloraxis_colorbar_title="Relevance %")
        st.plotly_chart(fig, use_container_width=True)
    else:
        _missing("relevancy_matrix_sector.csv")

    st.divider()
    col_o1, col_o2 = st.columns(2)

    # ── Opaque omission heatmap ──
    with col_o1:
        omit_mat = _load_csv("omission_opaque_matrix_sector.csv")
        chart_header(
            "Opaque Omission Rate (Silent + Not Evaluated, %)",
            "What 'opaque omission' means",
            "This heatmap shows the **percentage of companies that gave no useful response** "
            "to each category — either complete silence (NaN in the data) or an explicit "
            "'Not evaluated' response. Both mean the company has not assessed the category. "
            "\n\nThis is **not the same as 'Not relevant'**, where a company makes an active "
            "assessment that the category doesn't apply. 'Silent' and 'Not evaluated' are "
            "the suspicious states — a company may be avoiding a category it knows is material. "
            "\n\n**Red cells** are high-omission combinations warranting scrutiny."
        )
        if omit_mat is not None:
            omit_mat = omit_mat.set_index(omit_mat.columns[0]).apply(pd.to_numeric, errors='coerce')
            fig = px.imshow(omit_mat, color_continuous_scale="Reds",
                            labels=dict(color="Opaque %"), aspect="auto",
                            zmin=0, zmax=70, text_auto=".0f")
            fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            _missing("omission_opaque_matrix_sector.csv")

    # ── Strategic omissions tiered ──
    with col_o2:
        om_sec = _load_csv("strategic_omissions_tiered_sector.csv")
        chart_header(
            "Tiered Omission Profile by Sector",
            "Three tiers of non-reporting",
            "This table breaks omissions into three categories of increasing concern:\n\n"
            "- **Tier-1 Silent** — No response at all. Company never acknowledged the question.\n"
            "- **Tier-2 Not Evaluated** — Acknowledged but explicitly did not assess.\n"
            "- **Active Dismissal** — Said 'Not relevant' (some assessment made, even if debatable).\n\n"
            "Only Tier-1 and Tier-2 are 'opaque' omissions. "
            "**High opaque omission rate + high-exposure sector = strongest strategic omission signal.**"
        )
        if om_sec is not None:
            om_sec = om_sec.reset_index()
            if "sics_sector" in om_sec.columns:
                om_sec = om_sec.rename(columns={
                    "sics_sector":                "Sector",
                    "mean_tier1_silent":          "Avg Silent Count",
                    "mean_tier2_not_evaluated":   "Avg Not Eval Count",
                    "mean_opaque_omission_rate":  "Opaque Omission Rate (%)",
                    "high_omission_firms":        "High-Omission Firms",
                    "total_firms":                "Total Firms",
                })
            st.dataframe(om_sec, use_container_width=True, height=400)
        else:
            _missing("strategic_omissions_tiered_sector.csv")

    st.divider()

    # ── Persistence analysis ──
    chart_header(
        "Systematic Omitters: Persistence Score Distribution",
        "What is a systematic omitter?",
        "A company that leaves a **high-materiality category** (one that ≥50 % of its "
        "sector peers report) as 'Silent' or 'Not evaluated' for **3 or more consecutive "
        "years** is classified as a systematic omitter. "
        "\n\nOne year of silence could be an oversight. Three consecutive years is a pattern. "
        "The histogram below shows how many company–category pairs fall into each "
        "persistence length. The red dashed line marks the 3-year threshold. "
        "\n\n**Why this matters:** Persistent omission of a category that competitors "
        "consistently report is the strongest observable signal of deliberate non-disclosure."
    )
    persist_sec = _load_csv("persistence_scores_sector.csv")
    persist_cat = _load_csv("persistence_scores_category.csv")

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        persist_img = os.path.join(OUTPUT_DIR, "persistence_distribution.png")
        if os.path.exists(persist_img):
            st.image(persist_img, use_column_width=True)
        else:
            _missing("persistence_distribution.png")

    with col_p2:
        if persist_sec is not None:
            st.markdown("**Systematic omitters by sector** (≥3 consecutive years silent on material category)")
            ps = persist_sec.reset_index()
            ps.columns = [c.replace("_", " ").title() for c in ps.columns]
            st.dataframe(ps, use_container_width=True, height=380)
        else:
            _missing("persistence_scores_sector.csv")


# ────────────────────────────────────────────────────────────────────────────
# TAB 3 — METHOD QUALITY
# ────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    section_intro(
        "Reporting an emissions number is only meaningful if it is based on real data. "
        "This tab shows (1) what share of each category's emissions come from primary "
        "supplier data vs generic estimates, and (2) whether companies describe HOW they "
        "calculated the number. A company can claim primary data without explaining their "
        "method — both dimensions together reveal true methodological quality."
    )

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        pd_mat = _load_csv("primary_data_matrix_sector.csv")
        chart_header(
            "Primary Data Share by Sector & Category (%)",
            "What is primary vs secondary data?",
            "**Primary data** = actual supplier-specific activity data (e.g., a supplier's "
            "real energy bill). It is the most accurate basis for emissions calculation.\n\n"
            "**Secondary data** = generic industry averages or spend-based estimates "
            "(e.g., 'every €1,000 of electronics purchases = X kg CO₂'). "
            "Cheaper to collect but much less accurate.\n\n"
            "Cells show the **average % of primary data** used by companies in that sector "
            "for that category. Higher = more accurate. Most categories are under 40 %, "
            "meaning emissions are mostly estimated, not measured.\n\n"
            "**C6 (Business Travel) is an outlier** — companies have travel management "
            "systems that give exact trip data, so primary data rates are 47–69 %."
        )
        if pd_mat is not None:
            pd_mat = pd_mat.set_index(pd_mat.columns[0]).apply(pd.to_numeric, errors='coerce')
            fig = px.imshow(pd_mat, color_continuous_scale="Purples",
                            labels=dict(color="Primary Data %"), aspect="auto",
                            zmin=0, zmax=80, text_auto=".0f")
            fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=420)
            st.plotly_chart(fig, use_container_width=True)
        else:
            _missing("primary_data_matrix_sector.csv")

    with col_m2:
        mdr_mat = _load_csv("mdr_matrix_sector.csv")
        chart_header(
            "Method Disclosure Rate by Sector & Category (%)",
            "Why method disclosure matters",
            "A company can say '30 % primary data' without explaining what they counted "
            "as primary, who verified it, or what assumptions they made. "
            "**Method Disclosure Rate (MDR)** measures what share of relevant-reported "
            "categories also include a description of the calculation methodology.\n\n"
            "Low MDR + high claimed primary data = the claim cannot be independently "
            "verified. This is a new finding: PDR and MDR can diverge significantly, "
            "revealing a class of disclosures that look good on primary data but remain "
            "opaque on method.\n\n"
            "**What to look for:** Sector–category pairs where PDR is high but MDR is "
            "low — those are the most methodologically opaque combinations."
        )
        if mdr_mat is not None:
            mdr_mat = mdr_mat.set_index(mdr_mat.columns[0]).apply(pd.to_numeric, errors='coerce')
            fig = px.imshow(mdr_mat, color_continuous_scale="Blues",
                            labels=dict(color="MDR %"), aspect="auto",
                            zmin=0, zmax=80, text_auto=".0f")
            fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=420)
            st.plotly_chart(fig, use_container_width=True)
        else:
            _missing("mdr_matrix_sector.csv")

    st.divider()

    # Time trend for PDR and MDR
    pol_df = _load_csv("policy_shift_comparison.csv")
    chart_header(
        "Pre- vs Post-2022 Change in Method Quality (CSRD Anticipation Effect)",
        "Why 2022 is the dividing line",
        "The EU's Corporate Sustainability Reporting Directive (CSRD) was adopted in "
        "November 2022, requiring companies to disclose Scope 3 with much greater detail. "
        "Companies likely began preparing early.\n\n"
        "This table compares key quality metrics before and after 2022. "
        "**If CSRD is driving improvement, we expect EU companies to improve more than "
        "non-EU companies** — and that is exactly what the data shows:\n\n"
        "- EU primary data use rose +5.4 pp post-2022\n"
        "- Global primary data use barely moved (+0.4 pp)\n"
        "- US primary data use actually declined (−2.7 pp)\n\n"
        "This is the cleanest causal signal in the dataset for the policy impact of CSRD."
    )
    if pol_df is not None:
        display_cols = ["Group", "Pre-2022 SDQI", "Post-2022 SDQI", "SDQI Change",
                        "Pre-2022 Primary Data", "Post-2022 Primary Data", "Primary Data Change"]
        if "Pre-2022 MDR" in pol_df.columns:
            display_cols += ["Pre-2022 MDR", "Post-2022 MDR", "MDR Change"]
        pol_show = pol_df[[c for c in display_cols if c in pol_df.columns]].copy()
        for col in pol_show.select_dtypes("float64").columns:
            pol_show[col] = pol_show[col].map(lambda x: f"{x:+.4f}" if pd.notna(x) else "—")
        st.dataframe(pol_show, use_container_width=True)
    else:
        _missing("policy_shift_comparison.csv")


# ────────────────────────────────────────────────────────────────────────────
# TAB 4 — TIME TRENDS & CSRD
# ────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    section_intro(
        "How has reporting quality changed over 2018–2023? Has the global trend been "
        "uniform or are some regions pulling ahead? This tab shows SDQI trajectories "
        "broken down by region and sector."
    )

    trends_df = _load_csv("sdqi_trends_region.csv")
    col_t1, col_t2 = st.columns([3, 2])

    with col_t1:
        chart_header(
            "SDQI Trend by Region (2018–2023)",
            "How to read the SDQI trend chart",
            "Each line shows the **mean SDQI score** for all companies headquartered in "
            "that region, for each reporting year. SDQI ranges from 0 (worst) to 1 (best).\n\n"
            "**Key patterns to note:**\n"
            "- All regions improve 2021→2022 (CSRD announcement effect)\n"
            "- EU consistently higher than rest of world\n"
            "- China is lowest and improving most slowly\n"
            "- Japan improved sharply 2022–2023 (Tokyo Stock Exchange sustainability rules)\n\n"
            "The 2019–2020 dip is a reporting-year artefact: companies filing in those "
            "years had fewer prior-year data points and lower consistency scores."
        )
        if trends_df is not None:
            trends_df["reporting_year"] = pd.to_numeric(trends_df["reporting_year"])
            fig = go.Figure()
            colors = {"EU-27": "#17b978", "United States": "#1e3d59",
                      "Japan": "#ff6f61", "China": "#ffc93c", "Rest of World": "#adb5bd"}
            for col in [c for c in trends_df.columns if c != "reporting_year"]:
                fig.add_trace(go.Scatter(
                    x=trends_df["reporting_year"], y=trends_df[col],
                    mode="lines+markers", name=col,
                    line=dict(color=colors.get(col, "#888"), width=2.5),
                    marker=dict(size=7)
                ))
            fig.add_vline(x=2022, line_dash="dash", line_color="#ff6f61",
                          annotation_text="CSRD 2022", annotation_position="top right")
            fig.update_layout(
                xaxis_title="Reporting Year", yaxis_title="Mean SDQI",
                yaxis_range=[0, 0.7], legend_title="Region",
                margin=dict(l=5, r=5, t=10, b=5), height=380
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            _missing("sdqi_trends_region.csv")

    with col_t2:
        chart_header(
            "SDQI by Sector (latest year)",
            "Which sectors disclose best?",
            "Each bar shows the **mean SDQI** for all companies in that sector for the "
            "most recent reporting year available in the data.\n\n"
            "Higher SDQI = more complete and better-quality reporting. "
            "Sectors closer to 0.5+ are performing well; sectors under 0.35 have "
            "significant disclosure gaps.\n\n"
            "**Financials** often rank high because their material categories "
            "(C1, C6, C15 Investments) are well-defined. "
            "**Extractives** often rank lower despite high emissions exposure, "
            "suggesting disclosure quality does not match emissions risk."
        )
        latest_year = df["reporting_year"].max()
        sec_sdqi = (
            df[df["reporting_year"] == latest_year]
            .groupby("sics_sector")["sdqi_basic"]
            .mean()
            .drop("Information Not Available", errors="ignore")
            .sort_values()
            .reset_index()
        )
        sec_sdqi.columns = ["Sector", "Mean SDQI"]
        fig_bar = px.bar(sec_sdqi, x="Mean SDQI", y="Sector", orientation="h",
                         color="Mean SDQI", color_continuous_scale="YlGnBu",
                         range_x=[0, 0.7], height=380)
        fig_bar.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # Interactive year-over-year sector selector
    chart_header(
        "Year-by-Year SDQI Change by Sector",
        "What drives annual SDQI movements?",
        "This chart lets you select a sector and see how its SDQI evolved each year.\n\n"
        "Sharp jumps upward often follow regulatory announcements, index inclusion "
        "(e.g., MSCI ESG ratings), or mandatory reporting rules. Drops can indicate "
        "companies newly entering the sample with lower quality.\n\n"
        "The grey reference line shows the global average for context."
    )
    avail_sectors = sorted([s for s in df["sics_sector"].unique() if s != "Information Not Available"])
    sel_sector = st.selectbox("Select sector:", avail_sectors, key="trend_sector")
    sec_trend = (df[df["sics_sector"] == sel_sector]
                 .groupby("reporting_year")["sdqi_basic"].mean().reset_index())
    global_trend = df.groupby("reporting_year")["sdqi_basic"].mean().reset_index()

    fig_tr = go.Figure()
    fig_tr.add_trace(go.Scatter(x=sec_trend["reporting_year"], y=sec_trend["sdqi_basic"],
                                mode="lines+markers", name=sel_sector,
                                line=dict(color="#17b978", width=3), marker=dict(size=9)))
    fig_tr.add_trace(go.Scatter(x=global_trend["reporting_year"], y=global_trend["sdqi_basic"],
                                mode="lines", name="Global Average",
                                line=dict(color="#adb5bd", width=2, dash="dot")))
    fig_tr.add_vline(x=2022, line_dash="dash", line_color="#ff6f61")
    fig_tr.update_layout(xaxis_title="Year", yaxis_title="Mean SDQI",
                         yaxis_range=[0, 0.7], margin=dict(l=5, r=5, t=10, b=5), height=330)
    st.plotly_chart(fig_tr, use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────
# TAB 5 — VERIFICATION DEEP-DIVE
# ────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    section_intro(
        "The regression analysis shows that verified companies score 0.25–0.37 SDQI "
        "points higher. But does verification *cause* better disclosure, or do better-managed "
        "companies simply do both? This tab presents two analyses that address this directly: "
        "a within-company event study for companies that *switched* to verified, and a "
        "propensity-score-matched comparison."
    )

    col_v1, col_v2 = st.columns(2)

    with col_v1:
        ev_df = _load_csv("switcher_event_study.csv")
        chart_header(
            "Event Study: SDQI Before & After First Verification",
            "What is an event study and why does it matter?",
            "An **event study** tracks a company's performance relative to a key event — "
            "here, the year it first received third-party verification (T=0).\n\n"
            "**Green line (Switchers):** 1,917 companies that switched from unverified to "
            "verified during 2018–2023. Their SDQI at T-2, T-1, T, T+1, T+2.\n\n"
            "**Grey line (Never Verified):** Never-verified companies over the same "
            "calendar period, shown for comparison.\n\n"
            "**What to look for:** If the green line rises sharply at T=0 while the "
            "grey line stays flat, that is evidence that verification itself causes the "
            "improvement — not just that better companies tend to verify. "
            "If both lines move together, it is likely a general time trend."
        )
        if ev_df is not None:
            fig_ev = go.Figure()
            colors_ev = {"Switchers": "#17b978", "Never Verified": "#adb5bd"}
            for grp, sub in ev_df.groupby("group"):
                sub = sub.sort_values("event_time")
                fig_ev.add_trace(go.Scatter(
                    x=sub["event_time"], y=sub["mean_sdqi"],
                    mode="lines+markers", name=grp,
                    line=dict(color=colors_ev.get(grp, "#888"), width=2.5),
                    marker=dict(size=8),
                    error_y=dict(type='data', array=sub["se"] * 1.96,
                                 visible=True, color=colors_ev.get(grp, "#888"))
                ))
            fig_ev.add_vline(x=0, line_dash="dash", line_color="#1e3d59",
                             annotation_text="First verification year (T=0)",
                             annotation_position="top right")
            fig_ev.update_layout(
                xaxis_title="Years relative to first verification",
                yaxis_title="Mean SDQI",
                xaxis=dict(tickvals=[-2, -1, 0, 1, 2]),
                margin=dict(l=5, r=5, t=10, b=5), height=380
            )
            st.plotly_chart(fig_ev, use_container_width=True)
        else:
            _missing("switcher_event_study.csv")

    with col_v2:
        did_df = _load_csv("switcher_did_results.csv")
        chart_header(
            "Difference-in-Differences: Verification Effect (Company + Year FE)",
            "What is Difference-in-Differences?",
            "DiD compares the SDQI improvement of **switchers** (companies that began "
            "verifying) against **never-verified companies** over the same calendar years.\n\n"
            "By controlling for company fixed effects (each company's own baseline) and "
            "year fixed effects (global trends), the DiD estimate isolates the part of "
            "the SDQI improvement that is uniquely attributable to starting verification — "
            "not to the company already being good, and not to the general upward trend.\n\n"
            "**Interpretation:** A positive, significant DiD coefficient means verification "
            "independently raises quality. Standard errors are clustered at company level."
        )
        if did_df is not None:
            st.dataframe(did_df[["estimator", "coefficient", "std_error", "p_value",
                                 "r_squared", "n_obs", "interpretation"]].T,
                         use_container_width=True)
        else:
            _missing("switcher_did_results.csv")

        st.divider()

        psm_df = _load_csv("psm_results.csv")
        chart_header(
            "Propensity Score Matching (PSM) ATT vs OLS Coefficient",
            "What is PSM and why does it help?",
            "PSM matches each verified company to the most similar unverified company "
            "(same sector, year, region, and similar size) and compares their SDQI.\n\n"
            "If the PSM Average Treatment Effect on the Treated (ATT) is similar to the "
            "OLS coefficient (+0.35), it means the OLS is not driven by selection bias — "
            "the verification effect is real even when comparing like-for-like companies.\n\n"
            "If the PSM ATT is much smaller, it would suggest that better companies are "
            "self-selecting into verification, and verification alone explains less of the "
            "quality gap. The comparison gives an honest range for the true effect."
        )
        if psm_df is not None:
            st.dataframe(psm_df, use_container_width=True)
            overall = psm_df[psm_df["year"] == "Overall (weighted)"]
            if not overall.empty:
                att_val = float(overall["att"].values[0])
                st.metric("PSM ATT (overall)", f"{att_val:+.4f}",
                          delta=f"OLS reference: ~+0.35",
                          delta_color="normal")
        else:
            _missing("psm_results.csv")


# ────────────────────────────────────────────────────────────────────────────
# TAB 6 — REGRESSION MODELS
# ────────────────────────────────────────────────────────────────────────────
with tabs[5]:
    section_intro(
        "Four OLS regression models test which company characteristics predict higher SDQI. "
        "All models use **clustered standard errors** at the company level (correcting for "
        "repeated observations of the same company across years). "
        "Results are consistent across all four specifications."
    )

    col_r1, col_r2 = st.columns([1, 1.3])

    with col_r1:
        chart_header(
            "Coefficient Plot — Key Predictors of SDQI",
            "How to read a coefficient plot",
            "Each dot shows the **estimated effect** of one variable on the SDQI score "
            "(0–1 scale). The horizontal error bars show the **95 % confidence interval**.\n\n"
            "- A dot to the **right of zero** means that variable is associated with "
            "  **higher** SDQI (better disclosure quality).\n"
            "- A dot to the **left of zero** means lower SDQI.\n"
            "- If the error bar **does not cross zero**, the effect is statistically "
            "  significant (p < 0.05).\n\n"
            "The verification effect is the largest bar — far right, never crossing zero "
            "in any model specification."
        )

        model_options = ["(1) Global Basic", "(2) Panel Cohort", "(3) EU Subsample", "(4) Known Sectors"]
        selected_model = st.radio("Select model:", model_options, horizontal=True)
        safe_name = selected_model.replace("(", "").replace(")", "").replace(" ", "_").lower()
        sum_file  = os.path.join(OUTPUT_DIR, f"regression_summary_{safe_name}.txt")

        if os.path.exists(sum_file):
            import statsmodels.formula.api as smf

            @st.cache_data(show_spinner=False)
            def _fit_for_plot(model_name, _df):
                formulas = {
                    "(1) Global Basic":   ("sdqi_basic",
                        "sdqi_basic ~ C(boundary_approach_clean, Treatment(reference='Not Disclosed')) + "
                        "C(emissions_verified) + company_size_proxy + "
                        "C(region, Treatment(reference='Rest_of_World'))"),
                    "(2) Panel Cohort":   ("sdqi_panel",
                        "sdqi_panel ~ C(boundary_approach_clean, Treatment(reference='Not Disclosed')) + "
                        "C(emissions_verified) + company_size_proxy + "
                        "C(region, Treatment(reference='Rest_of_World'))"),
                    "(3) EU Subsample":   ("sdqi_basic",
                        "sdqi_basic ~ C(boundary_approach_clean, Treatment(reference='Not Disclosed')) + "
                        "C(emissions_verified) + company_size_proxy"),
                    "(4) Known Sectors":  ("sdqi_basic",
                        "sdqi_basic ~ C(boundary_approach_clean, Treatment(reference='Not Disclosed')) + "
                        "C(emissions_verified) + company_size_proxy + "
                        "C(sics_sector) + C(region, Treatment(reference='Rest_of_World'))"),
                }
                dv, formula = formulas[model_name]
                fit_df = _df.copy()
                if model_name == "(2) Panel Cohort":
                    fit_df = fit_df[fit_df["consistency_score"].notna()]
                elif model_name == "(3) EU Subsample":
                    fit_df = fit_df[fit_df["is_eu"]]
                elif model_name == "(4) Known Sectors":
                    fit_df = fit_df[fit_df["sics_sector"] != "Information Not Available"]
                model = smf.ols(formula, data=fit_df).fit(
                    cov_type='cluster', cov_kwds={'groups': fit_df['nz_id']})
                ci = model.conf_int()
                coef_df = pd.DataFrame({
                    "Coef": model.params,
                    "CI_lower": ci[0], "CI_upper": ci[1]
                })
                coef_df = coef_df[~coef_df.index.str.contains("Intercept|sics_sector|nz_id")]
                return coef_df

            try:
                coef_df = _fit_for_plot(selected_model, df)
                labels = (coef_df.index
                          .str.replace(r"C\(boundary_approach_clean.*?\)\[T\.", "Boundary: ", regex=True)
                          .str.replace("C(emissions_verified)[T.True]", "✅ Verified Emissions", regex=False)
                          .str.replace(r"C\(region.*?\)\[T\.", "Region: ", regex=True)
                          .str.replace("]", "", regex=False)
                          .str.replace("company_size_proxy", "Company Size (log)")
                          .str.strip())
                fig_coef = go.Figure()
                fig_coef.add_trace(go.Scatter(
                    x=coef_df["Coef"], y=labels, mode="markers",
                    marker=dict(color="#1e3d59", size=10),
                    error_x=dict(type='data', symmetric=False,
                                 array=coef_df["CI_upper"] - coef_df["Coef"],
                                 arrayminus=coef_df["Coef"] - coef_df["CI_lower"],
                                 color="#ff6f61"),
                    name="Coefficient"
                ))
                fig_coef.add_vline(x=0, line_dash="dash", line_color="#adb5bd")
                fig_coef.update_layout(
                    xaxis_title="Effect on SDQI score (0–1 scale)",
                    margin=dict(l=5, r=5, t=10, b=5), height=420
                )
                st.plotly_chart(fig_coef, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not render coefficient plot: {e}")

    with col_r2:
        chart_header(
            "Publication Regression Table (All 4 Models)",
            "How to read a regression table",
            "Each column is one regression model. Each row is one explanatory variable. "
            "Numbers are the estimated **coefficient** — how much SDQI changes for a one-unit "
            "change in that variable. Stars indicate statistical significance:\n\n"
            "- *** p < 0.01 (very strong evidence)\n"
            "- ** p < 0.05 (strong evidence)\n"
            "- * p < 0.10 (moderate evidence)\n\n"
            "Numbers in parentheses below each coefficient are **standard errors** (clustered "
            "at company level). Smaller SE = more precise estimate.\n\n"
            "The 'Verified Emissions' row is the most important — it is large, positive, "
            "and *** in all four models."
        )
        reg_table = _load_csv("regression_publication_table.csv")
        if reg_table is not None:
            st.dataframe(reg_table.fillna(""), height=500, use_container_width=True)
            tex_path = os.path.join(OUTPUT_DIR, "regression_publication_table.tex")
            if os.path.exists(tex_path):
                with open(tex_path, encoding="utf-8") as f:
                    tex_content = f.read()
                st.download_button("⬇️ Download LaTeX table", tex_content,
                                   "regression_table.tex", mime="text/plain")
        else:
            _missing("regression_publication_table.csv")

    st.divider()

    # Sensitivity table
    sens_df = _load_csv("sdqi_sensitivity_table.csv")
    chart_header(
        "SDQI Weight Sensitivity: Is the Verification Effect Robust?",
        "Why we test different SDQI weights",
        "The SDQI is constructed as: **0.7 × Completeness + 0.3 × Primary Data**. "
        "The weights 0.7/0.3 are chosen to reflect that completeness is a prerequisite "
        "for quality — but this choice is somewhat arbitrary.\n\n"
        "The sensitivity table re-runs the main regression (Model 1) with four alternative "
        "weight configurations and one extended specification that includes Method Disclosure "
        "Rate (MDR). If the verification coefficient is stable across all specifications, "
        "the results are **not sensitive to the weight choice** and the criticism is neutralised.\n\n"
        "**What to look for:** The verification coefficient should stay in a narrow band "
        "(e.g., 0.30–0.40) across all rows. Widening confidence intervals would be a concern."
    )
    if sens_df is not None:
        for col in ["Verification Coefficient", "CI Lower", "CI Upper", "P-value", "R-squared"]:
            if col in sens_df.columns:
                sens_df[col] = sens_df[col].apply(
                    lambda x: f"{x:.4f}" if pd.notna(x) else "—")
        st.dataframe(sens_df, use_container_width=True)
    else:
        _missing("sdqi_sensitivity_table.csv")


# ────────────────────────────────────────────────────────────────────────────
# TAB 7 — COMPANY EXPLORER
# ────────────────────────────────────────────────────────────────────────────
with tabs[6]:
    section_intro(
        "Use the filters below to narrow down companies by sector, industry, country, or "
        "verification status — then select a company to see its full Scope 3 reporting history."
    )

    # ── Cascading filters ────────────────────────────────────────────────────
    # Filters are cascading: each one narrows the options available to the next.
    # Leaving a filter blank means "show all".

    filter_df = df.copy()  # working copy, progressively narrowed

    # Row 1: Sector | Sub-sector (industry) | Country | Region
    fc1, fc2, fc3, fc4 = st.columns(4)

    with fc1:
        all_sectors = sorted([s for s in filter_df["sics_sector"].dropna().unique()
                               if s != "Information Not Available"])
        sel_sectors = st.multiselect("Sector", all_sectors, placeholder="All sectors")
        if sel_sectors:
            filter_df = filter_df[filter_df["sics_sector"].isin(sel_sectors)]

    with fc2:
        all_industries = sorted([i for i in filter_df["sics_industry"].dropna().unique()
                                  if str(i) not in ("nan", "Information Not Available", "")])
        sel_industries = st.multiselect("Industry / Sub-sector", all_industries,
                                        placeholder="All industries")
        if sel_industries:
            filter_df = filter_df[filter_df["sics_industry"].isin(sel_industries)]

    with fc3:
        all_countries = sorted(filter_df["jurisdiction_clean"].dropna().unique())
        sel_countries = st.multiselect("Country", all_countries, placeholder="All countries")
        if sel_countries:
            filter_df = filter_df[filter_df["jurisdiction_clean"].isin(sel_countries)]

    with fc4:
        all_regions = sorted(filter_df["region"].dropna().unique())
        sel_regions = st.multiselect("Region", all_regions, placeholder="All regions")
        if sel_regions:
            filter_df = filter_df[filter_df["region"].isin(sel_regions)]

    # Row 2: Verification status | SDQI range | Reset button
    fv1, fv2, fv3 = st.columns([1, 2, 1])

    with fv1:
        ver_options = ["All", "Verified only", "Unverified only"]
        sel_ver = st.selectbox("Verification", ver_options, key="expl_ver")
        if sel_ver == "Verified only":
            filter_df = filter_df[filter_df["emissions_verified"] == True]
        elif sel_ver == "Unverified only":
            filter_df = filter_df[filter_df["emissions_verified"] == False]

    with fv2:
        sdqi_min_val = float(filter_df["sdqi_basic"].min()) if not filter_df.empty else 0.0
        sdqi_max_val = float(filter_df["sdqi_basic"].max()) if not filter_df.empty else 1.0
        if sdqi_min_val < sdqi_max_val:
            sdqi_range = st.slider(
                "SDQI score range",
                min_value=0.0, max_value=1.0,
                value=(round(sdqi_min_val, 2), round(sdqi_max_val, 2)),
                step=0.01, key="expl_sdqi"
            )
            filter_df = filter_df[
                (filter_df["sdqi_basic"] >= sdqi_range[0]) &
                (filter_df["sdqi_basic"] <= sdqi_range[1])
            ]

    with fv3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Reset all filters", use_container_width=True):
            st.rerun()

    # Company count feedback
    matched_companies = filter_df["company_name"].dropna().unique()
    n_matched = len(matched_companies)
    total_companies = df["company_name"].dropna().nunique()
    if n_matched == total_companies:
        st.caption(f"Showing all {n_matched:,} companies — use filters above to narrow down.")
    elif n_matched == 0:
        st.warning("No companies match the current filters. Try broadening your selection.")
        st.stop()
    else:
        st.caption(f"✅ **{n_matched:,}** companies match your filters  "
                   f"(out of {total_companies:,} total)")

    st.divider()

    # Company selectbox — only shows filtered companies
    unique_cos = sorted(matched_companies)
    sel_company = st.selectbox("🔍 Search / select a company:", unique_cos)
    comp_df = df[df["company_name"] == sel_company].sort_values("reporting_year")

    if comp_df.empty:
        st.warning("No data found for this company.")
    else:
        comp_sector = comp_df["sics_sector"].iloc[-1]
        comp_region = comp_df["jurisdiction_clean"].iloc[-1]
        is_sw = bool(comp_df["is_switcher"].iloc[-1])

        # Info card
        latest = comp_df.iloc[-1]
        st.markdown(f"""
<div style='background:#f0f4f8; color:#1e3d59; border-radius:10px; padding:1rem 1.4rem;
     border-left:5px solid #1e3d59; margin-bottom:1.2rem;'>
  <b>Sector:</b> {comp_sector} &nbsp;|&nbsp;
  <b>Country:</b> {comp_region} &nbsp;|&nbsp;
  <b>Years in data:</b> {comp_df['reporting_year'].min()}–{comp_df['reporting_year'].max()} &nbsp;|&nbsp;
  <b>Verification status:</b> {'✅ Currently Verified' if latest['emissions_verified'] else '❌ Not Verified'}
  {'&nbsp;|&nbsp;<b>Verification switcher:</b> ✅ Yes (started verifying during panel)' if is_sw else ''}
</div>""", unsafe_allow_html=True)

        col_c1, col_c2 = st.columns(2)

        with col_c1:
            chart_header(
                f"SDQI Trajectory vs {comp_sector} Average",
                "Reading the company trajectory",
                "The **green line** is this company's SDQI over time. "
                "The **dashed navy line** is the mean SDQI of all companies in the same sector.\n\n"
                "If the company line is consistently below the sector average, it is "
                "under-performing its peers on disclosure quality. If it crosses the "
                "sector average upward, it has improved relative to peers.\n\n"
                "A jump in the company line coinciding with the start of verification "
                "is direct evidence that verification improved this company's disclosure quality."
            )
            sector_avg = (df[df["sics_sector"] == comp_sector]
                          .groupby("reporting_year")["sdqi_basic"].mean().reset_index())
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Scatter(
                x=comp_df["reporting_year"], y=comp_df["sdqi_basic"],
                mode="lines+markers", name=sel_company,
                line=dict(color="#17b978", width=3), marker=dict(size=9)
            ))
            fig_comp.add_trace(go.Scatter(
                x=sector_avg["reporting_year"], y=sector_avg["sdqi_basic"],
                mode="lines", name=f"{comp_sector} avg",
                line=dict(color="#1e3d59", width=2, dash="dash")
            ))
            # Mark verification switch year if applicable
            if is_sw and "first_verified_year" in comp_df.columns:
                fyear = comp_df["first_verified_year"].dropna()
                if not fyear.empty:
                    fig_comp.add_vline(x=int(fyear.iloc[0]), line_dash="dot",
                                       line_color="#ff6f61",
                                       annotation_text="First verified",
                                       annotation_position="top left")
            fig_comp.update_layout(xaxis_title="Year", yaxis_title="SDQI",
                                   yaxis_range=[0, 1.05],
                                   margin=dict(l=5, r=5, t=10, b=5), height=380)
            st.plotly_chart(fig_comp, use_container_width=True)

        with col_c2:
            # Build year list with explicit int conversion so the filter is
            # type-safe regardless of whether the slim CSV loaded years as
            # int64 or float32.
            years_avail = sorted(
                [int(y) for y in comp_df["reporting_year"].unique()],
                reverse=True
            )

            # Key includes the company name so each company gets its own
            # fresh selectbox state — prevents stale year selection when
            # switching between companies.
            sel_year = st.selectbox(
                "Select reporting year:",
                years_avail,
                key=f"comp_year_{sel_company}"
            )

            # Title updates dynamically with the selected year
            chart_header(
                f"Scope 3 Category Breakdown — {sel_year}",
                "Understanding the category breakdown",
                "Each row is one of the 15 GHG Protocol Scope 3 categories. "
                "The columns show:\n\n"
                "- **State** — What the company said: "
                "'🟢 Relevant' (reported), '🟡 Not relevant' (active dismissal), "
                "'🔴 Not evaluated' or '⚫ Silent' (no response — suspicious)\n"
                "- **Primary Data %** — Share of the calculation backed by real supplier data\n"
                "- **Method Disclosed** — Did the company describe how it calculated this?\n"
                "- **Emissions (tCO₂e)** — Reported volume for that category\n\n"
                "**Why does the table sometimes look the same across years?** "
                "Large mature reporters (e.g. major utilities) often keep the same "
                "relevancy designations year after year. Look at the Emissions column "
                "for year-on-year numerical changes."
            )

            # Filter to exactly the selected year
            sel_row_df = comp_df[comp_df["reporting_year"].astype(int) == sel_year]
            if sel_row_df.empty:
                st.warning(f"No data found for {sel_year}.")
            else:
                row = sel_row_df.iloc[0]
                cat_rows = []
                for i in range(1, 16):
                    c = f"c{i}"
                    rel  = row.get(f"s3_ghgp_{c}_emissions_relevancy", "—")
                    pdv  = row.get(f"s3_ghgp_{c}_emissions_primary_data", np.nan)
                    emv  = row.get(f"total_s3_ghgp_{c}_emissions_ghg", np.nan)
                    meth = row.get(f"disclose_s3_ghgp_{c}_emissions_method_bool", False)
                    rel_icon = {"Relevant": "🟢", "Not relevant": "🟡",
                                "Not evaluated": "🔴", "Silent": "⚫"}.get(str(rel), "❓")
                    cat_rows.append({
                        "Category": f"C{i}: {list(CAT_NAMES.values())[i-1][4:]}",
                        "State": f"{rel_icon} {rel}",
                        "Primary Data (%)": f"{pdv:.0f}%" if pd.notna(pdv) else "—",
                        "Method Disclosed": "✅" if meth else "—",
                        "Emissions (tCO₂e)": f"{emv:,.0f}" if pd.notna(emv) else "—",
                    })
                st.dataframe(pd.DataFrame(cat_rows), use_container_width=True, height=460)

        # SDQI sub-scores — track the selected year, not always the latest
        st.divider()
        sel_row_df2 = comp_df[comp_df["reporting_year"].astype(int) == sel_year]
        score_row = sel_row_df2.iloc[0] if not sel_row_df2.empty else latest
        score_year_label = sel_year if not sel_row_df2.empty else int(latest["reporting_year"])

        st.markdown(f"**SDQI sub-scores for {score_year_label}** "
                    f"(change the year selector above to compare across years)")
        m1c, m2c, m3c, m4c = st.columns(4)
        with m1c:
            st.metric("SDQI Score",         f"{score_row['sdqi_basic']:.3f}")
        with m2c:
            st.metric("Completeness Score", f"{score_row['completeness_score']:.3f}")
        with m3c:
            st.metric("Primary Data Ratio", f"{score_row['primary_data_ratio']:.3f}")
        with m4c:
            mdr_val = score_row.get("method_disclosure_rate", np.nan)
            st.metric("Method Disclosure Rate", f"{mdr_val:.3f}" if pd.notna(mdr_val) else "—")
