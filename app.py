import os
from typing import Tuple, List
import io  # <-- NEW: for CSV template

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# OR-Tools (for optimization)
try:
    from ortools.linear_solver import pywraplp
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False

# OpenAI for AI memo generation (new client style)
from openai import OpenAI
from dotenv import load_dotenv

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="AI Credit Analyst Assistant",
    page_icon="💼",
    layout="wide",
)

# --------------------------------------------------
# LOAD OPENAI KEY & INIT CLIENT
# --------------------------------------------------

load_dotenv()

# Try Streamlit Cloud secrets first, then fall back to local .env
api_key = None
try:
    if "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    api_key = None

if not api_key:
    api_key = os.getenv("OPENAI_API_KEY")

OPENAI_API_KEY = api_key
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# --------------------------------------------------
# BASIC STYLE (HEADER, CARDS, SECTIONS, MEMO BOX)
# --------------------------------------------------

CUSTOM_CSS = """
<style>
html, body, [class*="css"] {
    font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

/* App header bar */
.app-header {
    padding: 0.75rem 1.25rem 0.6rem;
    border-bottom: 1px solid #0f172a;
    margin-bottom: 0.75rem;
    background: linear-gradient(90deg, #0b1533 0%, #133b5c 100%);
}
.app-header-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #f9fafb;
    margin-bottom: 0.15rem;
}
.app-header-subtitle {
    font-size: 0.9rem;
    color: #d1d5db;
}

/* Main header (within pages, optional) */
.main-header {
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
}

.sub-header {
    font-size: 0.95rem;
    color: #6c757d;
    margin-bottom: 1.5rem;
}

/* Cards */
.metric-card {
    padding: 1rem 1.25rem;
    border-radius: 0.9rem;
    background: linear-gradient(135deg, #ffffff 0%, #f5f7ff 100%);
    border: 1px solid #e5e7eb;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
    margin-bottom: 0.75rem;
}

.metric-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    color: #6c757d;
    letter-spacing: 0.08em;
    margin-bottom: 0.2rem;
}

.metric-value {
    font-size: 1.4rem;
    font-weight: 600;
}

/* Section titles */
.section-title {
    font-size: 1.2rem;
    font-weight: 600;
    margin: 1.0rem 0 0.5rem 0;
}

/* Memo box styling */
.memo-box {
    background: #f8f9fb;
    border-radius: 0.75rem;
    border: 1px solid #e1e5ee;
    padding: 1rem 1.25rem;
    line-height: 1.5;
    white-space: pre-wrap;
    font-size: 0.95rem;
}

/* Footer */
.app-footer {
    margin-top: 2.0rem;
    padding-top: 0.75rem;
    border-top: 1px solid #e5e7eb;
    font-size: 0.75rem;
    color: #9ca3af;
    text-align: center;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def section_title(text: str):
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)


# --------------------------------------------------
# DATA LOADING & VALIDATION
# --------------------------------------------------

REQUIRED_COLUMNS = [
    "company_id",
    "revenue",
    "ebitda",
    "ebitda_margin",
    "total_debt",
    "cash",
    "interest_expense",
    "capex",
    "free_cash_flow",
    "leverage",
    "coverage",
    "liquidity_ratio",
    "rating_bucket",
    "pd",
]


@st.cache_data
def load_sample_data() -> pd.DataFrame:
    """
    Load the synthetic corporate borrower dataset (realistic PD version).
    """
    sample_path = os.path.join("data", "synthetic_corporate_borrowers.csv")
    if os.path.exists(sample_path):
        df = pd.read_csv(sample_path)
    else:
        # fallback name in case you saved it differently
        df = pd.read_csv("synthetic_corporate_borrowers_realistic_pd.csv")
    return df


def validate_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    return len(missing) == 0, missing


def create_template_csv() -> str:
    """Return a CSV string for a blank borrower template with the required schema."""
    df_template = pd.DataFrame(columns=REQUIRED_COLUMNS)
    buf = io.StringIO()
    df_template.to_csv(buf, index=False)
    return buf.getvalue()

# --------------------------------------------------
# SIMPLE RISK BAND (Corporate version)
# --------------------------------------------------

def assign_risk_level(row: pd.Series) -> str:
    lev = row.get("leverage", np.nan)
    cov = row.get("coverage", np.nan)
    liq = row.get("liquidity_ratio", np.nan)

    high_flags = []
    low_flags = []

    if pd.notna(lev):
        if lev > 4.0:
            high_flags.append("high leverage")
        elif lev < 2.5:
            low_flags.append("low leverage")

    if pd.notna(cov):
        if cov < 1.5:
            high_flags.append("weak coverage")
        elif cov > 3.0:
            low_flags.append("strong coverage")

    if pd.notna(liq):
        if liq < 0.10:
            high_flags.append("thin liquidity")
        elif liq > 0.30:
            low_flags.append("strong liquidity")

    if len(high_flags) > 0:
        return "High"
    if len(low_flags) >= 2:
        return "Low"
    return "Medium"


def risk_color(level: str) -> str:
    return {"Low": "🟢 Low", "Medium": "🟡 Medium", "High": "🔴 High"}.get(level, level)


def risk_driver_text(row: pd.Series) -> str:
    msgs = []
    if row["leverage"] > 4:
        msgs.append("high leverage")
    if row["coverage"] < 1.5:
        msgs.append("weak interest coverage")
    if row["liquidity_ratio"] < 0.10:
        msgs.append("thin liquidity")
    if not msgs:
        return "Balanced leverage, coverage, and liquidity."
    return "Key risk drivers: " + ", ".join(msgs)

# --------------------------------------------------
# STRESS TEST LOGIC
# --------------------------------------------------

def apply_stress(df: pd.DataFrame,
                 revenue_decline_pct: float,
                 interest_increase_pct: float,
                 liquidity_decline_pct: float) -> pd.DataFrame:

    stressed = df.copy()

    # Basic financial stresses
    stressed["revenue_stressed"] = stressed["revenue"] * (1 - revenue_decline_pct / 100)
    stressed["ebitda_stressed"] = stressed["ebitda"] * (1 - revenue_decline_pct / 100)
    stressed["interest_expense_stressed"] = stressed["interest_expense"] * (
        1 + interest_increase_pct / 100
    )
    stressed["cash_stressed"] = stressed["cash"] * (1 - liquidity_decline_pct / 100)

    # Recalculate ratios
    stressed["leverage_stressed"] = stressed["total_debt"] / stressed["ebitda_stressed"].replace(0, np.nan)
    stressed["coverage_stressed"] = stressed["ebitda_stressed"] / stressed["interest_expense_stressed"].replace(0, np.nan)
    stressed["liquidity_ratio_stressed"] = stressed["cash_stressed"] / stressed["total_debt"].replace(0, np.nan)

    # PD shock (simple, scenario-driven)
    shock_factor = (
        1
        + 0.5 * (revenue_decline_pct / 100)
        + 0.3 * (interest_increase_pct / 100)
        + 0.2 * (liquidity_decline_pct / 100)
    )
    stressed["pd_stressed"] = (stressed["pd"] * shock_factor).clip(0.01, 0.50)

    # Risk band under stress
    stressed["Risk_Level_Stressed"] = stressed.apply(
        lambda r: assign_risk_level(
            pd.Series({
                "leverage": r["leverage_stressed"],
                "coverage": r["coverage_stressed"],
                "liquidity_ratio": r["liquidity_ratio_stressed"],
            })
        ),
        axis=1,
    )

    return stressed

# --------------------------------------------------
# OPTIMIZATION LOGIC (OR-Tools) – MAXIMIZE PROFIT ONLY
# --------------------------------------------------

def optimize_portfolio(
    df: pd.DataFrame,
    budget: float,
    max_avg_pd: float,
    pd_col: str = "pd",
    lgd: float = 0.60,
):
    """
    Binary optimization:
    - Decision: x_i ∈ {0,1} lend or not lend to borrower i
    - Objective:
        Maximize risk-adjusted expected profit
    - Constraints:
        1) Sum(total_debt_i * x_i) <= budget
        2) Exposure-weighted PD <= max_avg_pd
    """

    if not ORTOOLS_AVAILABLE:
        return None, "OR-Tools is not installed. Run: pip install ortools"

    df_opt = df.copy().reset_index(drop=True)

    exposure = df_opt["total_debt"].values
    pd_vals = df_opt[pd_col].values

    spread_map = {
        "BBB": 0.03,
        "BB": 0.05,
        "B": 0.07,
        "CCC": 0.10,
    }
    spread_vals = df_opt["rating_bucket"].map(spread_map).fillna(0.06).values

    expected_profit = exposure * (spread_vals - pd_vals * lgd)

    n = len(df_opt)
    if n == 0:
        return None, "No borrowers available for optimization."

    solver = pywraplp.Solver.CreateSolver("CBC")
    if solver is None:
        return None, "Failed to create OR-Tools CBC solver."

    x = [solver.BoolVar(f"x_{i}") for i in range(n)]

    # Capital budget
    solver.Add(sum(exposure[i] * x[i] for i in range(n)) <= budget)

    # Exposure-weighted PD cap
    solver.Add(
        sum(pd_vals[i] * exposure[i] * x[i] for i in range(n))
        <= (max_avg_pd / 100.0) * budget
    )

    # Objective: maximize risk-adjusted expected profit
    solver.Maximize(sum(expected_profit[i] * x[i] for i in range(n)))

    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return None, "No feasible solution found for these settings."

    selected_flags = np.array([x[i].solution_value() for i in range(n)]) > 0.5
    df_opt["selected"] = selected_flags
    df_opt["_pd_used"] = df_opt[pd_col]

    sel = df_opt[df_opt["selected"]]
    if len(sel) == 0:
        return None, "Optimization returned no selected borrowers. Try relaxing constraints."

    total_exposure = sel["total_debt"].sum()
    weighted_pd = (sel["_pd_used"] * sel["total_debt"]).sum() / total_exposure

    sel_spread = sel["rating_bucket"].map(spread_map).fillna(0.06)
    total_profit = (sel["total_debt"] * (sel_spread - sel["_pd_used"] * lgd)).sum()

    stats = {
        "num_selected": int(len(sel)),
        "total_exposure": float(total_exposure),
        "weighted_avg_pd": float(weighted_pd),
        "total_expected_profit": float(total_profit),
        "pd_col": pd_col,
        "lgd": lgd,
    }

    return (df_opt, stats), None

# --------------------------------------------------
# OPENAI HELPERS FOR AI MEMOS (new client)
# --------------------------------------------------

def _fmt(val, digits=1):
    """Safe numeric formatter for prompts."""
    try:
        if pd.isna(val):
            return "N/A"
        return f"{float(val):.{digits}f}"
    except Exception:
        return "N/A"


def generate_quick_summary(base_row: pd.Series, stress_row: pd.Series, scenario_name: str,
                           rev_decline: float, ir_increase: float, liq_decline: float) -> str:
    company = base_row["company_id"]
    rating = base_row["rating_bucket"]
    pd_base = _fmt(base_row["pd"] * 100, 1)
    pd_stress = _fmt(stress_row["pd_stressed"] * 100, 1)

    lev_base = _fmt(base_row["leverage"], 1)
    cov_base = _fmt(base_row["coverage"], 1)
    liq_base = _fmt(base_row["liquidity_ratio"], 2)

    lev_stress = _fmt(stress_row["leverage_stressed"], 1)
    cov_stress = _fmt(stress_row["coverage_stressed"], 1)
    liq_stress = _fmt(stress_row["liquidity_ratio_stressed"], 2)

    risk_base = assign_risk_level(base_row)
    risk_stress = stress_row["Risk_Level_Stressed"]

    prompt = f"""
You are a middle-market corporate credit analyst. Write a concise 4–6 sentence narrative summary for the following borrower.

Focus on:
- overall credit profile and current rating bucket,
- leverage, interest coverage, and liquidity,
- how the stress scenario changes risk (directionally),
- whether this name still fits a typical bank lending portfolio.

Borrower: {company}
Initial internal rating bucket: {rating}
Base-case PD (1-yr): {pd_base}%
Base leverage: {lev_base}x
Base interest coverage: {cov_base}x
Base liquidity ratio (cash / debt): {liq_base}
Base risk band: {risk_base}

Stress scenario: {scenario_name}
Assumptions: revenue –{rev_decline}%, interest expense +{ir_increase}%, cash –{liq_decline}%

Stressed leverage: {lev_stress}x
Stressed coverage: {cov_stress}x
Stressed liquidity: {liq_stress}
Stressed PD (1-yr): {pd_stress}%
Stressed risk band: {risk_stress}

Write in a neutral, professional tone (no bullets, no emojis).
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a corporate credit analyst at a global bank."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def generate_full_credit_memo(base_row: pd.Series, stress_row: pd.Series, scenario_name: str,
                              rev_decline: float, ir_increase: float, liq_decline: float) -> str:
    company = base_row["company_id"]
    rating = base_row["rating_bucket"]
    pd_base = _fmt(base_row["pd"] * 100, 1)
    pd_stress = _fmt(stress_row["pd_stressed"] * 100, 1)

    rev = _fmt(base_row["revenue"], 0)
    ebitda = _fmt(base_row["ebitda"], 0)
    ebitda_margin = _fmt(base_row["ebitda_margin"], 1)

    lev_base = _fmt(base_row["leverage"], 1)
    cov_base = _fmt(base_row["coverage"], 1)
    liq_base = _fmt(base_row["liquidity_ratio"], 2)

    lev_stress = _fmt(stress_row["leverage_stressed"], 1)
    cov_stress = _fmt(stress_row["coverage_stressed"], 1)
    liq_stress = _fmt(stress_row["liquidity_ratio_stressed"], 2)

    risk_base = assign_risk_level(base_row)
    risk_stress = stress_row["Risk_Level_Stressed"]

    prompt = f"""
You are drafting an internal middle-market **credit memo** for a bank credit committee.

Structure the memo into short sections with headings, for example:
1) Business & Competitive Profile
2) Historical Financial Profile
3) Capital Structure, Leverage & Coverage
4) Liquidity, Covenants & Refinancing Risk
5) Stress Scenario Assessment ({scenario_name})
6) Indicative Rating View & Recommended Lending Stance

Borrower: {company}
Current internal rating bucket: {rating}
Base-case PD (1-yr): {pd_base}%
Revenue (latest): ${rev}m
EBITDA (latest): ${ebitda}m
EBITDA margin: {ebitda_margin}%
Base leverage (Debt/EBITDA): {lev_base}x
Base interest coverage (EBITDA/interest): {cov_base}x
Base liquidity ratio (cash / debt): {liq_base}
Base risk band: {risk_base}

Stress scenario: {scenario_name}
Assumptions: revenue –{rev_decline}%, interest expense +{ir_increase}%, cash –{liq_decline}%

Stressed leverage: {lev_stress}x
Stressed coverage: {cov_stress}x
Stressed liquidity: {liq_stress}
Stressed PD (1-yr): {pd_stress}%
Stressed risk band: {risk_stress}

In your memo:
- Briefly describe the scale and type of business (infer from metrics if needed).
- Discuss leverage, coverage, and liquidity in the base case.
- Explain how the stress scenario affects credit risk, especially coverage and liquidity.
- Provide an indicative rating range (e.g., BBB-/BB+/BB/BB-) and explain the main rating constraint.
- Conclude with a clear recommended lending stance (e.g., proceed with covenants, proceed at reduced size, or avoid) and 2–3 key conditions (covenants, leverage limit, minimum liquidity, or reporting requirements).

Write in a concise, professional, analytical tone. Use short paragraphs (no bullet lists).
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a conservative, fundamentals-focused corporate credit analyst."
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def generate_lending_conditions(base_row: pd.Series, stress_row: pd.Series, scenario_name: str) -> str:
    company = base_row["company_id"]
    lev = _fmt(base_row["leverage"], 1)
    cov = _fmt(base_row["coverage"], 1)
    liq = _fmt(base_row["liquidity_ratio"], 2)
    lev_st = _fmt(stress_row["leverage_stressed"], 1)
    cov_st = _fmt(stress_row["coverage_stressed"], 1)
    liq_st = _fmt(stress_row["liquidity_ratio_stressed"], 2)

    prompt = f"""
You are a senior credit officer reviewing lending terms for a middle-market borrower.

Borrower: {company}
Base leverage: {lev}x
Base coverage: {cov}x
Base liquidity ratio: {liq}
Stressed leverage ({scenario_name}): {lev_st}x
Stressed coverage: {cov_st}x
Stressed liquidity ratio: {liq_st}

Provide a short paragraph plus 3–5 clearly written covenant / lending conditions that would make this an acceptable bank loan (for example: maximum leverage, minimum interest coverage, minimum cash balance or liquidity, limits on additional debt, enhanced reporting).

Write in text form (no bullet markers like "-", just sentences separated clearly).
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a senior committee member setting loan covenants."
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
    )
    return response.choices[0].message.content.strip()

# --------------------------------------------------
# PAGE: OVERVIEW
# --------------------------------------------------

def page_overview(df: pd.DataFrame):
    section_title("Portfolio Snapshot")

    # Top metric cards
    col1, col2, col3, col4 = st.columns(4)

    num_borrowers = df["company_id"].nunique()
    med_leverage = df["leverage"].median()
    med_coverage = df["coverage"].median()
    avg_pd = df["pd"].mean() * 100

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Borrowers</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{num_borrowers}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Median Leverage</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{med_leverage:.1f}x</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Median Coverage</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{med_coverage:.1f}x</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Average PD (1-yr)</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{avg_pd:.1f}%</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    section_title("Rating Mix (Base Case)")
    rating_counts = df["rating_bucket"].value_counts().sort_index()
    fig = px.pie(
        names=rating_counts.index,
        values=rating_counts.values,
        title="Rating Distribution (BBB / BB / B / CCC)",
        hole=0.4,
    )
    st.plotly_chart(fig, use_container_width=True)

    section_title("Top 5 Exposures (by Total Debt)")
    top5 = df.sort_values("total_debt", ascending=False).head(5)[
        ["company_id", "total_debt", "rating_bucket", "pd", "leverage", "coverage"]
    ].copy()
    top5["PD (%)"] = (top5["pd"] * 100).round(1)
    top5["Total Debt"] = top5["total_debt"].round(0)

    top5 = top5.rename(
        columns={
            "company_id": "Company",
            "rating_bucket": "Rating",
            "leverage": "Leverage (x)",
            "coverage": "Coverage (x)",
        }
    ).drop(columns=["total_debt", "pd"])



    st.dataframe(top5, use_container_width=True)

# --------------------------------------------------
# PAGE: DATA & SETUP (with template, no mapping)
# --------------------------------------------------

def page_data(df: pd.DataFrame):
    section_title("Data & Setup")
    st.write(
        "Use the bundled synthetic corporate borrower dataset or upload your own file "
        "following the same schema."
    )

    # Downloadable CSV template
    st.markdown("#### Download CSV Template")
    template_csv = create_template_csv()
    st.download_button(
        label="Download borrower template CSV",
        data=template_csv,
        file_name="borrower_template.csv",
        mime="text/csv",
    )

    st.markdown("---")

    choice = st.radio(
        "Choose data source:",
        ["Use bundled synthetic dataset", "Upload CSV"],
        index=0
    )

    if choice == "Upload CSV":
        file = st.file_uploader("Upload borrower CSV (must match template schema)", type=["csv"])
        if file:
            user_df = pd.read_csv(file)
            valid, missing = validate_schema(user_df)

            if valid:
                st.success("Schema validated – using uploaded data (for this page only, not persisted).")
                st.dataframe(user_df.head(10))
            else:
                st.error(f"Missing required columns: {missing}")
                st.info(
                    "Please align your file with the downloadable template above. "
                    "Column mapping has been disabled to keep the pipeline simple and consistent."
                )
    else:
        st.success("Using bundled synthetic dataset (no real borrower data).")
        st.dataframe(df.head(10))

    st.info(
        "🔒 **Confidentiality Notice:** All uploaded data is processed locally in this session. "
        "No borrower identifiers or raw financial statements are transmitted outside the app. "
        "Only derived ratios may be used for AI explanations."
    )

# --------------------------------------------------
# PAGE: RISK & STRESS TESTING
# --------------------------------------------------

def page_risk_stress(df: pd.DataFrame):
    section_title("Risk Metrics & Structured Stress Testing")

    df = df.copy()
    df["Risk_Level"] = df.apply(assign_risk_level, axis=1)

    # Sidebar stress controls
    st.sidebar.header("Stress Test Assumptions")
    scenario = st.sidebar.selectbox(
        "Scenario preset (Stress Page)",
        ["Base Case", "Mild Downturn", "Recession", "Custom"]
    )

    if scenario == "Base Case":
        rev, ir, liq = 0, 0, 0
    elif scenario == "Mild Downturn":
        rev, ir, liq = 10, 10, 20
    elif scenario == "Recession":
        rev, ir, liq = 20, 25, 40
    else:
        rev = st.sidebar.slider("Revenue decline (%)", 0, 50, 10)
        ir = st.sidebar.slider("Interest expense increase (%)", 0, 50, 10)
        liq = st.sidebar.slider("Liquidity (cash) decline (%)", 0, 80, 20)

    df_stress = apply_stress(df, rev, ir, liq)

    tab1, tab2, tab3 = st.tabs(
        ["📊 Portfolio Overview", "👤 Borrower Detail", "⚠️ Stress Impact"]
    )

    # TAB 1 – Portfolio Overview
    with tab1:
        section_title("Portfolio Overview")

        if scenario == "Base Case":
            view_df = df
            risk_col = "Risk_Level"
            pd_col = "pd"
            st.caption("Showing base case portfolio metrics.")
        else:
            view_df = df_stress
            risk_col = "Risk_Level_Stressed"
            pd_col = "pd_stressed"
            st.caption(
                f"Showing stressed metrics under '{scenario}' "
                f"(Rev↓ {rev}%, Interest↑ {ir}%, Cash↓ {liq}%)."
            )

        # Charts
        risk_counts = view_df[risk_col].value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
        fig_risk = px.bar(
            risk_counts,
            labels={"index": "Risk Level", "value": "Count"},
            title="Risk Level Distribution"
        )
        st.plotly_chart(fig_risk, use_container_width=True)

        fig_pd = px.histogram(
            view_df,
            x=pd_col,
            nbins=20,
            title="PD Distribution",
            labels={pd_col: "PD (1-year)"}
        )
        st.plotly_chart(fig_pd, use_container_width=True)

        # ---- New borrower table with assigned risk band ----
        section_title("Borrower-Level View")

        if scenario == "Base Case":
            table_df = view_df[
                [
                    "company_id",
                    "rating_bucket",
                    "pd",
                    "leverage",
                    "coverage",
                    "liquidity_ratio",
                    "Risk_Level",
                ]
            ].copy()
            table_df["PD (1-yr, %)"] = (table_df["pd"] * 100).round(1)
            table_df = table_df.rename(
                columns={
                    "company_id": "Borrower",
                    "rating_bucket": "Rating",
                    "leverage": "Leverage (x)",
                    "coverage": "Coverage (x)",
                    "liquidity_ratio": "Liquidity Ratio",
                    "Risk_Level": "Risk Band",
                }
            ).drop(columns=["pd"])
        else:
            table_df = view_df[
                [
                    "company_id",
                    "rating_bucket",
                    "pd_stressed",
                    "leverage_stressed",
                    "coverage_stressed",
                    "liquidity_ratio_stressed",
                    "Risk_Level_Stressed",
                ]
            ].copy()
            table_df["PD (1-yr, %, stressed)"] = (table_df["pd_stressed"] * 100).round(1)
            table_df = table_df.rename(
                columns={
                    "company_id": "Borrower",
                    "rating_bucket": "Rating",
                    "leverage_stressed": "Leverage (x, stressed)",
                    "coverage_stressed": "Coverage (x, stressed)",
                    "liquidity_ratio_stressed": "Liquidity Ratio (stressed)",
                    "Risk_Level_Stressed": "Risk Band (stressed)",
                }
            ).drop(columns=["pd_stressed"])

        st.dataframe(table_df, use_container_width=True)
  


    # TAB 2 – Borrower Detail
    with tab2:
        section_title("Borrower Detail")

        selected = st.selectbox("Select borrower:", df["company_id"].unique())
        base_row = df[df["company_id"] == selected].iloc[0]
        stress_row = df_stress[df_stress["company_id"] == selected].iloc[0]

        colA, colB = st.columns(2)

        with colA:
            st.markdown("### Base Case")
            st.write(f"Leverage: {base_row['leverage']:.2f}x")
            st.write(f"Coverage: {base_row['coverage']:.2f}x")
            st.write(f"Liquidity: {base_row['liquidity_ratio']:.2f}")
            st.write(f"Rating: {base_row['rating_bucket']}")
            st.write(f"PD: {base_row['pd']*100:.1f}%")
            st.write(f"Risk Level: {risk_color(base_row['Risk_Level'])}")
            st.markdown("**Risk Drivers (Base):**")
            st.write(risk_driver_text(base_row))

        with colB:
            st.markdown(f"### Stressed Case ({scenario})")
            st.write(f"Leverage: {stress_row['leverage_stressed']:.2f}x")
            st.write(f"Coverage: {stress_row['coverage_stressed']:.2f}x")
            st.write(f"Liquidity: {stress_row['liquidity_ratio_stressed']:.2f}")
            st.write(f"PD (stressed): {stress_row['pd_stressed']*100:.1f}%")
            st.write(f"Risk Level (stressed): {risk_color(stress_row['Risk_Level_Stressed'])}")

    # TAB 3 – Stress Impact
    with tab3:
        section_title("Stress Impact Across Portfolio")

        base_counts = df["Risk_Level"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
        stress_counts = df_stress["Risk_Level_Stressed"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0)

        df_compare = pd.DataFrame({
            "Risk Level": ["Low", "Medium", "High"],
            "Base": base_counts.values,
            "Stressed": stress_counts.values,
        })

        fig2 = px.bar(
            df_compare.melt(id_vars="Risk Level", var_name="Scenario", value_name="Count"),
            x="Risk Level", y="Count", color="Scenario", barmode="group",
            title="Base vs Stressed Risk Levels"
        )
        st.plotly_chart(fig2, use_container_width=True)

        movers = pd.DataFrame({
            "company_id": df["company_id"],
            "Base_Score": df["Risk_Level"].map({"Low":1,"Medium":2,"High":3}),
            "Stressed_Score": df_stress["Risk_Level_Stressed"].map({"Low":1,"Medium":2,"High":3}),
        })
        movers["Change"] = movers["Stressed_Score"] - movers["Base_Score"]
        worst = movers.sort_values("Change", ascending=False).head(5)

        st.markdown("### Borrowers Most Impacted by Stress")
        st.dataframe(worst, use_container_width=True)

        moved_to_high = (
            (df["Risk_Level"] != "High") &
            (df_stress["Risk_Level_Stressed"] == "High")
        ).sum()

        st.markdown("### Scenario Summary")
        if moved_to_high == 0:
            st.write(
                "Under this scenario, no new borrowers migrate into High risk; "
                "the portfolio appears relatively resilient to these shocks in this simplified framework."
            )
        else:
            st.write(
                f"Under this scenario, {moved_to_high} borrowers migrate into High risk. "
                "These names would be top priorities for follow-up review by a credit analyst."
            )

# --------------------------------------------------
# PAGE: OPTIMIZATION (with stressed PDs & controls, profit only)
# --------------------------------------------------

def page_optimization(df: pd.DataFrame):
    section_title("Portfolio Optimization (OR-Tools)")

    if not ORTOOLS_AVAILABLE:
        st.error("OR-Tools is not installed. Run `pip install ortools` in the same environment.")
        return

    st.write(
        "This optimizer selects a set of borrowers to **lend to** by maximizing a risk-adjusted expected profit, "
        "subject to a capital budget (maximum total exposure) and a cap on the portfolio’s average PD."
    )

    st.markdown("---")
    use_stress = st.checkbox("Optimize using **stressed PDs** under a scenario", value=False)

    # -----------------------------
    # Choose base vs stressed PDs
    # -----------------------------
    if use_stress:
        scenario = st.selectbox(
            "Optimization scenario preset",
            ["Mild Downturn", "Recession", "Custom"],
            index=0
        )
        if scenario == "Mild Downturn":
            rev, ir, liq = 10, 10, 20
        elif scenario == "Recession":
            rev, ir, liq = 20, 25, 40
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                rev = st.slider("Revenue decline (%)", 0, 50, 10, key="opt_rev")
            with c2:
                ir = st.slider("Interest expense increase (%)", 0, 50, 10, key="opt_ir")
            with c3:
                liq = st.slider("Liquidity (cash) decline (%)", 0, 80, 20, key="opt_liq")

        df_for_opt = apply_stress(df, rev, ir, liq)
        pd_col = "pd_stressed"
        st.caption(
            f"Optimization uses stressed PDs under scenario "
            f"(Rev↓ {rev}%, Interest↑ {ir}%, Cash↓ {liq}%)."
        )
    else:
        df_for_opt = df.copy()
        pd_col = "pd"
        st.caption("Optimization uses base-case PDs.")

    # -----------------------------
    # Portfolio-level stats
    # -----------------------------
    total_debt = df_for_opt["total_debt"].sum()
    avg_pd = df_for_opt[pd_col].mean() * 100  # in %

    # -----------------------------
    # Optimization controls
    # -----------------------------
    col1, col2 = st.columns(2)
    with col1:
        budget = st.number_input(
            "Capital budget (maximum total portfolio exposure)",
            min_value=0.0,
            max_value=float(total_debt),
            value=float(min(total_debt * 0.4, total_debt)),  # default: 40% of book
            step=1_000_000.0,
            format="%.0f",
            help="Upper limit on the sum of total_debt across all selected borrowers."
        )
    with col2:
        max_avg_pd = st.slider(
            "Max allowed weighted-average portfolio PD (%)",
            min_value=1.0,
            max_value=float(max(30.0, avg_pd + 5)),
            value=min(float(avg_pd + 3), 12.0),
            step=0.5,
            help="Caps the exposure-weighted average PD of the optimized portfolio."
        )

    lgd_pct = st.slider(
        "Loss Given Default (LGD, %)",
        min_value=30,
        max_value=70,
        value=60,
        step=5,
        help="Used in expected profit calculation as PD × LGD."
    )

    # -----------------------------
    # Run optimization
    # -----------------------------
    if st.button("Run Optimization"):
        result, error = optimize_portfolio(
            df_for_opt,
            budget,
            max_avg_pd,
            pd_col=pd_col,
            lgd=lgd_pct / 100.0,
        )
        if error:
            st.error(error)
            return

        df_opt, stats = result

        # -------------------------
        # Summary metrics
        # -------------------------
        section_title("Optimization Results – Portfolio Summary")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Selected Borrowers", stats["num_selected"])
        with c2:
            st.metric("Total Exposure", f"${stats['total_exposure']:,.0f}")
        with c3:
            st.metric("Weighted Avg PD", f"{stats['weighted_avg_pd']*100:.1f}%")
        with c4:
            st.metric("Expected Profit (approx.)", f"${stats['total_expected_profit']:,.0f}")

        st.caption(
            f"Objective: **Maximize risk-adjusted expected profit**, "
            f"LGD = {int(stats['lgd']*100)}%, PD source = `{stats['pd_col']}`."
        )

        st.markdown("---")
        section_title("Selected vs Non-Selected Borrowers")

        sel = df_opt[df_opt["selected"]]
        not_sel = df_opt[~df_opt["selected"]]

        st.markdown("**Selected Borrowers (top 15 by exposure)**")
        st.dataframe(
            sel.sort_values("total_debt", ascending=False)
               .head(15)[["company_id", "total_debt", "rating_bucket", "_pd_used", "leverage", "coverage"]],  # type: ignore[index]
            use_container_width=True,
        )

        with st.expander("View Non-Selected Borrowers"):
            st.dataframe(
                not_sel.sort_values("_pd_used", ascending=False)[
                    ["company_id", "total_debt", "rating_bucket", "_pd_used", "leverage", "coverage"]
                ],
                use_container_width=True,
            )

        # -------------------------
        # PD vs exposure scatter
        # -------------------------
        section_title("PD vs Exposure – Optimization Decisions")
        plot_df = df_opt.copy()
        plot_df["pd_pct"] = plot_df["_pd_used"] * 100
        plot_df["Selected"] = np.where(plot_df["selected"], "Selected", "Not Selected")

        fig = px.scatter(
            plot_df,
            x="pd_pct",
            y="total_debt",
            color="Selected",
            hover_data=["company_id", "rating_bucket", "leverage", "coverage"],
            labels={"pd_pct": "PD (%)", "total_debt": "Total Debt"},
            title="PD vs Exposure with Optimization Decisions",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "The solver tends to favor borrowers with stronger risk-adjusted returns while respecting "
            "capital and portfolio PD constraints, similar to how a middle-market credit portfolio manager "
            "would construct a loan book."
        )
    else:
        st.info(
            "Choose whether to optimize on base or stressed PDs, set your capital budget, PD limit, and LGD, "
            "then click **Run Optimization**."
        )


# --------------------------------------------------
# PAGE: AI CREDIT MEMO (styled & downloadable)
# --------------------------------------------------

def page_ai(df: pd.DataFrame):
    section_title("AI Credit Memo & Lending View")

    if client is None:
        st.error(
            "No OpenAI API key found. Please add `OPENAI_API_KEY` to a `.env` file "
            "in the project folder to enable AI credit memos."
        )
        return

    df = df.copy()
    df["Risk_Level"] = df.apply(assign_risk_level, axis=1)

    st.write(
        "Select a borrower and scenario. The app will combine base and stressed metrics to generate "
        "a narrative credit summary, a full memo, and suggested lending conditions using GenAI."
    )

    # Scenario presets – mirror Risk & Stress page
    scenario = st.selectbox(
        "Scenario preset",
        ["Base Case", "Mild Downturn", "Recession", "Custom"],
        index=1
    )

    if scenario == "Base Case":
        rev, ir, liq = 0, 0, 0
    elif scenario == "Mild Downturn":
        rev, ir, liq = 10, 10, 20
    elif scenario == "Recession":
        rev, ir, liq = 20, 25, 40
    else:
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            rev = st.slider("Revenue decline (%)", 0, 50, 10)
        with col_s2:
            ir = st.slider("Interest expense increase (%)", 0, 50, 10)
        with col_s3:
            liq = st.slider("Liquidity (cash) decline (%)", 0, 80, 20)

    df_stress = apply_stress(df, rev, ir, liq)

    selected = st.selectbox("Select borrower:", df["company_id"].unique())
    base_row = df[df["company_id"] == selected].iloc[0]
    stress_row = df_stress[df_stress["company_id"] == selected].iloc[0]

    colA, colB = st.columns(2)
    with colA:
        st.markdown("### Base Case Snapshot")
        st.write(f"Rating bucket: {base_row['rating_bucket']}")
        st.write(f"PD (1-yr): {base_row['pd']*100:.1f}%")
        st.write(f"Leverage: {base_row['leverage']:.2f}x")
        st.write(f"Coverage: {base_row['coverage']:.2f}x")
        st.write(f"Liquidity ratio: {base_row['liquidity_ratio']:.2f}")
        st.write(f"Risk band: {risk_color(base_row['Risk_Level'])}")

    with colB:
        st.markdown(f"### Stressed Snapshot ({scenario})")
        st.write(f"Stressed PD (1-yr): {stress_row['pd_stressed']*100:.1f}%")
        st.write(f"Stressed leverage: {stress_row['leverage_stressed']:.2f}x")
        st.write(f"Stressed coverage: {stress_row['coverage_stressed']:.2f}x")
        st.write(f"Stressed liquidity: {stress_row['liquidity_ratio_stressed']:.2f}")
        st.write(f"Stressed risk band: {risk_color(stress_row['Risk_Level_Stressed'])}")

    st.markdown("---")
    st.subheader("Generate AI Narrative")

    col_b1, col_b2, col_b3 = st.columns(3)
    memo_container = st.container()

    # Ensure session_state keys exist
    if "latest_memo" not in st.session_state:
        st.session_state["latest_memo"] = ""
        st.session_state["latest_memo_type"] = ""

    # Buttons ONLY update session_state
    with col_b1:
        if st.button("Quick Summary"):
            with st.spinner("Generating quick AI summary..."):
                try:
                    text = generate_quick_summary(base_row, stress_row, scenario, rev, ir, liq)
                    st.session_state["latest_memo"] = text
                    st.session_state["latest_memo_type"] = "Quick Summary"
                except Exception as e:
                    st.error(f"Error calling OpenAI: {e}")

    with col_b2:
        if st.button("Full Credit Memo"):
            with st.spinner("Generating full AI credit memo..."):
                try:
                    text = generate_full_credit_memo(base_row, stress_row, scenario, rev, ir, liq)
                    st.session_state["latest_memo"] = text
                    st.session_state["latest_memo_type"] = "Full Credit Memo"
                except Exception as e:
                    st.error(f"Error calling OpenAI: {e}")

    with col_b3:
        if st.button("Lending Conditions & Covenants"):
            with st.spinner("Generating suggested lending conditions..."):
                try:
                    text = generate_lending_conditions(base_row, stress_row, scenario)
                    st.session_state["latest_memo"] = text
                    st.session_state["latest_memo_type"] = "Lending Conditions & Covenants"
                except Exception as e:
                    st.error(f"Error calling OpenAI: {e}")

    # Single display + download, based on latest memo
    if st.session_state.get("latest_memo"):
        memo_title = st.session_state.get("latest_memo_type", "Most Recent Memo")
        memo_text = st.session_state["latest_memo"]

        memo_container.markdown(f"#### {memo_title}")
        memo_container.markdown(
            f'<div class="memo-box">{memo_text}</div>',
            unsafe_allow_html=True,
        )

        st.download_button(
            label="Download memo as .txt",
            data=memo_text,
            file_name=f"{selected}_{memo_title.replace(' ','_').lower()}.txt",
            mime="text/plain",
        )


    st.info(
        "These GenAI outputs are drafted from ratios and PDs only; a human credit analyst would refine, "
        "challenge, and ultimately own the final credit memo and lending decision."
    )

# --------------------------------------------------
# PAGE: ABOUT / PRIVACY
# --------------------------------------------------

def page_about():
    section_title("About This Application")

    st.markdown("""
This application is a proof-of-concept **AI Credit Analyst Assistant** designed to demonstrate how 
GenAI, Python-based financial modeling, OR-Tools optimization, and credit risk analytics can come 
together to support institutional-quality lending decisions.

### 🔍 Purpose
The app evaluates corporate borrowers using:
- Financial ratios (leverage, coverage, liquidity)
- Probability of Default estimates (PD)
- Macro stress testing (revenue, liquidity, interest shocks)
- Generative AI summaries, memos, and rating explanations
- OR-Tools portfolio optimization (lend / not-lend decisions)

It replicates key components of a **middle-market corporate lending workflow**, including underwriting,
risk segmentation, scenario testing, and portfolio construction.

### 🔒 Confidentiality & Data Handling
This tool is designed with confidentiality in mind:

- **All uploaded data is processed locally** within the user session.
- **No borrower data is stored, transmitted, or logged externally.**
- The generative AI module receives **only derived ratios** (e.g., leverage, coverage, PD) — 
  *never raw financial statements or company identifiers.*
- Data remains in-memory and is discarded when the session ends.
- The tool can be run entirely inside a secure environment, making it compatible with 
  enterprise privacy standards.

### 📌 Assumptions & Model Limitations
- Probability of default (PD) is an approximation for demonstration purposes.
- Expected profit is modeled as spread – (PD × LGD), with LGD assumed at 60% (configurable).
- Optimization focuses on binary lending decisions and does not model amortization or multi-period credit behavior.
- Stress testing is simplified but directionally realistic for middle-market credit.

### 🛠️ Technologies Used
- **Python** for data and modeling
- **Streamlit** for the interactive UI
- **Pandas / NumPy** for analytics
- **Plotly** for visualizations
- **OR-Tools** for optimization
- **OpenAI API** for credit memos & narrative analysis

### ✨ Project Goal
To demonstrate how modern AI and optimization can enhance traditional credit analysis by providing:
- Richer insights  
- Faster underwriting  
- Transparent risk explanations  
- Optimized lending decisions  

This project is educational and not intended for live, production credit decisioning.
""")

# --------------------------------------------------
# MAIN APP
# --------------------------------------------------

def main():
    # Global header banner (now dark blue)
    st.markdown(
        """
        <div class="app-header">
            <div class="app-header-title">AI Credit Analyst Assistant</div>
            <div class="app-header-subtitle">
                Middle-Market Corporate Lending · Risk & Stress Testing · Portfolio Optimization · GenAI Memos
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = load_sample_data()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to:",
        ["Overview", "Data & Setup", "Risk & Stress Testing", "Optimization", "AI Credit Memo", "About"]
    )

    if page == "Overview":
        page_overview(df)
    elif page == "Data & Setup":
        page_data(df)
    elif page == "Risk & Stress Testing":
        page_risk_stress(df)
    elif page == "Optimization":
        page_optimization(df)
    elif page == "AI Credit Memo":
        page_ai(df)
    elif page == "About":
        page_about()

    # Footer
    st.markdown(
        """
        <div class="app-footer">
            For educational use only – synthetic data and simplified models. Not for real credit decisioning.
        </div>
        """,
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
