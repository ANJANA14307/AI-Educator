from __future__ import annotations

import hashlib
import io
import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "apachejit_processed.csv"
RAW_TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "apachejit_train.csv"
FUTURE_LARGE_PATH = PROJECT_ROOT / "data" / "raw" / "apachejit_test_large.csv"
FUTURE_SMALL_PATH = PROJECT_ROOT / "data" / "raw" / "apachejit_test_small.csv"
METRICS_PATH = PROJECT_ROOT / "reports" / "evaluation_metrics.json"
SHAP_PATH = PROJECT_ROOT / "reports" / "shap_feature_importance.csv"
CONFUSION_MATRIX_PATH = PROJECT_ROOT / "reports" / "confusion_matrix.png"
ROC_CURVE_PATH = PROJECT_ROOT / "reports" / "roc_curve.png"
PRECISION_RECALL_PATH = PROJECT_ROOT / "reports" / "precision_recall_curve.png"
SHAP_IMPORTANCE_PATH = PROJECT_ROOT / "reports" / "shap_feature_importance.png"
SHAP_SUMMARY_PATH = PROJECT_ROOT / "reports" / "shap_summary.png"
API_URL = "http://127.0.0.1:8000"
LOGGER = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "la", "ld", "nf", "nd", "ns", "ent",
    "ndev", "age", "nuc", "aexp", "arexp", "asexp",
]
INTEGER_FEATURES = {"la", "ld", "nf", "nd", "ns", "aexp"}
FEATURE_DETAILS = {
    "la": ("Lines added", "Number of lines added in the commit."),
    "ld": ("Lines deleted", "Number of lines deleted in the commit."),
    "nf": ("Files changed", "Number of files modified by the commit."),
    "nd": ("Directories changed", "Number of directories modified."),
    "ns": ("Subsystems changed", "Number of subsystems modified."),
    "ent": ("Code entropy", "Entropy of the modified code."),
    "ndev": ("Developers", "Number of developers who changed the files."),
    "age": ("File age", "Average age of the modified files."),
    "nuc": ("Unique changes", "Number of unique changes to the modified files."),
    "aexp": ("Author experience", "Experience of the commit author."),
    "arexp": ("Author review experience", "Author review experience."),
    "asexp": ("Subsystem experience", "The author's experience in the affected subsystem."),
}
EXAMPLE_VALUES = {
    "la": 20,
    "ld": 10,
    "nf": 5,
    "nd": 3,
    "ns": 2,
    "ent": 1.5,
    "ndev": 4,
    "age": 30,
    "nuc": 3,
    "aexp": 2,
    "arexp": 1,
    "asexp": 2,
}
NAV_GROUPS = {
    "WORKSPACE": ["Overview", "Predict a Commit", "Batch Prediction"],
    "ANALYSIS": ["Model Insights", "Dataset Explorer"],
    "HELP": ["How It Works"],
}
NAV_ICONS = {
    "Overview": "▦",
    "Predict a Commit": "＋",
    "Batch Prediction": "⇧",
    "Model Insights": "◷",
    "Dataset Explorer": "▤",
    "How It Works": "?",
}
COLORS = {
    "dark": "#141414",
    "light_grey": "#E8E8E8",
    "secondary_grey": "#B4B4B4",
    "muted": "#5A5A5A",
    "green": "#50AD77",
    "deep_green": "#066E26",
    "coral": "#FF8051",
    "canvas": "#F4F5F3",
    "white": "#FFFFFF",
}

st.set_page_config(
    page_title="Software Defect Prediction System",
    layout="wide",
    initial_sidebar_state="auto",
)

show_splash = not st.session_state.get("splash_seen", False)
st.session_state["splash_seen"] = True

st.markdown(
    """
    <style>
    :root {
        --ink: #141414;
        --paper: #F4F5F3;
        --surface: #FFFFFF;
        --line: #E8E8E8;
        --muted: #5A5A5A;
        --quiet: #B4B4B4;
        --green: #50AD77;
        --green-deep: #066E26;
        --coral: #FF8051;
    }
    html, body, [class*="css"] {
        font-family: Inter, "Segoe UI", Arial, sans-serif;
        color: var(--ink);
    }
    .stApp { background: var(--paper); color: var(--ink); }
    .main .block-container {
        max-width: 1400px;
        padding: 2.25rem 2.5rem 3.5rem;
    }
    header[data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"] { background: transparent; }
    [data-testid="stAppDeployButton"], [data-testid="stMainMenu"] { display: none !important; }
    [data-testid="stSidebarCollapseButton"] { visibility: visible !important; }
    [data-testid="stDecoration"] { display: none; }
    section[data-testid="stSidebar"] {
        background: var(--ink);
        border-right: 1px solid #292929;
        min-width: 272px !important;
        max-width: 272px !important;
    }
    section[data-testid="stSidebar"] > div:first-child { background: var(--ink); }
    section[data-testid="stSidebar"] .block-container { padding: 1.65rem 1rem 1.4rem; }
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: var(--quiet); }
    .brand-lockup { display: flex; align-items: center; gap: 0.8rem; padding: 0.15rem 0.2rem 0.4rem; }
    .brand-symbol {
        width: 42px; height: 42px; display: grid; place-items: center;
        background: #202820; border: 1px solid #344537; border-radius: 12px;
        flex: 0 0 42px;
    }
    .brand-title { color: #F4F5F3; font-size: 0.98rem; line-height: 1.25; font-weight: 650; letter-spacing: -0.02em; }
    .brand-caption { color: #8A918B; font-size: 0.62rem; letter-spacing: 0.12em; margin-top: 0.32rem; }
    .nav-group {
        color: #858A85; font-size: 0.66rem; font-weight: 650;
        letter-spacing: 0.13em; margin: 1.1rem 0 0.35rem 0.65rem;
    }
    section[data-testid="stSidebar"] .stButton { margin: 0.12rem 0; }
    section[data-testid="stSidebar"] .stButton button {
        min-height: 42px; justify-content: flex-start; text-align: left;
        padding: 0.45rem 0.75rem; border-radius: 9px;
        font-size: 0.88rem; font-weight: 480; box-shadow: none;
        transition: background-color 140ms ease, color 140ms ease, border-color 140ms ease;
    }
    section[data-testid="stSidebar"] .stButton button p {
        color: inherit; margin: 0; text-align: left; width: 100%;
    }
    section[data-testid="stSidebar"] .stButton button[kind="secondary"] {
        color: #B4B4B4; background: transparent; border: 1px solid transparent;
    }
    section[data-testid="stSidebar"] .stButton button[kind="secondary"]:hover {
        color: #F5F6F4; background: #202120; border-color: #303330;
    }
    section[data-testid="stSidebar"] .stButton button[kind="primary"] {
        color: #E8F4EB; background: #203328; border: 1px solid #2D4936;
        border-left: 3px solid var(--green); padding-left: calc(0.75rem - 2px);
    }
    section[data-testid="stSidebar"] .stButton button[kind="primary"]:hover {
        background: #294331; border-color: #3B6248;
    }
    section[data-testid="stSidebar"] hr { border-color: #2B2D2B; margin: 1rem 0; }
    .service-indicator { display: flex; align-items: center; gap: 0.5rem; color: #A7ACA7; font-size: 0.74rem; }
    .service-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--green); display: inline-block; }
    .service-dot.offline { background: var(--coral); }
    h1, h2, h3, h4 { color: var(--ink); letter-spacing: -0.035em; font-weight: 620; }
    h1 { font-size: clamp(1.8rem, 3vw, 2.4rem); line-height: 1.15; }
    h2 { font-size: 1.45rem; }
    h3 { font-size: 1.12rem; }
    p, label, [data-testid="stCaptionContainer"] { color: var(--muted); }
    [data-testid="stCaptionContainer"] { font-size: 0.86rem; }
    .hero {
        position: relative; overflow: hidden;
        background: var(--ink); color: #F4F5F3;
        border: 1px solid #262826; border-radius: 16px;
        padding: 2rem 2.1rem; margin: 0.25rem 0 1.2rem;
        box-shadow: 0 12px 30px rgba(20, 20, 20, 0.08);
    }
    .hero h1 { color: #F5F6F4; max-width: 780px; font-size: clamp(2rem, 3.5vw, 2.8rem); margin: 0.35rem 0 0.65rem; }
    .hero p { color: #B4B4B4; max-width: 720px; margin-bottom: 0; line-height: 1.65; }
    .hero-eyebrow { color: var(--green); font-size: 0.68rem; letter-spacing: 0.14em; font-weight: 650; }
    .muted { color: var(--muted); }
    div[data-testid="stMetric"] {
        background: var(--surface); border: 1px solid var(--line);
        border-radius: 12px; padding: 0.9rem 1rem;
        box-shadow: 0 2px 8px rgba(20, 20, 20, 0.025);
    }
    div[data-testid="stMetric"] label { color: var(--muted); font-size: 0.78rem; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: var(--ink); font-weight: 620; }
    .stButton button, .stDownloadButton button, .stFormSubmitButton button {
        min-height: 42px; border-radius: 9px; font-weight: 560;
        transition: background-color 140ms ease, border-color 140ms ease, transform 140ms ease;
    }
    .stButton button[kind="primary"], .stFormSubmitButton button[kind="primary"] {
        background: var(--green-deep); border: 1px solid var(--green-deep); color: #FFFFFF;
    }
    [data-testid="stBaseButton-primaryFormSubmit"] {
        background: var(--green-deep) !important;
        border: 1px solid var(--green-deep) !important;
        color: #FFFFFF !important;
    }
    [data-testid="stBaseButton-primaryFormSubmit"]:hover {
        background: #05591F !important; border-color: #05591F !important;
    }
    [data-testid="stBaseButton-primaryFormSubmit"] p { color: #FFFFFF !important; }
    .stButton button[kind="primary"] p, .stFormSubmitButton button[kind="primary"] p {
        color: #FFFFFF;
    }
    .stButton button[kind="secondary"] p, .stDownloadButton button p {
        color: var(--ink);
    }
    .stButton button[kind="primary"]:hover, .stFormSubmitButton button[kind="primary"]:hover {
        background: #05591F; border-color: #05591F;
    }
    .stButton button[kind="secondary"], .stDownloadButton button {
        background: #FFFFFF; border: 1px solid #D7DAD6; color: var(--ink);
    }
    .stButton button[kind="secondary"]:hover, .stDownloadButton button:hover {
        border-color: #969D97; background: #F8F9F7;
    }
    .stButton button:focus-visible, .stDownloadButton button:focus-visible,
    .stFormSubmitButton button:focus-visible {
        outline: 2px solid var(--green); outline-offset: 2px;
    }
    [data-testid="stForm"] {
        background: #FFFFFF; border: 1px solid var(--line);
        border-radius: 14px; padding: 1.25rem;
    }
    [data-testid="stNumberInputContainer"] {
        background: #FFFFFF !important; color: var(--ink) !important;
        border: 1px solid #D7DAD6 !important; border-radius: 8px !important;
    }
    [data-testid="stNumberInputContainer"] input {
        background: transparent !important; color: var(--ink) !important;
        -webkit-text-fill-color: var(--ink) !important;
    }
    [data-testid="stNumberInputContainer"] button {
        background: #F4F5F3 !important; color: var(--muted) !important;
        border-left: 1px solid var(--line) !important;
    }
    [data-testid="stNumberInputContainer"] button:hover { background: var(--line) !important; }
    [data-testid="stNumberInputContainer"] button svg { fill: var(--muted) !important; }
    [data-testid="stTextInput"] [data-baseweb="input"],
    [data-testid="stTextArea"] [data-baseweb="textarea"],
    [data-baseweb="select"] > div {
        background: #FFFFFF !important; color: var(--ink) !important;
        border: 1px solid #D7DAD6 !important; border-radius: 8px !important;
    }
    [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea {
        background: transparent !important; color: var(--ink) !important;
        -webkit-text-fill-color: var(--ink) !important;
    }
    input:focus, textarea:focus { border-color: var(--green) !important; box-shadow: 0 0 0 1px var(--green) !important; }
    [data-testid="stDataFrame"], [data-testid="stTable"] { border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }
    [data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 10px; background: #FFFFFF; }
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] { color: var(--green-deep); border-bottom-color: var(--green); }
    [data-testid="stAlert"] { border-radius: 10px; }
    [data-testid="stAlert"][kind="warning"] { border-left-color: var(--coral); }
    [data-testid="stPlotlyChart"] { background: #FFFFFF; border: 1px solid var(--line); border-radius: 12px; padding: 0.35rem; }
    .prediction-verdict {
        display: flex; align-items: center; gap: 0.85rem;
        padding: 1rem 1.1rem; border: 1px solid var(--line);
        border-radius: 12px; margin: 0.75rem 0;
    }
    .prediction-verdict strong { font-size: 1.18rem; font-weight: 650; color: var(--ink); }
    .prediction-verdict.defective { background: #FFF3ED; border-color: #FFD5C2; border-left: 4px solid var(--coral); }
    .prediction-verdict.non-defective { background: #EFF6F0; border-color: #D7E8DA; border-left: 4px solid var(--green-deep); }
    .prediction-verdict .verdict-mark {
        width: 10px; height: 10px; border-radius: 50%; flex: 0 0 10px;
    }
    .prediction-verdict.defective .verdict-mark { background: var(--coral); }
    .prediction-verdict.non-defective .verdict-mark { background: var(--green-deep); }
    .splash-screen {
        position: fixed; inset: 0; z-index: 999999; display: flex;
        align-items: center; justify-content: center; flex-direction: column;
        background: #141414; color: #F4F5F3; text-align: center;
        pointer-events: none; animation: splash-dismiss 0.42s ease-in 1.05s forwards;
    }
    .splash-mark {
        width: 68px; height: 68px; display: grid; place-items: center;
        border: 1px solid #354339; background: #202820; border-radius: 20px;
        margin-bottom: 1.5rem; animation: splash-mark-in 0.45s ease-out both;
    }
    .splash-title { color: #F5F6F4; font-size: clamp(1.6rem, 4vw, 2.25rem); font-weight: 620; line-height: 1.2; letter-spacing: -0.035em; }
    .splash-tagline { color: #B4B4B4; margin-top: 0.85rem; font-size: 0.98rem; }
    .splash-loading { display: flex; align-items: center; gap: 0.7rem; margin-top: 2rem; color: #929892; font-size: 0.75rem; letter-spacing: 0.03em; }
    .splash-track { width: 110px; height: 3px; background: #353835; border-radius: 9px; overflow: hidden; }
    .splash-progress { display: block; width: 0; height: 100%; background: #50AD77; animation: splash-progress 0.85s ease-out 0.08s forwards; }
    @keyframes splash-progress { to { width: 100%; } }
    @keyframes splash-mark-in { from { opacity: 0; transform: translateY(6px) scale(0.96); } to { opacity: 1; transform: translateY(0) scale(1); } }
    @keyframes splash-dismiss { to { opacity: 0; visibility: hidden; } }
    @media (max-width: 760px) {
        .main .block-container { padding: 1.4rem 1rem 2.5rem; }
        .hero { padding: 1.45rem; }
        section[data-testid="stSidebar"] { min-width: 260px !important; max-width: 260px !important; }
    }
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after { animation-duration: 0.01ms !important; transition-duration: 0.01ms !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_training_data():
    path = PROCESSED_PATH if PROCESSED_PATH.exists() else RAW_TRAIN_PATH
    if not path.exists():
        raise FileNotFoundError("The training dataset is not available.")

    data = pd.read_csv(path)
    if "buggy" not in data.columns:
        raise ValueError("The training dataset does not contain its defect target.")
    if not set(FEATURE_COLUMNS).issubset(data.columns):
        raise ValueError("The training dataset is missing one or more model features.")

    data["buggy"] = pd.to_numeric(
        data["buggy"].replace(
            {"True": 1, "False": 0, True: 1, False: 0}
        ),
        errors="raise",
    ).astype(int)
    return data


@st.cache_data
def load_future_data(path_string):
    path = Path(path_string)
    return pd.read_csv(path) if path.exists() else None


@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        return None
    with METRICS_PATH.open("r", encoding="utf-8") as metrics_file:
        return json.load(metrics_file)


@st.cache_data(ttl=15, show_spinner=False)
def backend_is_available():
    try:
        response = requests.get(f"{API_URL}/health", timeout=2.5)
        if response.status_code != 200:
            return False
        health = response.json()
        return health.get("model_loaded") is True
    except (requests.RequestException, ValueError):
        return False


def api_post(path, payload):
    return requests.post(f"{API_URL}{path}", json=payload, timeout=60)


def show_api_error(response, operation):
    if response.status_code == 503:
        st.error(f"{operation} is temporarily unavailable. Please try again later.")
    elif response.status_code == 422:
        st.error("Some input values could not be processed. Check the fields and try again.")
    else:
        st.error(f"{operation} could not be completed. Please try again.")
    LOGGER.warning("%s request failed with status %s", operation, response.status_code)


def page_header(title, description):
    st.title(title)
    st.caption(description)


def navigation_button(title, active_page):
    if st.button(
        f"{NAV_ICONS[title]}   {title}",
        key=f"navigate_{title}",
        type="primary" if title == active_page else "secondary",
        use_container_width=True,
    ):
        st.session_state["active_page"] = title
        st.rerun()


def render_sidebar(api_available):
    with st.sidebar:
        st.markdown(
            """
            <div class="brand-lockup">
              <div class="brand-symbol" aria-hidden="true">
                <svg width="25" height="25" viewBox="0 0 25 25" fill="none">
                  <path d="M4 20.5h17" stroke="#68726A" stroke-width="1.4" stroke-linecap="round"/>
                  <path d="M6.5 17V12.5M12.5 17V7.5M18.5 17V10" stroke="#50AD77" stroke-width="2.6" stroke-linecap="round"/>
                  <circle cx="18.5" cy="7" r="1.4" fill="#FF8051"/>
                </svg>
              </div>
              <div>
                <div class="brand-title">Software Defect<br>Prediction</div>
                <div class="brand-caption">QUALITY ANALYTICS</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        active_page = st.session_state.get("active_page", "Overview")
        for group, pages in NAV_GROUPS.items():
            st.markdown(f'<div class="nav-group">{group}</div>', unsafe_allow_html=True)
            for title in pages:
                navigation_button(title, active_page)

        st.divider()
        if api_available:
            st.markdown(
                '<div class="service-indicator"><span class="service-dot"></span>Prediction service connected</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="service-indicator"><span class="service-dot offline"></span>Predictions temporarily unavailable</div>',
                unsafe_allow_html=True,
            )


def render_splash():
    st.markdown(
        """
        <div class="splash-screen" role="status" aria-live="polite">
          <div class="splash-mark" aria-hidden="true">
            <svg width="38" height="38" viewBox="0 0 38 38" fill="none">
              <path d="M6 31h26" stroke="#667068" stroke-width="1.5" stroke-linecap="round"/>
              <path d="M10 27V20M19 27V11M28 27V16" stroke="#50AD77" stroke-width="3" stroke-linecap="round"/>
              <path d="m8 17 9-7 9 4 5-6" stroke="#FF8051" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
              <circle cx="31" cy="8" r="2" fill="#FF8051"/>
            </svg>
          </div>
          <div class="splash-title">Software Defect Prediction System</div>
          <div class="splash-tagline">Predict defects. Understand risk. Build better software.</div>
          <div class="splash-loading">
            <span>Preparing your workspace</span>
            <span class="splash-track"><span class="splash-progress"></span></span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_class_distribution(data, show_legend=False):
    counts = data["buggy"].value_counts().reindex([0, 1], fill_value=0)
    distribution = pd.DataFrame({
        "Commit outcome": ["Non-defective", "Defective"],
        "Commits": [int(counts.loc[0]), int(counts.loc[1])],
    })
    fig = px.bar(
        distribution,
        x="Commit outcome",
        y="Commits",
        color="Commit outcome",
        color_discrete_map={
            "Non-defective": COLORS["deep_green"],
            "Defective": COLORS["coral"],
        },
        text="Commits",
    )
    fig.update_layout(
        showlegend=show_legend,
        template="plotly_white",
        margin=dict(t=10, b=10, l=10, r=10),
        xaxis_title=None,
        yaxis_title="Commits",
    )
    return fig


def render_metric_summary(metrics, compact=False):
    if not metrics:
        st.info("Saved validation metrics are not available.")
        return

    metric_names = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1-score", "f1"),
        ("ROC-AUC", "roc_auc"),
    ]
    if compact:
        metric_names = [("Accuracy", "accuracy"), ("F1-score", "f1"), ("ROC-AUC", "roc_auc")]

    columns = st.columns(len(metric_names))
    for column, (label, key) in zip(columns, metric_names):
        if key in metrics:
            column.metric(label, f"{metrics[key]:.1%}")


def render_overview(data, metrics):
    st.markdown(
        """
        <div class="hero">
          <div class="hero-eyebrow">COMMIT-LEVEL QUALITY SIGNAL</div>
          <h1>Predict software defects before they reach production.</h1>
          <p>Estimate whether a software commit is likely to contain a defect using its change and author-history metrics. Use the result to help focus review and testing—not as a guarantee.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    action_col, _ = st.columns([1, 3])
    with action_col:
        if st.button("Predict a Commit", type="primary", use_container_width=True):
            st.session_state["active_page"] = "Predict a Commit"
            st.rerun()
        if st.button("Explore Dataset", use_container_width=True):
            st.session_state["active_page"] = "Dataset Explorer"
            st.rerun()

    counts = data["buggy"].value_counts().reindex([0, 1], fill_value=0)
    defective_share = float(counts.loc[1] / len(data)) if len(data) else 0.0
    stat_columns = st.columns(3)
    stat_columns[0].metric("Training commits", f"{len(data):,}")
    stat_columns[1].metric("Defective in training data", f"{defective_share:.1%}")
    stat_columns[2].metric("Commit metrics used", str(len(FEATURE_COLUMNS)))

    st.divider()
    distribution_col, performance_col = st.columns([1.15, 1])
    with distribution_col:
        st.subheader("Training dataset")
        st.caption("ApacheJIT training records only")
        st.plotly_chart(render_class_distribution(data), use_container_width=True)
    with performance_col:
        st.subheader("Model performance")
        st.caption("Chronological validation · historical holdout")
        render_metric_summary(metrics, compact=True)
        if metrics:
            st.caption(
                f"{metrics.get('training_rows', 0):,} training rows · "
                f"{metrics.get('validation_rows', 0):,} validation rows. "
                "These are not official future-test results."
            )

    st.divider()
    st.subheader("From commit metrics to an explanation")
    st.caption("A short, transparent path from a change to a review signal.")
    steps = [
        ("01", "Describe the change", "Enter 12 documented commit metrics."),
        ("02", "Estimate defect risk", "The saved Gradient Boosting model scores the commit."),
        ("03", "Review the drivers", "A SHAP explanation highlights factors behind the score."),
    ]
    columns = st.columns(3)
    for column, (number, title, description) in zip(columns, steps):
        with column:
            st.markdown(f"**{number} · {title}**")
            st.write(description)


def feature_number_input(feature, key_prefix="single"):
    label, description = FEATURE_DETAILS[feature]
    saved_values = st.session_state.get("single_input_values", {})
    default_value = float(saved_values.get(feature, EXAMPLE_VALUES[feature]))
    if feature in INTEGER_FEATURES:
        value = st.number_input(
            label,
            min_value=0.0,
            step=1.0,
            format="%.0f",
            value=default_value,
            help=f"{description} Enter a whole number of zero or greater.",
            key=f"{key_prefix}_{feature}",
        )
    else:
        value = st.number_input(
            label,
            min_value=0.0,
            step=0.1,
            format="%.2f",
            value=default_value,
            help=f"{description} Enter a value of zero or greater.",
            key=f"{key_prefix}_{feature}",
        )
    saved_values = dict(saved_values)
    saved_values[feature] = value
    st.session_state["single_input_values"] = saved_values
    return value


def validate_feature_values(values):
    missing = [feature for feature in FEATURE_COLUMNS if feature not in values]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")

    validated = {}
    for feature in FEATURE_COLUMNS:
        try:
            value = float(values[feature])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{FEATURE_DETAILS[feature][0]} must be numeric.") from error
        if not math.isfinite(value):
            raise ValueError(f"{FEATURE_DETAILS[feature][0]} must be a finite number.")
        if value < 0:
            raise ValueError(f"{FEATURE_DETAILS[feature][0]} cannot be negative.")
        if feature in INTEGER_FEATURES and not value.is_integer():
            raise ValueError(f"{FEATURE_DETAILS[feature][0]} must be a whole number.")
        validated[feature] = value
    return validated


def render_shap_explanation(explanation):
    contributions = explanation.get("contributions", [])
    if not contributions:
        st.info("A feature explanation was not returned for this prediction.")
        return

    contributions_df = pd.DataFrame(contributions)
    if not {"feature", "shap_value"}.issubset(contributions_df.columns):
        st.info("A feature explanation was not returned in a readable format.")
        return
    contributions_df["feature_label"] = contributions_df["feature"].map(
        lambda feature: FEATURE_DETAILS.get(feature, (feature, ""))[0]
    )
    contributions_df["direction"] = contributions_df["shap_value"].map(
        lambda value: "Toward defective" if value >= 0 else "Away from defective"
    )
    fig = px.bar(
        contributions_df.sort_values("shap_value"),
        x="shap_value",
        y="feature_label",
        color="direction",
        orientation="h",
        color_discrete_map={
            "Toward defective": COLORS["coral"],
            "Away from defective": COLORS["deep_green"],
        },
        labels={"shap_value": "SHAP contribution", "feature_label": "Commit metric"},
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(t=10, b=10, l=10, r=10),
        legend_title_text="Model influence",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Positive contributions move the model output toward defective; negative "
        "contributions move it away. These describe model behavior, not causes."
    )


def render_prediction(api_available):
    page_header(
        "Predict a Commit",
        "Enter the change metrics for one commit. Each field includes a plain-language definition.",
    )
    if not api_available:
        st.warning("The prediction service is unavailable. Your inputs are ready, but a prediction cannot be run right now.")

    with st.expander("What should I enter?", expanded=False):
        st.write(
            "Use metrics calculated for a single commit. The example values below "
            "are illustrative only; they are not a prediction or a recommended risk profile."
        )
        st.code(json.dumps(EXAMPLE_VALUES, indent=2), language="json")
        st.caption(
            "Count fields accept whole numbers. Averaged metrics accept decimals. "
            "All values must be zero or greater."
        )

    with st.form("prediction_form"):
        groups = [
            ("Change size and scope", ["la", "ld", "nf", "nd", "ns"]),
            ("Change complexity", ["ent", "nuc"]),
            ("Team and file history", ["ndev", "age"]),
            ("Author history", ["aexp", "arexp", "asexp"]),
        ]
        values = {}
        for group_title, features in groups:
            st.markdown(f"**{group_title}**")
            columns = st.columns(3)
            for index, feature in enumerate(features):
                with columns[index % 3]:
                    values[feature] = feature_number_input(feature)
        submitted = st.form_submit_button(
            "Predict",
            type="primary",
            use_container_width=True,
            disabled=not api_available,
        )

    if submitted:
        st.session_state.pop("single_prediction", None)
        try:
            payload = validate_feature_values(values)
        except ValueError as error:
            st.error(str(error))
            return

        try:
            with st.spinner("Assessing this commit..."):
                response = api_post("/predict", payload)
        except requests.RequestException:
            LOGGER.exception("Single prediction request could not reach the backend.")
            st.error("The prediction service could not be reached. Please try again when it is available.")
            return
        if response.status_code != 200:
            show_api_error(response, "Prediction")
            return

        try:
            prediction = response.json()
            if not isinstance(prediction, dict) or prediction.get("prediction") not in (0, 1):
                raise ValueError("Prediction field is missing or invalid.")
            expected_label = "Defective" if prediction["prediction"] == 1 else "Non-defective"
            if prediction.get("label") != expected_label:
                raise ValueError("Prediction label is missing or inconsistent.")
            probabilities = prediction.get("probability")
            if not isinstance(probabilities, dict):
                raise ValueError("Prediction probabilities are missing.")
            defective_probability = probabilities.get("defective")
            non_defective_probability = probabilities.get("non_defective")
            threshold = prediction.get("threshold")
            if any(
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                or value > 1
                for value in (defective_probability, non_defective_probability, threshold)
            ):
                raise ValueError("Prediction probabilities or threshold are invalid.")
        except (TypeError, ValueError):
            st.error("The prediction service returned an unreadable result.")
            LOGGER.error("Prediction endpoint returned a non-JSON response.")
            return

        st.session_state["single_prediction"] = {
            "input": payload,
            "result": prediction,
            "explanation": None,
        }
        try:
            explanation_response = api_post("/explain", payload)
            if explanation_response.status_code == 200:
                explanation = explanation_response.json()
                if isinstance(explanation, dict) and isinstance(explanation.get("contributions"), list):
                    st.session_state["single_prediction"]["explanation"] = explanation
                else:
                    LOGGER.warning("SHAP endpoint returned an unexpected response.")
            else:
                LOGGER.warning("SHAP explanation request failed with status %s", explanation_response.status_code)
        except (requests.RequestException, ValueError):
            LOGGER.exception("SHAP explanation request failed.")

    saved = st.session_state.get("single_prediction")
    if not saved:
        return

    result = saved["result"]
    is_defective = result.get("prediction") == 1
    st.divider()
    st.subheader("Prediction result")
    verdict_class = "defective" if is_defective else "non-defective"
    st.markdown(
        f'<div class="prediction-verdict {verdict_class}" role="status">'
        f'<span class="verdict-mark" aria-hidden="true"></span>'
        f"<strong>{result['label']}</strong></div>",
        unsafe_allow_html=True,
    )
    if is_defective:
        st.write(
            "The model flagged this commit as more likely to be defective. "
            "Consider giving it additional review and testing."
        )
    else:
        st.write(
            "The model did not flag this commit as likely to be defective. "
            "This is not a guarantee that the change is defect-free."
        )

    probability = result["probability"]["defective"]
    if isinstance(probability, (int, float)) and math.isfinite(probability) and 0 <= probability <= 1:
        st.metric("Model-estimated defective probability", f"{probability:.1%}")
        st.caption(
            f"The model applies a {result['threshold']:.0%} decision threshold. "
            "This is a model probability, not a certainty about the commit."
        )

    st.markdown("#### What influenced this result")
    if saved.get("explanation"):
        render_shap_explanation(saved["explanation"])
    else:
        st.info("The prediction is available, but its feature explanation could not be loaded.")


def validate_batch_frame(batch_df):
    if batch_df.empty:
        raise ValueError("The uploaded CSV has no data rows.")
    duplicate_columns = batch_df.columns[batch_df.columns.duplicated()].tolist()
    if duplicate_columns:
        raise ValueError(f"Duplicate column names found: {', '.join(duplicate_columns)}")

    missing = [column for column in FEATURE_COLUMNS if column not in batch_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    try:
        numeric_df = batch_df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="raise").astype(float)
    except (ValueError, TypeError) as error:
        raise ValueError("Every required feature column must contain numeric values.") from error

    values = numeric_df.to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("The CSV contains blank or non-finite feature values.")
    if (values < 0).any():
        raise ValueError("Feature values cannot be negative.")
    for feature in INTEGER_FEATURES:
        if not np.equal(numeric_df[feature], np.floor(numeric_df[feature])).all():
            raise ValueError(f"{FEATURE_DETAILS[feature][0]} must contain whole numbers.")
    return numeric_df


def render_batch_prediction(api_available):
    page_header(
        "Batch Prediction",
        "Score multiple commits from a CSV and download the results.",
    )
    st.caption("Required input columns, in any order:")
    st.code(", ".join(FEATURE_COLUMNS), language=None)
    sample_df = pd.DataFrame([EXAMPLE_VALUES], columns=FEATURE_COLUMNS)
    st.download_button(
        "Download sample input CSV",
        data=sample_df.to_csv(index=False).encode("utf-8"),
        file_name="defect_prediction_sample.csv",
        mime="text/csv",
    )
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"], key="batch_csv")
    if uploaded is not None:
        file_bytes = uploaded.getvalue()
        stored_file = {"bytes": file_bytes, "name": uploaded.name}
        st.session_state["batch_uploaded_file"] = stored_file
    else:
        stored_file = st.session_state.get("batch_uploaded_file")
        if stored_file is None:
            st.info("Upload a CSV with one commit per row to preview and validate the input.")
            return
        file_bytes = stored_file["bytes"]

    file_name = stored_file["name"]
    file_signature = hashlib.sha256(file_bytes).hexdigest()
    try:
        batch_df = pd.read_csv(io.BytesIO(file_bytes))
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
        st.error("This file could not be read as a CSV. Check the file and upload it again.")
        return

    try:
        numeric_df = validate_batch_frame(batch_df)
    except ValueError as error:
        st.error(str(error))
        return

    st.markdown("#### Input preview")
    st.caption(f"{file_name} · {len(batch_df):,} commit(s) · extra columns are retained in the downloaded results.")
    if st.button("Clear uploaded file", key="clear_batch_file"):
        st.session_state.pop("batch_uploaded_file", None)
        st.session_state.pop("batch_prediction", None)
        st.rerun()
    st.dataframe(batch_df.head(10), use_container_width=True, hide_index=True)

    if not api_available:
        st.warning("The prediction service is unavailable, so this file cannot be scored right now.")
        return

    if st.button("Run batch prediction", type="primary", disabled=not api_available):
        st.session_state.pop("batch_prediction", None)
        try:
            with st.spinner(f"Scoring {len(numeric_df):,} commit(s)..."):
                response = api_post(
                    "/predict/batch",
                    {"records": numeric_df.to_dict(orient="records")},
                )
        except requests.RequestException:
            LOGGER.exception("Batch prediction request could not reach the backend.")
            st.error("The prediction service could not be reached. Please try again when it is available.")
            return
        if response.status_code != 200:
            show_api_error(response, "Batch prediction")
            return

        try:
            result = response.json()
            if not isinstance(result, dict) or result.get("count") != len(batch_df):
                raise ValueError("The service returned an unexpected batch size.")
            results_df = pd.DataFrame(result["results"])
            if len(results_df) != len(batch_df):
                raise ValueError("The service returned a different number of rows than it received.")
            output = batch_df.reset_index(drop=True).copy()
            output["prediction"] = results_df["prediction"].astype(int)
            output["prediction_label"] = results_df["label"]
            output["defective_probability"] = results_df["defective_probability"]
            if (
                result["defective_count"] + result["non_defective_count"] != len(batch_df)
                or len(results_df) != result["count"]
            ):
                raise ValueError("The service returned inconsistent batch counts.")
        except (KeyError, TypeError, ValueError):
            LOGGER.exception("Batch prediction endpoint returned an unexpected response.")
            st.error("The prediction service returned results in an unexpected format.")
            return

        st.session_state["batch_prediction"] = {
            "signature": file_signature,
            "result": result,
            "output": output,
        }

    saved = st.session_state.get("batch_prediction")
    if not saved or saved["signature"] != file_signature:
        return

    result = saved["result"]
    output = saved["output"]
    st.divider()
    st.subheader("Prediction results")
    count_columns = st.columns(3)
    count_columns[0].metric("Commits scored", f"{result['count']:,}")
    count_columns[1].metric("Defective", f"{result['defective_count']:,}")
    count_columns[2].metric("Non-defective", f"{result['non_defective_count']:,}")
    st.dataframe(output, use_container_width=True, hide_index=True)
    st.download_button(
        "Download prediction results",
        data=output.to_csv(index=False).encode("utf-8"),
        file_name="defect_prediction_results.csv",
        mime="text/csv",
    )


def render_model_insights(metrics):
    page_header(
        "Model Insights",
        "Explore saved Gradient Boosting evaluation results and global SHAP explanations.",
    )
    if not metrics:
        st.warning("Saved model evaluation results are not available.")
        return

    st.info(
        "Chronological historical validation results. These metrics are not measured "
        "on the official future-test datasets."
    )
    render_metric_summary(metrics)
    training_rows = metrics.get("training_rows")
    validation_rows = metrics.get("validation_rows")
    training_label = f"{training_rows:,}" if isinstance(training_rows, int) else "—"
    validation_label = f"{validation_rows:,}" if isinstance(validation_rows, int) else "—"
    st.caption(
        f"Training rows: {training_label} · Validation rows: {validation_label} · "
        f"Decision threshold: {metrics.get('threshold', 0.5):.2f}"
    )

    st.divider()
    matrix = metrics.get("confusion_matrix")
    if matrix and len(matrix) == 2 and all(len(row) == 2 for row in matrix):
        confusion_df = pd.DataFrame(
            matrix,
            index=["Actual non-defective", "Actual defective"],
            columns=["Predicted non-defective", "Predicted defective"],
        )
        st.subheader("Confusion matrix")
        st.dataframe(confusion_df, use_container_width=True)
    if CONFUSION_MATRIX_PATH.exists():
        st.image(str(CONFUSION_MATRIX_PATH), caption="Chronological validation confusion matrix")

    curve_columns = st.columns(2)
    with curve_columns[0]:
        if ROC_CURVE_PATH.exists():
            st.image(str(ROC_CURVE_PATH), caption="ROC curve · chronological validation")
        else:
            st.info("A saved ROC curve is not available.")
    with curve_columns[1]:
        if PRECISION_RECALL_PATH.exists():
            st.image(str(PRECISION_RECALL_PATH), caption="Precision-recall curve · chronological validation")
        else:
            st.info("A saved precision-recall curve is not available.")

    st.divider()
    st.subheader("Global feature influence")
    st.caption(
        "Mean absolute SHAP values from the saved evaluation explain which features "
        "most influenced model outputs overall; they do not establish causation."
    )
    if SHAP_PATH.exists():
        try:
            shap_df = pd.read_csv(SHAP_PATH)
            shap_df["feature_label"] = shap_df["feature"].map(
                lambda feature: FEATURE_DETAILS.get(feature, (feature, ""))[0]
            )
            importance_fig = px.bar(
                shap_df.sort_values("mean_absolute_shap"),
                x="mean_absolute_shap",
                y="feature_label",
                orientation="h",
                color_discrete_sequence=[COLORS["deep_green"]],
                labels={
                    "mean_absolute_shap": "Mean absolute SHAP value",
                    "feature_label": "Commit metric",
                },
            )
            importance_fig.update_layout(
                template="plotly_white",
                margin=dict(t=10, b=10, l=10, r=10),
            )
            st.plotly_chart(importance_fig, use_container_width=True)
        except (OSError, pd.errors.ParserError, KeyError, ValueError):
            LOGGER.exception("Saved SHAP feature importance could not be loaded.")
            st.info("Saved SHAP feature importance could not be displayed.")
    else:
        st.info("Saved SHAP feature importance is not available.")

    if SHAP_SUMMARY_PATH.exists():
        st.image(
            str(SHAP_SUMMARY_PATH),
            caption="SHAP summary plot · historical development data",
        )
    else:
        st.info("A saved SHAP summary plot is not available.")


def render_training_explorer(data):
    st.markdown("#### Training dataset")
    st.caption(
        f"{len(data):,} commits · {len(data.columns)} columns · "
        "used for model development and historical validation."
    )
    counts = data["buggy"].value_counts().reindex([0, 1], fill_value=0)
    status_columns = st.columns(2)
    status_columns[0].metric("Non-defective", f"{int(counts.loc[0]):,}")
    status_columns[1].metric("Defective", f"{int(counts.loc[1]):,}")
    st.plotly_chart(render_class_distribution(data), use_container_width=True)

    filter_columns = st.columns([1, 1, 1.3])
    projects = ["All projects"] + sorted(data["project"].dropna().astype(str).unique().tolist()) if "project" in data.columns else ["All projects"]
    selected_project = filter_columns[0].selectbox("Project", projects)
    statuses = filter_columns[1].multiselect(
        "Defect status",
        ["Non-defective", "Defective"],
        default=["Non-defective", "Defective"],
    )
    query = filter_columns[2].text_input(
        "Search commit or project",
        placeholder="Search commit ID or project name",
    )

    filtered = data.copy()
    if selected_project != "All projects" and "project" in filtered.columns:
        filtered = filtered[filtered["project"].astype(str) == selected_project]
    selected_targets = [
        value for label, value in [("Non-defective", 0), ("Defective", 1)]
        if label in statuses
    ]
    filtered = filtered[filtered["buggy"].isin(selected_targets)]
    if query:
        searchable_columns = [column for column in ["commit_id", "project"] if column in filtered.columns]
        if searchable_columns:
            matches = pd.Series(False, index=filtered.index)
            for column in searchable_columns:
                matches |= filtered[column].astype("string").str.contains(query, case=False, na=False, regex=False)
            filtered = filtered[matches]

    selected_feature = st.selectbox(
        "Feature to visualize and filter",
        FEATURE_COLUMNS,
        format_func=lambda feature: FEATURE_DETAILS[feature][0],
    )
    feature_min = float(data[selected_feature].min())
    feature_max = float(data[selected_feature].max())
    if feature_min < feature_max:
        value_range = st.slider(
            "Feature value range",
            min_value=feature_min,
            max_value=feature_max,
            value=(feature_min, feature_max),
            key=f"explorer_feature_range_{selected_feature}",
        )
        filtered = filtered[
            filtered[selected_feature].between(value_range[0], value_range[1])
        ]
    if filtered.empty:
        st.info("No training records match these filters.")
    else:
        st.caption(f"{len(filtered):,} matching record(s)")
        st.dataframe(filtered, use_container_width=True, hide_index=True, height=420)

    plot_data = filtered.copy()
    plot_data["Defect status"] = plot_data["buggy"].map(
        {0: "Non-defective", 1: "Defective"}
    )
    if not plot_data.empty:
        distribution_fig = px.histogram(
            plot_data,
            x=selected_feature,
            color="Defect status",
            barmode="overlay",
            opacity=0.7,
            color_discrete_map={
                "Non-defective": COLORS["deep_green"],
                "Defective": COLORS["coral"],
            },
            labels={selected_feature: FEATURE_DETAILS[selected_feature][0]},
        )
        distribution_fig.update_layout(
            template="plotly_white",
            margin=dict(t=10, b=10, l=10, r=10),
        )
        st.plotly_chart(distribution_fig, use_container_width=True)

    with st.expander("Feature descriptions"):
        descriptions = pd.DataFrame(
            [
                {"Metric": feature, "Name": FEATURE_DETAILS[feature][0], "Meaning": FEATURE_DETAILS[feature][1]}
                for feature in FEATURE_COLUMNS
            ]
        )
        st.dataframe(descriptions, use_container_width=True, hide_index=True)


def render_future_inventory(path, title, description):
    future_data = load_future_data(str(path))
    st.markdown(f"#### {title}")
    st.caption(description)
    if future_data is None:
        st.info("This future-test file is not available in the current project data.")
        return
    st.metric("Reserved records", f"{len(future_data):,}")
    st.dataframe(future_data.head(10), use_container_width=True, hide_index=True)
    st.caption(
        "This official future-test dataset is kept separate from training and "
        "chronological validation; it is not used to calculate the scores shown here."
    )


def render_dataset_explorer(data):
    page_header(
        "Dataset Explorer",
        "Search and inspect the training data while keeping official future-test data separate.",
    )
    st.info(
        "Training and official future-test datasets are shown in separate views. "
        "Future-test files remain reserved for a future evaluation."
    )
    training_tab, large_tab, small_tab = st.tabs(
        ["Training dataset", "Official future test · large", "Official future test · small"]
    )
    with training_tab:
        render_training_explorer(data)
    with large_tab:
        render_future_inventory(
            FUTURE_LARGE_PATH,
            "Official future-test dataset · large",
            "Reserved evaluation data. It is not combined with or used to filter the training dataset.",
        )
    with small_tab:
        render_future_inventory(
            FUTURE_SMALL_PATH,
            "Official future-test dataset · small",
            "Reserved evaluation data. It is not combined with or used to filter the training dataset.",
        )


def render_how_it_works():
    page_header(
        "How It Works",
        "A plain-language guide to what the system predicts, how it does it, and what the result can—and cannot—tell you.",
    )
    pipeline = [
        ("User input", "Commit change metrics"),
        ("Streamlit Interface", "Input and result screens"),
        ("FastAPI", "Sends metrics to the model"),
        ("Gradient Boosting Model", "Estimates defect likelihood"),
        ("Prediction", "Defective or non-defective"),
        ("SHAP Explanation", "Highlights model influences"),
        ("Results", "A review signal for your team"),
    ]
    columns = st.columns(len(pipeline))
    for column, (title, description) in zip(columns, pipeline):
        with column:
            st.markdown(f"**{title}**")
            st.caption(description)
    st.divider()

    st.subheader("What does commit-level prediction mean?")
    st.write(
        "The system looks at one software commit at a time and estimates whether "
        "it is likely to be associated with defects. It is intended to help teams "
        "prioritize code review and testing—not to replace either."
    )

    st.subheader("What are the 12 input metrics?")
    descriptions = pd.DataFrame(
        [
            {
                "Metric": feature,
                "Name": FEATURE_DETAILS[feature][0],
                "What it describes": FEATURE_DETAILS[feature][1],
            }
            for feature in FEATURE_COLUMNS
        ]
    )
    st.dataframe(descriptions, use_container_width=True, hide_index=True)

    st.subheader("How is a prediction made?")
    st.write(
        "The application sends the 12 numeric metrics to the existing Gradient "
        "Boosting model. The model compares the combination of values with patterns "
        "learned from historical ApacheJIT training data, then returns a label and "
        "a defective-class probability. A saved threshold determines the label."
    )

    st.subheader("What does the SHAP explanation mean?")
    st.write(
        "SHAP assigns each input feature a contribution to the model output for "
        "this commit. A positive contribution moves the model output toward "
        "defective; a negative contribution moves it away. It describes how the "
        "model behaved, not why a defect occurred."
    )

    st.subheader("Important limitations")
    st.markdown(
        "- Historical ApacheJIT data may not represent a different codebase, team, or time period.\n"
        "- A predicted label is a signal, not certainty; a non-defective prediction does not guarantee safe code.\n"
        "- Probabilities are model outputs and should not be treated as calibrated confidence.\n"
        "- Validation scores shown in Model Insights use a chronological historical holdout, not the official future-test datasets.\n"
        "- SHAP contributions are explanatory, not causal."
    )


try:
    training_data = load_training_data()
except (FileNotFoundError, ValueError, pd.errors.ParserError):
    LOGGER.exception("The training dataset could not be loaded.")
    st.error("The training dataset could not be loaded. Check the project data files and try again.")
    st.stop()

api_available = backend_is_available()
render_sidebar(api_available)
metrics = None
try:
    metrics = load_metrics()
except (OSError, json.JSONDecodeError):
    LOGGER.exception("Saved evaluation metrics could not be loaded.")

active_page = st.session_state.get("active_page", "Overview")
if active_page == "Overview":
    render_overview(training_data, metrics)
elif active_page == "Predict a Commit":
    render_prediction(api_available)
elif active_page == "Batch Prediction":
    render_batch_prediction(api_available)
elif active_page == "Model Insights":
    render_model_insights(metrics)
elif active_page == "Dataset Explorer":
    render_dataset_explorer(training_data)
elif active_page == "How It Works":
    render_how_it_works()

if show_splash:
    render_splash()
