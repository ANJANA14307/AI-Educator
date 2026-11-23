
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "apachejit_processed.csv"
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "apachejit_train.csv"

FEATURE_COLUMNS = [
    "la", "ld", "nf", "nd", "ns", "ent",
    "ndev", "age", "nuc", "aexp", "arexp", "asexp"
]

st.set_page_config(
    page_title="DefectGuard | Software Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# 2. HYBRID UI STYLING
# --------------------------------------------------

st.markdown("""
<style>
    .stApp {
        background-color: #F4F6FA;
    }

    section[data-testid="stSidebar"] {
        background-color: #111827;
    }

    section[data-testid="stSidebar"] * {
        color: #E5E7EB;
    }

    section[data-testid="stSidebar"] [data-testid="stRadio"] label {
        padding: 9px 12px;
        border-radius: 8px;
    }

    .main-title {
        color: #111827;
        font-size: 32px;
        font-weight: 750;
        margin-bottom: 2px;
    }

    .subtitle {
        color: #64748B;
        font-size: 14px;
        margin-bottom: 25px;
    }

    .section-title {
        color: #111827;
        font-size: 21px;
        font-weight: 700;
        margin-top: 15px;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #E2E8F0;
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    }

    div[data-testid="stMetricLabel"] {
        color: #64748B;
    }

    div[data-testid="stMetricValue"] {
        color: #111827;
    }

    div[data-testid="stForm"] {
        background: white;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
    }

    .status-tag {
        color: #15803D;
        background: #DCFCE7;
        padding: 5px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }

    hr {
        border-color: #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 3. DATA LOADING
# --------------------------------------------------

@st.cache_data
def load_data():
    if PROCESSED_PATH.exists():
        df = pd.read_csv(PROCESSED_PATH)
        source = "Processed"

    elif RAW_PATH.exists():
        df = pd.read_csv(RAW_PATH)
        source = "Raw"

    else:
        raise FileNotFoundError(
            "ApacheJIT dataset not found. "
            "Place the dataset inside data/raw/ or run preprocessing."
        )

    if "buggy" not in df.columns:
        raise ValueError("Required target column 'buggy' is missing.")

    # Normalize target labels to 0 and 1.
    if df["buggy"].dtype == bool:
        df["buggy"] = df["buggy"].astype(int)
    else:
        df["buggy"] = (
            df["buggy"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({
                "true": 1,
                "false": 0,
                "1": 1,
                "0": 0
            })
        )

    if df["buggy"].isna().any():
        raise ValueError(
            "The target column contains missing or unsupported labels."
        )

    return df, source


try:
    df, source = load_data()

except Exception as exc:
    st.error(f"Dataset loading error: {exc}")
    st.stop()


available_features = [
    col for col in FEATURE_COLUMNS if col in df.columns
]

missing_features = [
    col for col in FEATURE_COLUMNS if col not in df.columns
]

target_counts = df["buggy"].value_counts().reindex(
    [0, 1], fill_value=0
)

defective_count = int(target_counts[1])
non_defective_count = int(target_counts[0])

defective_percentage = (
    defective_count / len(df) * 100 if len(df) else 0
)


# --------------------------------------------------
# 4. SIDEBAR NAVIGATION
# --------------------------------------------------

with st.sidebar:

    st.markdown("# 🛡️ DefectGuard")
    st.caption("SOFTWARE QUALITY INTELLIGENCE")

    st.divider()

    st.markdown("### WORKSPACE")

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Defect Prediction",
            "Dataset Explorer",
            "Model Insights"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### DATA SOURCE")

    st.markdown(
        f"**ApacheJIT**  \n"
        f"44,834 training records"
    )

    st.caption(f"Loaded source: {source}")

    st.divider()

    st.markdown("### SYSTEM STATUS")

    st.markdown(
        '<span class="status-tag">● DATASET AVAILABLE</span>',
        unsafe_allow_html=True
    )

    st.caption("ML prediction API: Not connected yet")


# --------------------------------------------------
# 5. REUSABLE PAGE HEADER
# --------------------------------------------------

def page_header(title, description):
    st.markdown(
        f'<div class="main-title">{title}</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="subtitle">{description}</div>',
        unsafe_allow_html=True
    )


# --------------------------------------------------
# 6. OVERVIEW PAGE
# --------------------------------------------------

if page == "Overview":

    page_header(
        "Overview",
        "Monitor software quality, commit statistics and defect patterns."
    )

    st.markdown("### Dataset Summary")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Commits", f"{len(df):,}")
    col2.metric("Defective Commits", f"{defective_count:,}")
    col3.metric("Change Features", len(available_features))
    col4.metric("Dataset Columns", len(df.columns))

    st.divider()

    left, right = st.columns([1.4, 1])

    with left:

        st.markdown("### Defect Distribution")

        distribution = pd.DataFrame({
            "Commit Type": [
                "Non-defective",
                "Defective"
            ],
            "Count": [
                non_defective_count,
                defective_count
            ]
        })

        fig = px.pie(
            distribution,
            names="Commit Type",
            values="Count",
            hole=0.65,
            color="Commit Type",
            color_discrete_map={
                "Non-defective": "#64748B",
                "Defective": "#16A34A"
            }
        )

        fig.update_layout(
            margin=dict(t=20, b=20, l=10, r=10),
            paper_bgcolor="white",
            plot_bgcolor="white"
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:

        st.markdown("### Dataset Health")

        st.metric(
            "Missing Values",
            f"{int(df.isna().sum().sum()):,}"
        )

        st.metric(
            "Duplicate Rows",
            f"{int(df.duplicated().sum()):,}"
        )

        st.metric(
            "Defective Commit Ratio",
            f"{defective_percentage:.2f}%"
        )

        if missing_features:
            st.warning(
                f"Some expected features are missing: {missing_features}"
            )
        else:
            st.success("All 12 expected change features are available.")

    st.divider()

    st.markdown("### Platform Modules")

    col1, col2, col3 = st.columns(3)

    col1.info(
        "**Dataset Explorer**\n\n"
        "Explore distributions, correlations and data quality."
    )

    col2.info(
        "**Defect Prediction**\n\n"
        "Prepare commit metrics for machine learning predictions."
    )

    col3.info(
        "**Model Insights**\n\n"
        "Review model evaluation and explainability."
    )

    st.caption(
        "Prediction and model evaluation modules become active "
        "after backend integration."
    )


# --------------------------------------------------
# 7. DEFECT PREDICTION PAGE
# --------------------------------------------------

elif page == "Defect Prediction":

    page_header(
        "Defect Prediction",
        "Submit software commit metrics for defect risk analysis."
    )

    st.info(
        "The prediction interface is being prepared. "
        "The trained ML model and FastAPI endpoint have not been "
        "connected yet. No predictions are generated at this stage."
    )

    if missing_features:

        st.error(
            f"Cannot prepare the input form. Missing features: "
            f"{missing_features}"
        )

    else:

        st.markdown("### Commit Metrics")

        st.write(
            "Enter the 12 change metrics used by the ApacheJIT model."
        )

        with st.form("prediction_form"):

            col1, col2, col3 = st.columns(3)

            values = {}

            with col1:
                values["la"] = st.number_input(
                    "Lines Added (la)", min_value=0, value=10
                )
                values["ld"] = st.number_input(
                    "Lines Deleted (ld)", min_value=0, value=5
                )
                values["nf"] = st.number_input(
                    "Files Changed (nf)", min_value=0, value=2
                )
                values["nd"] = st.number_input(
                    "Directories Changed (nd)", min_value=0, value=1
                )

            with col2:
                values["ns"] = st.number_input(
                    "Subsystems Changed (ns)", min_value=0, value=1
                )
                values["ent"] = st.number_input(
                    "Change Entropy (ent)", min_value=0.0, value=0.5
                )
                values["ndev"] = st.number_input(
                    "Number of Developers (ndev)",
                    min_value=0.0,
                    value=1.0
                )
                values["age"] = st.number_input(
                    "Commit Age (age)", min_value=0.0, value=10.0
                )

            with col3:
                values["nuc"] = st.number_input(
                    "Unique Changes (nuc)", min_value=0.0, value=1.0
                )
                values["aexp"] = st.number_input(
                    "Author Experience (aexp)", min_value=0, value=10
                )
                values["arexp"] = st.number_input(
                    "Recent Author Experience (arexp)",
                    min_value=0.0,
                    value=5.0
                )
                values["asexp"] = st.number_input(
                    "Author Subsystem Experience (asexp)",
                    min_value=0.0,
                    value=5.0
                )

            submitted = st.form_submit_button(
                "Prepare Prediction",
                use_container_width=True
            )

        if submitted:
            st.session_state["pending_prediction_input"] = values

            st.success(
                "Input captured successfully. "
                "Connect the trained model API to generate a real prediction."
            )

            st.json(values)


# --------------------------------------------------
# 8. DATASET EXPLORER PAGE
# --------------------------------------------------

elif page == "Dataset Explorer":

    page_header(
        "Dataset Explorer",
        "Explore ApacheJIT commit-level data and investigate patterns."
    )

    st.markdown("### Dataset Information")

    col1, col2, col3 = st.columns(3)

    col1.metric("Records", f"{len(df):,}")
    col2.metric("Features", len(available_features))
    col3.metric("Missing Values", int(df.isna().sum().sum()))

    with st.expander("View dataset preview"):

        st.dataframe(
            df.head(20),
            use_container_width=True
        )

    st.divider()

    st.markdown("### Target Distribution")

    distribution = pd.DataFrame({
        "Label": [
            "Non-defective",
            "Defective"
        ],
        "Count": [
            non_defective_count,
            defective_count
        ]
    })

    fig = px.bar(
        distribution,
        x="Label",
        y="Count",
        color="Label",
        text="Count",
        color_discrete_map={
            "Non-defective": "#64748B",
            "Defective": "#16A34A"
        }
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.markdown("### Missing Value Analysis")

    missing = (
        df.isna().sum()
        .rename_axis("Column")
        .reset_index(name="Missing Count")
    )

    missing = missing[missing["Missing Count"] > 0]

    if missing.empty:
        st.success("No missing values found.")
    else:
        st.dataframe(missing, use_container_width=True)

        fig = px.bar(
            missing,
            x="Column",
            y="Missing Count"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.markdown("### Feature Distribution")

    if available_features:

        selected_feature = st.selectbox(
            "Select a feature",
            available_features
        )

        chart_type = st.radio(
            "Visualization",
            ["Histogram", "Box Plot"],
            horizontal=True
        )

        plot_df = df.copy()
        plot_df["Defect Status"] = plot_df["buggy"].map({
            0: "Non-defective",
            1: "Defective"
        })

        if chart_type == "Histogram":

            fig = px.histogram(
                plot_df,
                x=selected_feature,
                color="Defect Status",
                barmode="overlay",
                opacity=0.65,
                title=f"Distribution of {selected_feature}"
            )

        else:

            fig = px.box(
                plot_df,
                x="Defect Status",
                y=selected_feature,
                color="Defect Status",
                title=f"{selected_feature} by defect status"
            )

        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.markdown("### Feature Correlation")

    correlation = df[available_features + ["buggy"]].corr(
        numeric_only=True
    )

    if not correlation.empty:

        fig = px.imshow(
            correlation,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1,
            title="Feature Correlation Heatmap"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.markdown("### Dataset Limitations")

    st.markdown("""
    - ApacheJIT contains historical commit-level observations.
    - Results may not generalize to every software repository.
    - Commit identifiers and project/time metadata should not be used
      as predictive features.
    - Correlation does not establish causation.
    - Chronological evaluation is important to avoid data leakage.
    """)


# --------------------------------------------------
# 9. MODEL INSIGHTS PAGE
# --------------------------------------------------

elif page == "Model Insights":

    page_header(
        "Model Insights",
        "Understand the performance and behaviour of the trained model."
    )

    st.info(
        "Model evaluation results will appear here after Person 2 "
        "integrates the training and evaluation pipeline."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Precision", "Pending")
    col2.metric("Recall", "Pending")
    col3.metric("F1 Score", "Pending")
    col4.metric("ROC-AUC", "Pending")

    st.divider()

    st.markdown("### Planned Model Analysis")

    st.markdown("""
    - Confusion matrix
    - Precision and recall comparison
    - F1-score and ROC-AUC
    - Random Forest feature importance
    - SHAP-based explainability
    """)

    st.warning(
        "No model metrics are displayed yet because the trained model "
        "has not been connected. We will populate this page with "
        "actual evaluation results, not sample values."
    )


# --------------------------------------------------
# 10. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "DefectGuard | Software Defect Prediction System | "
    "ApacheJIT Dataset"
)
