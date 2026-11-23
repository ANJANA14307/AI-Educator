from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "apachejit_processed.csv"
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "apachejit_train.csv"

st.set_page_config(page_title="ApacheJIT Defect Prediction", layout="wide")
st.title("Software Defect Prediction System")
st.write(
    "Dashboard for exploratory analysis of the ApacheJIT Just-In-Time "
    "software defect prediction dataset."
)


@st.cache_data
def load_data():
    if PROCESSED_PATH.exists():
        return pd.read_csv(PROCESSED_PATH), "processed"
    if RAW_PATH.exists():
        df = pd.read_csv(RAW_PATH)
        if "buggy" in df.columns:
            df["buggy"] = df["buggy"].map(
                {True: 1, False: 0, "True": 1, "False": 0}
            )
        return df, "raw"
    raise FileNotFoundError(
        "No ApacheJIT dataset is available. "
        "Place apachejit_train.csv in data/raw/ or run preprocessing first."
    )


try:
    df, source = load_data()
except Exception as exc:
    st.error(str(exc))
    st.stop()

target = "buggy"
if target not in df.columns:
    st.error("The dataset is missing the required 'buggy' target column.")
    st.stop()

feature_columns = [
    column for column in [
        "la", "ld", "nf", "nd", "ns", "ent", "ndev", "age",
        "nuc", "aexp", "arexp", "asexp",
    ]
    if column in df.columns
]

st.caption(f"Data source: {source} dataset")
metric1, metric2, metric3 = st.columns(3)
metric1.metric("Total records", f"{len(df):,}")
metric2.metric("Available change features", len(feature_columns))
metric3.metric("Columns", len(df.columns))

st.subheader("Target distribution")
target_counts = df[target].value_counts(dropna=False).rename_axis("Label").reset_index(name="Count")
target_counts["Label"] = target_counts["Label"].map(
    {0: "Non-defective (0)", 1: "Defective (1)", False: "Non-defective", True: "Defective"}
).fillna(target_counts["Label"].astype(str))
st.plotly_chart(
    px.bar(target_counts, x="Label", y="Count", text="Count", title="Defective versus non-defective"),
    use_container_width=True,
)

st.subheader("Missing-value summary")
missing = df.isna().sum()
missing = (
    missing[missing > 0]
    .rename_axis("Column")
    .reset_index(name="Missing count")
)
if missing.empty:
    st.success("No missing values are present in the loaded dataset.")
else:
    st.dataframe(missing, use_container_width=True)
    st.plotly_chart(
        px.bar(missing, x="Column", y="Missing count", title="Missing Values by Column"),
        use_container_width=True,
    )

st.subheader("Interactive feature visualization")
if feature_columns:
    selected_feature = st.selectbox("Choose a numerical feature", feature_columns)
    chart_type = st.radio("Chart type", ["Histogram", "Box plot"], horizontal=True)
    if chart_type == "Histogram":
        fig = px.histogram(
            df,
            x=selected_feature,
            color=df[target].astype(str),
            barmode="overlay",
            opacity=0.65,
            title=f"Distribution of {selected_feature}",
        )
    else:
        fig = px.box(
            df,
            x=df[target].astype(str),
            y=selected_feature,
            title=f"{selected_feature} by defect label",
        )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No known ApacheJIT change metrics are available in the loaded dataset.")

st.subheader("Correlation heatmap")
correlation_columns = feature_columns + ([target] if target in df.columns else [])
correlation = df[correlation_columns].corr(numeric_only=True)
if not correlation.empty:
    fig = px.imshow(
        correlation,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        title="Correlation between available features",
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Key EDA findings")
st.markdown(
    "- Findings are derived only from the loaded data.\n"
    "- Review the actual target counts and missing-value table above.\n"
    "- Feature distributions and correlations are available for interactive investigation.\n"
    "- No outliers are automatically removed; preprocessing preserves observed values."
)

st.subheader("Dataset limitations")
st.markdown(
    "- ApacheJIT is historical and project-specific; performance may not generalize to other repositories.\n"
    "- Commit identifiers and project/time metadata should not be used as standalone predictive features.\n"
    "- Correlation does not establish causation.\n"
    "- Use the official chronological train/test files and a future-only test set for defensible evaluation."
)