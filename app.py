"""Streamlit interface for evaluating the saved occupancy classifiers."""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score


# Resolve paths relative to this file so the app works no matter where it is launched from.
ROOT = Path(__file__).resolve().parent
# Folder written by training.ipynb: fitted .joblib models.
MODEL_DIR = ROOT / "model"
# Model inputs and labels used by the UCI Room Occupancy Estimation dataset.
FEATURE_COLUMNS = [
    "S1_Temp", "S2_Temp", "S3_Temp", "S4_Temp",
    "S1_Light", "S2_Light", "S3_Light", "S4_Light",
    "S1_Sound", "S2_Sound", "S3_Sound", "S4_Sound",
    "S5_CO2", "S5_CO2_Slope", "S6_PIR", "S7_PIR",
]
TARGET_COLUMN = "Room_Occupancy_Count"
CLASS_LABELS = [0, 1, 2, 3]
MODEL_NAMES = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "knn": "K-Nearest Neighbors",
    "gaussian_nb": "Gaussian Naive Bayes",
    "support_vector_machine": "Support Vector Machine",
}
# Metric names, in the order they appear in the comparison table and metric cards.
METRIC_COLUMNS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

# Must be the first Streamlit call; sets the browser tab title, icon and page width.
st.set_page_config(
    page_title="Room Occupancy | Model Evaluation",
    page_icon="◉",
    layout="wide",
)
# Custom CSS for the hero banner and metric cards.
st.markdown(
    """
    <style>
    .block-container { max-width: 1180px; padding-top: 1.5rem; padding-bottom: 3rem; }
    .hero { background: linear-gradient(120deg, #0b4f6c 0%, #1f8a87 100%); color: #ffffff;
      border-radius: 12px; padding: 1.5rem 1.75rem; margin-bottom: 1.25rem; }
    .hero h1 { color: #ffffff; font-size: 2rem; margin: 0 0 0.35rem 0; }
    .hero p { margin: 0; opacity: 0.92; font-size: 1rem; }
    [data-testid="stMetric"] { background: rgba(128, 128, 128, 0.08); border: 1px solid rgba(128, 128, 128, 0.3);
      border-radius: 10px; padding: 0.8rem 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# Cache the models so they are loaded from disk once, not on every widget interaction.
@st.cache_resource
def load_estimators() -> dict:
    return {
        key: joblib.load(MODEL_DIR / f"{key}.joblib")
        for key in MODEL_NAMES
    }


# Score one fitted model on the evaluation data and return its predictions and six metrics.
def evaluate(estimator, features: pd.DataFrame, target: pd.Series, labels: list[int]) -> dict:
    # Hard class predictions are used for accuracy, precision, recall, F1 and MCC.
    predictions = estimator.predict(features)
    # Per-class probabilities are needed for the AUC calculation.
    probabilities = estimator.predict_proba(features)
    return {
        "predictions": predictions,
        "metrics": {
            "Accuracy": accuracy_score(target, predictions),
            "AUC": roc_auc_score(
                target,
                probabilities,
                labels=labels,
                multi_class="ovr",
                average="macro",
            ),
            "Precision": precision_score(target, predictions, average="macro", zero_division=0),
            "Recall": recall_score(target, predictions, average="macro", zero_division=0),
            "F1": f1_score(target, predictions, average="macro", zero_division=0),
            "MCC": matthews_corrcoef(target, predictions),
        },
    }


# Page header.
st.markdown(
    """
    <div class="hero">
      <h1>Room Occupancy Estimation</h1>
      <p>Compare five classifiers on UCI environmental sensor readings and estimate how many
      people are in a room (0 to 3).</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Stop early with a clear message if training.ipynb has not been run yet.
missing_models = [
    key for key in MODEL_NAMES
    if not (MODEL_DIR / f"{key}.joblib").is_file()
]
if missing_models:
    st.error("Saved models are missing. Run all cells in `training.ipynb` before starting the app.")
    st.stop()

# Load the saved models.
estimators = load_estimators()

# Left-hand navigation: Overview, the model comparison, or one model at a time.
VIEW_LABELS = {"overview": "Overview", "comparison": "Model comparison"}
with st.sidebar:
    st.markdown("### Navigate")
    view = st.radio(
        "Go to",
        options=["overview", "comparison", *MODEL_NAMES],
        format_func=lambda key: VIEW_LABELS.get(key) or MODEL_NAMES[key],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("### Evaluate your data")
    uploaded_file = st.file_uploader("Upload a labeled CSV (optional)", type=["csv"])

# Overview: built from the bundled held-out test data.
try:
    bundled_test = pd.read_csv(ROOT / "test_data.csv")
except (OSError, pd.errors.ParserError) as error:
    st.error(f"Could not read the bundled test data: {error}")
    st.stop()

if view == "overview":
    st.subheader("About the dataset")
    st.write(
        "The **Room Occupancy Estimation** dataset from the UCI Machine Learning Repository. "
        "Temperature, light, sound, CO2 and PIR motion sensors record a reading about every 30 seconds, "
        "and the target is the number of occupants (0 to 3)."
    )
    overview_columns = st.columns(4)
    overview_columns[0].metric("Total observations", "10,129")
    overview_columns[1].metric("Sensor features", len(FEATURE_COLUMNS))
    overview_columns[2].metric("Occupancy classes (target)", len(CLASS_LABELS))
    overview_columns[3].metric("Held-out test rows", f"{len(bundled_test):,}")
    st.caption(
        "Held-out test day: 22 Dec 2017. All other days were used for training. "
        "The 16 sensor columns are the model inputs."
    )

    download_column, link_column, _ = st.columns([1, 1, 2])
    with download_column:
        st.download_button(
            label="Download test data (CSV)",
            data=(ROOT / "test_data.csv").read_bytes(),
            file_name="test_data.csv",
            mime="text/csv",
            icon=":material/download:",
            on_click="ignore",
            width="stretch",
        )
    with link_column:
        st.link_button(
            "View dataset on UCI",
            "https://archive.ics.uci.edu/dataset/864/room+occupancy+estimation",
            width="stretch",
        )
    st.info(
        "Download the test CSV above and upload it in the sidebar to evaluate the models on your own copy. "
        "Your file must include the 16 sensor columns and `Room_Occupancy_Count`."
    )
    st.markdown("**Occupancy class distribution in the test data**")
    distribution_column, _ = st.columns([1, 2])
    with distribution_column:
        st.dataframe(
            bundled_test[TARGET_COLUMN].value_counts().sort_index().rename("Rows").rename_axis("Occupancy").to_frame(),
            width="stretch",
        )

# Without an upload, the bundled held-out test data is used.
if uploaded_file is None:
    evaluation_data = bundled_test
    data_origin = "held-out test data"
else:
    try:
        evaluation_data = pd.read_csv(uploaded_file)
    except (OSError, pd.errors.ParserError) as error:
        st.error(f"Could not read the CSV: {error}")
        st.stop()
    data_origin = uploaded_file.name

# Validate the data before scoring. Each check shows an error and stops the run if it fails.
# 1) All 16 sensor columns and the target column must be present.
required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
missing_columns = [column for column in required_columns if column not in evaluation_data.columns]
if missing_columns:
    st.error(f"Missing required columns: {', '.join(missing_columns)}")
    st.stop()
# 2) The file must contain at least one row.
if evaluation_data.empty:
    st.error("The uploaded CSV contains no rows.")
    st.stop()

# 3) Convert values to numbers. Anything non-numeric becomes NaN and is caught by the next check.
features = evaluation_data[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
target = pd.to_numeric(evaluation_data[TARGET_COLUMN], errors="coerce")
if features.isna().any().any() or target.isna().any():
    st.error("Sensor features and target values must be numeric and contain no missing values.")
    st.stop()
# 4) Class labels must be whole numbers (for example 2.0 is accepted, 2.5 is not).
if not target.mod(1).eq(0).all():
    st.error("Occupancy target values must be whole numbers: 0, 1, 2, or 3.")
    st.stop()
# 5) Every label must be a class the models were trained on.
unknown_labels = sorted(set(target.astype(int)) - set(CLASS_LABELS))
if unknown_labels:
    st.error(f"Target values not present in the trained models: {unknown_labels}")
    st.stop()
target = target.astype(int)
# 6) AUC needs at least one example of every class, so all four classes must be present.
if set(target) != set(CLASS_LABELS):
    st.error("The labeled test data must contain all four occupancy classes to calculate multiclass AUC.")
    st.stop()

# Score every saved model on the same validated data.
results = {
    key: evaluate(estimator, features, target, CLASS_LABELS)
    for key, estimator in estimators.items()
}

# Shows at the top of the comparison and model views which data the scores come from.
def show_evaluation_source() -> None:
    if uploaded_file is None:
        st.info(
            f"**Evaluated on the held-out test data** · {len(evaluation_data):,} rows from the "
            "held-out test day. The scores below are for the bundled test set."
        )
    else:
        st.success(
            f"**Evaluated on your uploaded file** · `{uploaded_file.name}` · {len(evaluation_data):,} rows. "
            "The scores below are calculated on this file."
        )


if view == "comparison":
    comparison = pd.DataFrame(
        {MODEL_NAMES[key]: result["metrics"] for key, result in results.items()}
    ).T[METRIC_COLUMNS]

    st.subheader("Model comparison")
    show_evaluation_source()
    st.dataframe(
        comparison.style.format("{:.3f}").highlight_max(axis=0, color="rgba(31, 138, 135, 0.35)"),
        width="stretch",
    )
    st.success(f"Best model by macro F1: **{comparison['F1'].idxmax()}**")

    with st.expander(f"Sample data (first 5 of {len(evaluation_data):,} rows)"):
        st.dataframe(evaluation_data.head(), hide_index=True, width="stretch")

elif view in MODEL_NAMES:
    detail = results[view]
    st.subheader(MODEL_NAMES[view])
    show_evaluation_source()
    # One metric card per column.
    metric_columns = st.columns(6)
    for column, metric in zip(metric_columns, METRIC_COLUMNS):
        column.metric(metric, f"{detail['metrics'][metric]:.3f}")

    # Two-column layout: confusion matrix on the left, classification report on the right.
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        st.markdown("#### Confusion matrix")
        # Rows are actual occupancy classes, columns are predicted classes.
        matrix = confusion_matrix(target, detail["predictions"], labels=CLASS_LABELS)
        figure, axis = plt.subplots(figsize=(5.4, 4.2))
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="crest",
            cbar=False,
            xticklabels=CLASS_LABELS,
            yticklabels=CLASS_LABELS,
            linewidths=0.5,
            linecolor="white",
            ax=axis,
        )
        axis.set_xlabel("Predicted occupancy")
        axis.set_ylabel("Actual occupancy")
        figure.tight_layout()
        st.pyplot(figure, width="stretch")
        plt.close(figure)

    with right:
        st.markdown("#### Classification report")
        # Per-class precision, recall and F1, returned as a dict so it can be shown as a table.
        report = classification_report(
            target,
            detail["predictions"],
            labels=CLASS_LABELS,
            target_names=[str(label) for label in CLASS_LABELS],
            output_dict=True,
            zero_division=0,
        )
        st.dataframe(pd.DataFrame(report).T.style.format("{:.3f}"), width="stretch")