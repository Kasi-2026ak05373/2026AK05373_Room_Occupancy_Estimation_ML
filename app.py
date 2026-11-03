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
# Inject custom CSS to style the metric cards, headings, sidebar and file uploader.
st.markdown(
    """
    <style>
    :root { --ink: #1d2928; --muted: #62706c; --green: #167c68; --line: #dce5e1; }
    .block-container { max-width: 1220px; padding-top: 2rem; padding-bottom: 3rem; }
    h1 { color: var(--ink); letter-spacing: 0; font-size: 2.15rem; }
    [data-testid="stMetric"] { background: #f0f5f2; border-left: 3px solid var(--green);
      padding: 0.9rem 1rem; border-radius: 4px; }
    [data-testid="stMetricLabel"] { color: var(--muted); font-weight: 500; }
    [data-testid="stMetricValue"] { color: var(--ink); font-weight: 600; font-size: 1.5rem; }
    [data-testid="stSidebar"] { border-right: 1px solid var(--line); }
    div[data-testid="stFileUploader"] { border: 1px dashed #8ca49b; border-radius: 5px; }
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


def mark_upload_changed() -> None:
    st.session_state["upload_changed"] = True


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
st.title("Room Occupancy Estimation")
st.caption(
    "A four-class sensor benchmark · "
    "UCI Room Occupancy Estimation · "
    "[View dataset here](https://archive.ics.uci.edu/dataset/864/room+occupancy+estimation)"
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

# Sidebar: upload control, model picker and a summary of the training data.
with st.sidebar:
    st.subheader("Evaluation data")
    uploaded_file = st.file_uploader(
        "Upload a labeled test CSV",
        type=["csv"],
        key="evaluation_upload",
        on_change=mark_upload_changed,
    )
    st.caption("Include all 16 sensor columns and `Room_Occupancy_Count`.")
    st.download_button(
        label="Download Test_Data.csv",
        data=(ROOT / "test_data.csv").read_bytes(),
        file_name="test_data.csv",
        mime="text/csv",
        icon=":material/download:",
        on_click="ignore",
        width="stretch",
    )
    selected_model = st.radio(
        "Select a saved model",
        options=list(MODEL_NAMES),
        format_func=MODEL_NAMES.get,
    )
    st.divider()
    st.caption(f"{len(FEATURE_COLUMNS)} sensor features · {len(MODEL_NAMES)} saved models")

# Use the bundled held-out day unless the user uploads their own labeled CSV.
try:
    if uploaded_file is None:
        evaluation_data = pd.read_csv(ROOT / "test_data.csv")
        data_origin = "Held-out test data"
    else:
        evaluation_data = pd.read_csv(uploaded_file)
        data_origin = f"Evaluated on : ({uploaded_file.name})"
except (OSError, pd.errors.ParserError) as error:
    st.error(f"Could not read the CSV: {error}")
    st.stop()

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
if st.session_state.pop("upload_changed", False):
    if uploaded_file is None:
        st.toast("Showing results for the held-out test data", icon=":material/check_circle:")
    else:
        st.toast(f"Results refreshed using {uploaded_file.name}", icon=":material/check_circle:")

# Build the comparison table: one row per model, one column per metric.
comparison = pd.DataFrame(
    {MODEL_NAMES[key]: result["metrics"] for key, result in results.items()}
).T[METRIC_COLUMNS]

if uploaded_file is None:
    result_source = data_origin
    st.info(
        f"**Model evaluation on held-out test data** · {len(evaluation_data):,} labeled rows.\n\n"
        "Metrics below are calculated by applying each saved model to this held-out test data."
    )
else:
    result_source = data_origin
    st.success(
        f"**Model evaluation on your uploaded data** · {uploaded_file.name} · "
        f"{len(evaluation_data):,} labeled rows.\n\n"
        
        
    )

st.subheader(f"Model comparison · {result_source}")
st.dataframe(
    comparison.style.format("{:.3f}"),
    width="stretch",
    hide_index=False,
)

# Detailed results for the model chosen in the sidebar.
detail = results[selected_model]
st.subheader(f"{MODEL_NAMES[selected_model]} · {result_source}")
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