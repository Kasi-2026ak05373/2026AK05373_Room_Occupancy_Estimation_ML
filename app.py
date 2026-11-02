"""Streamlit interface for evaluating the saved occupancy classifiers."""

from __future__ import annotations

import json
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
# Folder written by training.ipynb: fitted .joblib models and metadata.json.
MODEL_DIR = ROOT / "model"
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
def load_artifacts() -> tuple[dict, dict]:
    # metadata.json lists the model keys, feature names, target and class labels.
    with (MODEL_DIR / "metadata.json").open(encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    estimators = {
        key: joblib.load(MODEL_DIR / f"{key}.joblib")
        for key in metadata["models"]
    }
    return metadata, estimators


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
st.title("Room occupancy estimation")
st.caption("A four-class sensor benchmark · UCI Machine Learning Repository")

# Stop early with a clear message if training.ipynb has not been run yet.
if not (MODEL_DIR / "metadata.json").exists():
    st.error("Saved models are missing. Run all cells in `training.ipynb` before starting the app.")
    st.stop()

# Load the saved models and unpack the names the rest of the app needs.
metadata, estimators = load_artifacts()
feature_columns = metadata["features"]
target_column = metadata["target"]
class_labels = metadata["class_labels"]
# Maps model key (e.g. "knn") to display name (e.g. "K-Nearest Neighbors").
model_names = metadata["models"]

# Sidebar: upload control, model picker and a summary of the training data.
with st.sidebar:
    st.subheader("Evaluation data")
    uploaded_file = st.file_uploader("Upload a labeled test CSV", type=["csv"])
    st.caption("Include all 16 sensor columns and `Room_Occupancy_Count`.")
    selected_model = st.selectbox("Model detail", options=list(model_names), format_func=model_names.get)
    st.divider()
    st.caption(f"Trained on {metadata['train_rows']:,} rows · {len(feature_columns)} sensor features")

# Use the bundled held-out day unless the user uploads their own labeled CSV.
try:
    if uploaded_file is None:
        evaluation_data = pd.read_csv(ROOT / "test_data.csv")
        data_origin = f"Bundled held-out day: {metadata['test_day']}"
    else:
        evaluation_data = pd.read_csv(uploaded_file)
        data_origin = f"Uploaded file: {uploaded_file.name}"
except (OSError, pd.errors.ParserError) as error:
    st.error(f"Could not read the CSV: {error}")
    st.stop()

# Validate the data before scoring. Each check shows an error and stops the run if it fails.
# 1) All 16 sensor columns and the target column must be present.
required_columns = feature_columns + [target_column]
missing_columns = [column for column in required_columns if column not in evaluation_data.columns]
if missing_columns:
    st.error(f"Missing required columns: {', '.join(missing_columns)}")
    st.stop()
# 2) The file must contain at least one row.
if evaluation_data.empty:
    st.error("The uploaded CSV contains no rows.")
    st.stop()

# 3) Convert values to numbers. Anything non-numeric becomes NaN and is caught by the next check.
features = evaluation_data[feature_columns].apply(pd.to_numeric, errors="coerce")
target = pd.to_numeric(evaluation_data[target_column], errors="coerce")
if features.isna().any().any() or target.isna().any():
    st.error("Sensor features and target values must be numeric and contain no missing values.")
    st.stop()
# 4) Class labels must be whole numbers (for example 2.0 is accepted, 2.5 is not).
if not target.mod(1).eq(0).all():
    st.error("Occupancy target values must be whole numbers: 0, 1, 2, or 3.")
    st.stop()
# 5) Every label must be a class the models were trained on.
unknown_labels = sorted(set(target.astype(int)) - set(class_labels))
if unknown_labels:
    st.error(f"Target values not present in the trained models: {unknown_labels}")
    st.stop()
target = target.astype(int)
# 6) AUC needs at least one example of every class, so all four classes must be present.
if set(target) != set(class_labels):
    st.error("The labeled test data must contain all four occupancy classes to calculate multiclass AUC.")
    st.stop()

# Score every saved model on the same validated data.
results = {
    key: evaluate(estimator, features, target, class_labels)
    for key, estimator in estimators.items()
}
# Build the comparison table: one row per model, one column per metric.
comparison = pd.DataFrame(
    {model_names[key]: result["metrics"] for key, result in results.items()}
).T[METRIC_COLUMNS]

st.caption(f"{data_origin} · {len(evaluation_data):,} labeled observations")
st.subheader("Model comparison")
st.dataframe(
    comparison.style.format("{:.3f}"),
    use_container_width=True,
    hide_index=False,
)

# Detailed results for the model chosen in the sidebar.
detail = results[selected_model]
st.subheader(f"{model_names[selected_model]} · test results")
# One metric card per column.
metric_columns = st.columns(6)
for column, metric in zip(metric_columns, METRIC_COLUMNS):
    column.metric(metric, f"{detail['metrics'][metric]:.3f}")

# Two-column layout: confusion matrix on the left, classification report on the right.
left, right = st.columns([1, 1.15], gap="large")
with left:
    st.markdown("#### Confusion matrix")
    # Rows are actual occupancy classes, columns are predicted classes.
    matrix = confusion_matrix(target, detail["predictions"], labels=class_labels)
    figure, axis = plt.subplots(figsize=(5.4, 4.2))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="crest",
        cbar=False,
        xticklabels=class_labels,
        yticklabels=class_labels,
        linewidths=0.5,
        linecolor="white",
        ax=axis,
    )
    axis.set_xlabel("Predicted occupancy")
    axis.set_ylabel("Actual occupancy")
    figure.tight_layout()
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)

with right:
    st.markdown("#### Classification report")
    # Per-class precision, recall and F1, returned as a dict so it can be shown as a table.
    report = classification_report(
        target,
        detail["predictions"],
        labels=class_labels,
        target_names=[str(label) for label in class_labels],
        output_dict=True,
        zero_division=0,
    )
    st.dataframe(pd.DataFrame(report).T.style.format("{:.3f}"), use_container_width=True)