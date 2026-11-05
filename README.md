# Room Occupancy Classification

## a. Problem statement

Estimate the number of people in a room (0, 1, 2, or 3) from environmental sensor readings. Five classification algorithms are trained on the same dataset and compared on a held-out day. The Streamlit app loads saved estimators and evaluates labeled test CSV files without fitting or retraining.

## b. Dataset description

**Room Occupancy Estimation**, UCI Machine Learning Repository, dataset 864. The data contains 10,129 observations collected from non-intrusive temperature, light, sound, CO2, and PIR sensors. The original table has 18 predictors plus the target; this project excludes the `Date` and `Time` fields and uses the 16 numeric sensor readings as predictors. The target, `Room_Occupancy_Count`, has four classes: 0, 1, 2, and 3. The dataset has no reported missing values.

Source: [UCI Room Occupancy Estimation](https://archive.ics.uci.edu/dataset/864/room+occupancy+estimation) · DOI: [10.24432/C5P605](https://doi.org/10.24432/C5P605) · License: CC BY 4.0.

The training notebook downloads the original UCI ZIP archive. One complete date containing all four classes is held out for testing; all other dates are used for training. This keeps temporally adjacent samples from the same day out of both partitions. The notebook prints the selected test date and row counts, and displays the scores. The app's feature names, target, classes, and model labels are defined in `app.py`.

## c. GitHub Repository Link

[https://github.com/Kasi-2026ak05373/2026AK05373_Room_Occupancy_Estimation_ML](https://github.com/Kasi-2026ak05373/2026AK05373_Room_Occupancy_Estimation_ML)

Live Streamlit app: [https://2026ak05373-room-occupancy-estimation-ml.streamlit.app/](https://2026ak05373-room-occupancy-estimation-ml.streamlit.app/)

## d. Models used

All scores below are calculated on the same whole-day test set. Precision, recall, and F1 use macro averaging so each occupancy class contributes equally. AUC is macro one-vs-rest. The result table is populated by running `training.ipynb`.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.8953 | 0.9837 | 0.9037 | 0.8937 | 0.8771 | 0.8635 |
| Decision Tree | 0.8181 | 0.8742 | 0.8344 | 0.8042 | 0.7798 | 0.7657 |
| KNN | 0.9083 | 0.9883 | 0.9035 | 0.9007 | 0.8922 | 0.8751 |
| Gaussian Naive Bayes | 0.6306 | 0.7938 | 0.5942 | 0.6656 | 0.5897 | 0.5462 |
| Support Vector Machine | 0.8208 | 0.9937 | 0.8735 | 0.8191 | 0.7870 | 0.7798 |

### Performance observations

The test day is imbalanced: 637 rows of class 0, 205 of class 1, 257 of class 2, and 363 of class 3. Macro-averaged scores are reported so that the smaller classes count equally. Per-class recall (class 0 / 1 / 2 / 3) was computed from the saved models on `test_data.csv`:

| Model | Recall class 0 | Recall class 1 | Recall class 2 | Recall class 3 |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.994 | 0.980 | 1.000 | 0.601 |
| Decision Tree | 0.998 | 0.941 | 0.911 | 0.366 |
| KNN | 0.992 | 0.956 | 0.949 | 0.705 |
| Gaussian Naive Bayes | 0.750 | 0.912 | 1.000 | 0.000 |
| Support Vector Machine | 0.972 | 0.946 | 1.000 | 0.358 |

Class 3 is the hardest class for every model. Most errors are class-3 rows predicted as class 2 (for example, 144 of 363 for Logistic Regression, 228 of 363 for Decision Tree, and 233 of 363 for SVM). This also pulls down class 2 precision (0.636 for Logistic Regression, 0.494 for Decision Tree, 0.497 for SVM). Gaussian NB never predicts class 3. Accuracy and macro F1 therefore hide a large part of the difference between models, and a model that looks strong on accuracy can still miss the four-occupant case.

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Recall of at least 0.98 on classes 0, 1 and 2, but only 0.601 on class 3 (144 of 363 predicted as class 2). Macro F1 0.8771 and MCC 0.8635 are the second-highest of the five. |
| Decision Tree | Recall of 0.911 on class 2 and 0.366 on class 3 (228 of 363 class-3 rows predicted as class 2). Macro F1 0.7798 and MCC 0.7657 are the second-lowest of the five. The class 2/3 split is the main source of error. |
| KNN | Best on macro F1 (0.8922), accuracy (0.9083), macro recall (0.9007) and MCC (0.8751). It also has the best class 3 recall (0.705), although 107 of 363 class-3 rows are still predicted as class 2. |
| Gaussian Naive Bayes | Lowest on accuracy (0.6306) and macro F1 (0.5897). It never predicts class 3 and predicts 115 class-0 rows as class 3. Its Gaussian feature assumption fits these sensor readings poorly. |
| Support Vector Machine | Highest macro AUC (0.9937), so its ranking of classes is strong. At the default decision rule, though, 233 of 363 class-3 rows are predicted as class 2, which gives a macro F1 of 0.7870 and places it below KNN and Logistic Regression. |
| Overall winner for this dataset | KNN, selected by the highest macro F1 (0.8922). It is also the best model on class 3 recall, but about 30% of class-3 rows are still predicted as class 2, which is the remaining limit of all five models on this held-out day. |

## Run locally

```bash
python -m pip install -r requirements.txt
```

Next, open `training.ipynb` in VS Code or Jupyter, select the project Python environment as the kernel, and run all cells. The notebook creates `model/` with all five fitted models and creates `test_data.csv` for the app. Then launch the app:

```bash
streamlit run app.py
```

The app is a single page. A left-hand sidebar navigates between the dataset overview (with a download link for `test_data.csv`), the model comparison, and one view per model showing its metrics, confusion matrix and classification report. The sidebar also has an optional CSV upload. Without an upload, the app scores the bundled held-out test data. An uploaded file must contain the 16 sensor columns and `Room_Occupancy_Count`, and it must include all four target classes so multiclass AUC is defined.

## Files

- `app.py`: Streamlit evaluation interface.
- `training.ipynb`: reproducible dataset download, validation, split, training, scoring, and artifact creation.
- `requirements.txt`: runtime and training dependencies.
- `test_data.csv`: held-out labeled sensor readings generated by the trainer.
- `model/`: five serialized fitted estimators generated by the trainer.