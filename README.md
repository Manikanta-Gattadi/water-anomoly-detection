# PRAVAHA — Water Consumption Anomaly Classification

## Overview

PRAVAHA is a machine-learning-based water consumption anomaly classification system designed to identify unusual water-consumption observations relative to contextual and historical usage patterns.

The project combines:

- Water-consumption data
- Occupancy context
- Time-based features
- Historical usage features
- Rolling statistical features
- Supervised machine learning
- Streamlit-based deployment

The system classifies observations into two categories:

- **Normal (0)**
- **Anomaly (1)**

PRAVAHA is intended as a screening and decision-support prototype. An anomaly prediction does not by itself prove a physical water leak or equipment fault and should be followed by contextual or physical verification.

---

## Problem Statement

Water consumption changes naturally according to time of day, day of week, occupancy, and recent usage behaviour. Therefore, a consumption value cannot always be classified as unusual using a simple fixed threshold.

The objective of PRAVAHA is to investigate whether contextual and historical variables can help distinguish unusual water-consumption observations from normal observations.

---

## Objectives

1. Understand and prepare the water-consumption dataset.
2. Perform data-quality checks and exploratory analysis.
3. Investigate class imbalance and consumption patterns.
4. Create temporal, occupancy, lag, and rolling statistical features.
5. Train and compare multiple classification algorithms.
6. Evaluate models using metrics appropriate for imbalanced classification.
7. Analyse feature importance and model behaviour.
8. Save the trained model and preprocessing artifacts.
9. Build a Streamlit application for prediction.
10. Deploy the application using GitHub and Streamlit Community Cloud.

---

## Dataset

The project uses an **Aalborg residential water-consumption dataset** containing water-use and occupancy information.

### Dataset dimensions

| Dataset | Shape |
|---|---:|
| Daily dataset | 87,360 × 9 |
| Hourly dataset | 2,928 × 8 |
| Final modelling dataset | 87,300 × 9 |

### Daily dataset columns

- `building`
- `date_time`
- `date`
- `water_use_l`
- `occ_01`
- `occ_02`
- `occ_03`
- `occ_04`
- `occ_05`

### Hourly dataset columns

- `building`
- `date_time`
- `water_use_l`
- `water_use_l_high_res`
- `electrcity_use_kwh`
- `occupied`
- `washing`
- `dishwasher`

### Repository dataset

The repository includes the daily project dataset as:

```text
daily_data.csv
```

The complete analysis, feature engineering, model training, and evaluation workflow is available in:

```text
Pravaha.ipynb
```

### Target distribution

| Class | Meaning | Count | Approx. Share |
|---:|---|---:|---:|
| 0 | Normal | 85,908 | 98.41% |
| 1 | Anomaly | 1,392 | 1.59% |

The strong class imbalance is an important consideration during model evaluation.

---

## Methodology

The project follows the workflow:

```text
Raw Water Data
      ↓
Data Cleaning & Validation
      ↓
Feature Engineering
      ↓
Train / Validation Split
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Model Selection & Artifact Saving
      ↓
Streamlit Application
      ↓
Normal / Anomaly Prediction
```

---

## Feature Engineering

The final model uses nine engineered features:

| Feature | Description |
|---|---|
| `hour` | Hour extracted from the timestamp |
| `day_of_week` | Day of the week |
| `month` | Calendar month |
| `occupied_count` | Number of active occupancy indicators |
| `previous_usage` | Most recent previous usage |
| `previous_usage_2` | Second previous usage observation |
| `rolling_mean_6` | Recent six-observation rolling mean |
| `rolling_mean_24` | Recent 24-observation rolling mean |
| `rolling_std_24` | Recent 24-observation rolling standard deviation |

### Why these features?

- Temporal features capture recurring consumption patterns.
- Occupancy features provide behavioural context.
- Lag features describe recent consumption changes.
- Rolling statistics provide a local baseline and measure recent variability.

Historical features are constructed using previous observations so that information from the future is not intentionally introduced into the prediction input.

---

## Training Notebook

The repository includes `Pravaha.ipynb`, which documents the main data science and modelling workflow:

- Dataset loading and inspection
- Data preprocessing and quality checks
- Exploratory analysis
- Anomaly labelling
- Feature engineering
- Train/validation split
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Model evaluation
- Feature importance analysis
- Model and preprocessing artifact creation

The notebook is provided to make the modelling workflow easier to inspect and reproduce.

---

## Machine Learning Models

Three foundational supervised learning models were evaluated:

### 1. Logistic Regression

Used as a simple and interpretable baseline model.

### 2. K-Nearest Neighbors (KNN)

Used to investigate local/nonlinear relationships between observations.

### 3. Decision Tree

Used because it can represent nonlinear relationships and provides interpretable feature importance.

---

## Training and Validation

The recorded modelling workflow contains:

- **Training observations:** 52,380
- **Validation observations:** 17,460
- **Features:** 9

The project evaluates models using the same validation setting so that their recorded results can be compared consistently.

Because the target is highly imbalanced, accuracy is not treated as the only evaluation criterion.

---

## Model Evaluation

The following metrics are used:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

### Current validation results

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6100 | 0.0243 | 0.6007 | 0.0468 | 0.6717 |
| KNN | 0.9840 | 0.4286 | 0.0108 | 0.0211 | 0.5637 |
| Decision Tree | 0.4958 | 0.0268 | 0.8669 | 0.0519 | 0.7456 |

### Interpretation

The results demonstrate the importance of evaluating an imbalanced classification problem using multiple metrics.

KNN produces high accuracy but detects very few anomaly observations, as shown by its very low recall.

Logistic Regression detects more anomalies but has low precision.

The recorded Decision Tree results show the highest recall and ROC-AUC among the three tested models, while its precision remains low. This indicates a trade-off between detecting more labelled anomalies and producing false alarms.

---

## Decision Tree Feature Importance

The recorded Decision Tree feature importance is:

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `hour` | 0.6292 |
| 2 | `previous_usage` | 0.2621 |
| 3 | `occupied_count` | 0.0576 |
| 4 | `day_of_week` | 0.0327 |
| 5 | `rolling_mean_6` | 0.0184 |
| 6+ | Other features | Approximately 0 |

### Interpretation

`hour` is the most influential feature in the current Decision Tree, followed by `previous_usage`.

This indicates that time-of-day patterns and immediate recent consumption behaviour play an important role in the fitted decision structure.

Feature importance is model-specific and should not be interpreted as proof of a causal relationship.

---

## PRAVAHA Streamlit Application

The project includes a Streamlit web application that demonstrates the trained model in an accessible interface.

### Application workflow

```text
User Input
    ↓
Input Validation
    ↓
Feature Construction
    ↓
Saved Model + Scaler
    ↓
Prediction
    ↓
Normal / Anomaly Result
    ↓
Probability / Risk Information
```

The application loads the saved modelling artifacts rather than retraining the model whenever the application starts.

---

## Project Structure

The current repository contains:

```text
water-anomoly-detection/
│
├── Pravaha.ipynb
├── daily_data.csv
├── app.py
├── requirements.txt
├── README.md
│
├── best_model.pkl
├── scaler.pkl
├── feature_columns.txt
└── features_engineered.csv
```

### File descriptions

| File | Purpose |
|---|---|
| `Pravaha.ipynb` | Complete data analysis, feature engineering, model training, and evaluation notebook |
| `daily_data.csv` | Daily water-consumption dataset used in the project |
| `app.py` | Streamlit application and inference logic |
| `requirements.txt` | Required Python dependencies |
| `README.md` | Project documentation |
| `best_model.pkl` | Saved trained model used for prediction |
| `scaler.pkl` | Saved preprocessing/scaling object |
| `feature_columns.txt` | Expected feature names and ordering |
| `features_engineered.csv` | Engineered project dataset used in the modelling workflow |

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Manikanta-Gattadi/water-anomoly-detection.git
```

Move into the project directory:

```bash
cd water-anomoly-detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Application Locally

Run:

```bash
streamlit run app.py
```

The application will open in the browser.

---

## Run the Training Notebook

Open:

```text
Pravaha.ipynb
```

The notebook contains the project workflow for data preparation, feature engineering, model training, evaluation, and model artifact generation.

When reproducing the workflow, ensure that the required dataset and package versions are available.

---

## Deployment

The application can be deployed through **Streamlit Community Cloud** using the GitHub repository.

### Deployment flow

```text
GitHub Repository
       ↓
Streamlit Community Cloud
       ↓
Install requirements.txt
       ↓
Run app.py
       ↓
Load model artifacts
       ↓
PRAVAHA Web Application
```

The deployment requires the repository to contain the application file, dependency file, and all runtime model/data artifacts required by `app.py`.

---

## Model Artifacts

The deployed application uses the following saved artifacts:

### `best_model.pkl`

Contains the trained machine-learning model used for prediction.

### `scaler.pkl`

Contains the preprocessing/scaling object used during the modelling workflow.

### `feature_columns.txt`

Defines the expected feature names and ordering used by the model.

Keeping the feature definition separately helps prevent feature-order mismatches during deployment.

---

## Testing and Validation

The application should be tested for:

- Valid normal inputs
- Anomaly-like inputs
- Missing inputs
- Invalid numeric inputs
- Extreme values
- Model loading
- Feature-order consistency
- Prediction repeatability
- Correct deployment configuration

A successful application launch alone is not sufficient; the inference feature schema must also match the schema used during training.

---

## Limitations

1. The anomaly label depends on the project-defined labelling or threshold rule.
2. The dataset represents its source population and measurement context.
3. The anomaly class is highly imbalanced.
4. The prototype uses foundational machine-learning models.
5. The application does not directly diagnose physical leaks.
6. Probability estimates should be interpreted cautiously unless explicitly calibrated.
7. Building-specific conditions may differ from the dataset.
8. Additional variables such as weather, holidays, fixtures, maintenance events, and pressure may improve future versions.
9. The Streamlit application is a prototype rather than a production monitoring platform.

---

## Future Scope

Potential extensions include:

- Advanced time-series models
- Unsupervised and semi-supervised anomaly detection
- Building-specific baseline models
- Weather and holiday information
- Additional water-system variables
- Probability calibration
- Threshold optimization
- Model drift monitoring
- Alert history
- Human feedback on detected anomalies
- Independent external-dataset validation
- Explainable individual predictions
- Ensemble and deep-learning approaches

---

## Responsible Interpretation

PRAVAHA should be treated as a screening and decision-support system.

A prediction of **Anomaly** means that the observation resembles the anomaly pattern learned from the project-defined labelled data.

It does not establish that:

- a physical water leak exists,
- a pipe or appliance has failed,
- the sensor is faulty,
- or a particular physical cause is responsible.

A flagged observation should therefore be investigated using appropriate contextual and physical evidence.

---

## Reproducibility

For reproducible results:

1. Use the documented dataset.
2. Apply the documented preprocessing workflow.
3. Generate the same engineered features.
4. Preserve feature ordering.
5. Use the same training/validation procedure.
6. Record package versions.
7. Use the saved model and preprocessing artifacts.
8. Verify the deployed application against the final experiment.

---

## References

1. Schaffer, M., Jensen, R. L., Larsen, T. S., Marszal-Pomianowska, A., Rohde, L., Rubak, E., & Vera-Valdés, J. E. (2025). *Residential Household Dataset: Occupancy, Water, and Electricity Data*. Department of the Built Environment, Aalborg University, DCE Technical Reports No. 327. DOI: 10.54337/aau780546283.

2. Schaffer, M., Veit, M., Marszal-Pomianowska, A., Frandsen, M., Pomianowski, M. Z., Dichmann, E., Sørensen, C. G., & Kragh, J. (2024). *Dataset of smart heat and water meter data with accompanying building characteristics*. Data in Brief, 52, 109964. DOI: 10.1016/j.dib.2023.109964.

3. Candelieri, A. (2017). *Clustering and Support Vector Regression for Water Demand Forecasting and Anomaly Detection*. Water, 9(3), 224.

4. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825–2830.

---

## Project Status

**Status:** Completed prototype

**Application:** PRAVAHA — Water Consumption Anomaly Classification

**Technology:** Python, Pandas, NumPy, Scikit-learn, Streamlit

**Task:** Binary Classification

**Deployment:** GitHub + Streamlit Community Cloud
