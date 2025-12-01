# AI Process Optimizer
created by Anna Dudek @Anna-portfolio

## Overview

This project is a prototype for an internal AI solution aimed at providing recommendations for process optimization based on synthetic business data. <br><br>
The workflow simulates a real-life scenario where business teams provide datasets, and machine learning models generate actionable insights, such as whether to increase automation or improve quality.<br><br>
The project demonstrates the end-to-end workflow from data generation to ML model training and recommendation visualization.
<br>
## Project Structure

```
project_root/
│
├─ data/                         # Synthetic datasets and prediction outputs
│   ├─ synthetic_business_process_data.csv
│   └─ recommended_actions.csv
│
├─ model/                        # Trained ML model and label encoder
│   ├─ random_forest_model.pkl
│   └─ label_encoder.pkl
│
├─ src/                          # Python scripts for workflow
│   ├─ generate_data.py          # Generates synthetic business data
│   ├─ train_model.py            # Trains RandomForest model on synthetic data
│   └─ recommendation_logic.py   # Applies model to data and outputs recommendations
│
├─ notebooks/                    # Prototypes 
│   └─ model_prototype.py        # Exploratory analysis & visualizations of predictions
│
└─ README.md                     # Project documentation
```

## Key Components

### 1. Data Generation

* `generate_data.py` creates a synthetic dataset of internal business processes.
* Features include: `process_id`, `volume`, `automation_rate`, `avg_handling_time`, `error_rate`, `customer_satisfaction`, `cost_per_case`.
* Target label (`recommended_action`) is generated based on simple business rules.

### 2. ML Model

* `train_model.py` trains a RandomForestClassifier to predict recommended actions.
* Includes label encoding, train-test split, and saving the trained model and label encoder.

### 3. Recommendation Logic

* `recommendation_logic.py` loads the trained model and label encoder.
* Applies the model to the dataset and outputs predicted actions to `data/recommended_actions.csv`.

### 4. Prototype

* `model_prototype.py` provides exploratory data analysis and visualizations.
* Bar charts, scatter plots, and interactive Plotly visualizations help understand predictions.

## Technologies / Stack

* **Python** (data processing, ML, scripting)
* **scikit-learn** (RandomForestClassifier)
* **pandas, numpy** (data manipulation)
* **matplotlib, seaborn, plotly** (visualizations)
* **joblib** (model persistence)


