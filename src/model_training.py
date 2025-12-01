import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# ML Model: RandomForestClassifier
# This script loads the synthetic_business_process_data.csv dataset, 
# preprocesses it, trains a baseline RandomForest model, evaluates it,
# and saves the trained model 

# Load data
data_path = "data/synthetic_business_process_data.csv"
df = pd.read_csv(data_path)

# Encode target label
le = LabelEncoder()
df["label_encoded"] = le.fit_transform(df["recommended_action"])

# Feature selection
features = [
    "process_id",
    "volume",
    "automation_rate",
    "avg_handling_time",
    "error_rate",
    "customer_satisfaction",
    "cost_per_case",
]
X = df[features]
y = df["label_encoded"]

# Train/validation split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=120, random_state=42)
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))


# Save model
import os
import joblib

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/random_forest_model.pkl")
joblib.dump(le, "model/label_encoder.pkl")

print("Model and label encoder saved to /model directory.")