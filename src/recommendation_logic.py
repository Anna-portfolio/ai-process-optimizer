import pandas as pd
import joblib

# Recommendation Logic
# This script loads the trained RandomForest model and label encoder,
# applies the model to new data, and outputs recommended actions.

# Load data
data_path = "data/synthetic_business_process_data.csv"
df = pd.read_csv(data_path)



# Load model and label encoder
model = joblib.load("model/random_forest_model.pkl")
le = joblib.load("model/label_encoder.pkl")

# Prepare features
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


# Predict recommended actions
df["predicted_action"] = le.inverse_transform(model.predict(X))


# Save results
output_path = "data/recommended_actions.csv"
df.to_csv(output_path, index=False)

print(f"Predicted recommendations saved to {output_path}")


# Display sample results
print(df[["process_id", "volume", "automation_rate", "predicted_action"]].head(10))
