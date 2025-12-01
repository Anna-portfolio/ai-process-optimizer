import pandas as pd
import numpy as np

# Synthetic dataset generator for Process Optimization Recommendations
# This script generates a dataset simulating internal business process data 
# used to train a recommendation/forecasting model

# Dataset includes: process volume, automation rate, avg handling time, error rate,
# satisfaction score, and business KPIs

# Dataset is saved to data/ as .csv

np.random.seed(42)

n_samples = 500

data = pd.DataFrame({
    "process_id": np.random.randint(1, 20, n_samples),
    "volume": np.random.randint(200, 5000, n_samples),
    "automation_rate": np.round(np.random.uniform(0.1, 0.95, n_samples), 2),
    "avg_handling_time": np.round(np.random.uniform(2.0, 45.0, n_samples), 1),
    "error_rate": np.round(np.random.uniform(0.0, 0.25, n_samples), 3),
    "customer_satisfaction": np.round(np.random.uniform(2.5, 5.0, n_samples), 2),
    "cost_per_case": np.round(np.random.uniform(1.0, 25.0, n_samples), 2)
})

# Create a target label as follows:
# High volume + high handling time + low automation -> recommendation to automate
# High error rate + low satisfaction -> recommendation to improve quality
# Else -> no change needed

conditions = []
for _, row in data.iterrows():
    if row["automation_rate"] < 0.4 and row["volume"] > 2000 and row["avg_handling_time"] > 20:
        conditions.append("increase_automation")
    elif row["error_rate"] > 0.15 and row["customer_satisfaction"] < 3.5:
        conditions.append("improve_quality")
    else:
        conditions.append("no_action")

data["recommended_action"] = conditions

# Save dataset
output_path = "data/synthetic_business_process_data.csv"
data.to_csv(output_path, index=False)

print(f"Synthetic dataset saved to {output_path}")
