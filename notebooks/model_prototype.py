import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load datasets and overview
df = pd.read_csv("data/synthetic_business_process_data.csv")
df_pred = pd.read_csv("data/recommended_actions.csv")


df.head()
df.describe()

# Merge original + predicted actions (if needed)
df_full = df_pred.copy()

# Visualize recommended actions counts
plt.figure(figsize=(6,4))
sns.countplot(x='predicted_action', data=df_full)
plt.title("Distribution of Predicted Actions")
plt.show()

# Scatter: volume vs handling time colored by recommendation
plt.figure(figsize=(8,5))
sns.scatterplot(
    x='volume', y='avg_handling_time', hue='predicted_action',
    data=df_full, palette='Set2'
)
plt.title("Volume vs Avg Handling Time by Predicted Action")
plt.show()

# Optional: interactive plot with plotly (run in browser)
#fig = px.scatter(df_full, x='volume', y='avg_handling_time',
              #   color='predicted_action',
              # hover_data=['process_id', 'automation_rate'])
#fig.show()
