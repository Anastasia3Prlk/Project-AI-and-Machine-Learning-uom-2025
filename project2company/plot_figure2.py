import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the Excel file
df = pd.read_excel("Dataset2Use_Assignment2.xlsx")

# Rename the column for company status
df.rename(columns={"ΕΝΔΕΙΞΗ ΑΣΥΝΕΠΕΙΑΣ (=2) (ν+1)": "Status"}, inplace=True)

# Select the first 8 columns (performance indicators)
features = df.columns[:8]

# Split the data into healthy and bankrupt companies
healthy = df[df["Status"] == 1]
bankrupt = df[df["Status"] == 2]

# Calculate statistics (min, mean, max) for each group
stats_healthy = pd.DataFrame({
    'Min': healthy[features].min(),
    'Mean': healthy[features].mean(),
    'Max': healthy[features].max()
})

stats_bankrupt = pd.DataFrame({
    'Min': bankrupt[features].min(),
    'Mean': bankrupt[features].mean(),
    'Max': bankrupt[features].max()
})

# Setup for bar chart
x = np.arange(len(features))      # X-axis positions
width = 0.25                      # Width of each bar

# Create two subplots (side by side)
fig, axs = plt.subplots(1, 2, figsize=(18, 6))

# -------- Plot for Healthy Companies --------
axs[0].bar(x - width, stats_healthy['Min'], width, label='Min')
axs[0].bar(x, stats_healthy['Mean'], width, label='Mean')
axs[0].bar(x + width, stats_healthy['Max'], width, label='Max')
axs[0].set_xticks(x)
axs[0].set_xticklabels(features, rotation=45, ha='right')
axs[0].set_yscale('log')  # Logarithmic scale
axs[0].set_title("Healthy Companies (Log Scale)")
axs[0].legend()
axs[0].grid(True, which="both", linestyle="--", linewidth=0.5)

# -------- Plot for Bankrupt Companies --------
axs[1].bar(x - width, stats_bankrupt['Min'], width, label='Min')
axs[1].bar(x, stats_bankrupt['Mean'], width, label='Mean')
axs[1].bar(x + width, stats_bankrupt['Max'], width, label='Max')
axs[1].set_xticks(x)
axs[1].set_xticklabels(features, rotation=45, ha='right')
axs[1].set_yscale('log')  # Logarithmic scale
axs[1].set_title("Bankrupt Companies (Log Scale)")
axs[1].legend()
axs[1].grid(True, which="both", linestyle="--", linewidth=0.5)

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
