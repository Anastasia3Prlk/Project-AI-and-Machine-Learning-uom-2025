import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the Excel file
df = pd.read_excel("Dataset2Use_Assignment2.xlsx")

# Rename columns for easier reference
df.rename(columns={
    "ΕΝΔΕΙΞΗ ΑΣΥΝΕΠΕΙΑΣ (=2) (ν+1)": "Status",
    "ΕΤΟΣ": "Year"
}, inplace=True)

# Group by year and status
status_counts = df.groupby(["Year", "Status"]).size().unstack(fill_value=0)

# Grouped bar chart using matplotlib directly
years = status_counts.index.astype(str)
x = np.arange(len(years))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width/2, status_counts[1], width, label="Healthy (1)")
ax.bar(x + width/2, status_counts[2], width, label="Bankrupt (2)")

ax.set_title("Number of Healthy and Bankrupt Companies per Year")
ax.set_xlabel("Year")
ax.set_ylabel("Number of Companies")
ax.set_xticks(x)
ax.set_xticklabels(years, rotation=0)
ax.legend()
plt.tight_layout()
plt.show()
