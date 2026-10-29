import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.autolayout': True
})

# 1. Load all result CSVs from results2/
csv_files = glob.glob("./results2/*results*.csv") + glob.glob("results2/*results*.csv")
if not csv_files:
    raise FileNotFoundError("No CSV files found in results2/. Check your path.")

df_list = []
for f in csv_files:
    temp_df = pd.read_csv(f)
    match = re.search(r'results_(.*?)\.csv', f, re.IGNORECASE)
    temp_df['model'] = match.group(1) if match else 'Unknown'
    df_list.append(temp_df)

df = pd.concat(df_list, ignore_index=True)
df['method'] = df['method'].str.lower().str.strip()
df['model'] = df['model'].str.strip()

# Focus on clean runs for general scaling trends
clean_df = df[(df['label_noise_rate'].isna()) | (df['label_noise_rate'] == 0.0)]

# Define model ordering if present
model_order = [m for m in ['ResNet18', 'ViTSmall', 'ViTMedium'] if m in clean_df['model'].unique()]
if not model_order:
    model_order = clean_df['model'].unique().tolist()

# ---------------------------------------------------------
# MULTI-PANEL FIGURE: Scaling Impact on ECE, Accuracy, & OOD AUROC
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)

metrics = [
    ('accuracy', 'Clean Accuracy (%)', lambda x: x * 100, 'upper left'),
    ('ece', 'Expected Calibration Error (ECE)', lambda x: x, 'upper right'),
    ('ood_auroc', 'OOD AUROC', lambda x: x, 'lower right')
]

# Aggregate across seeds and ratios (or fix ratio = 0.5 for comparison)
# Here we aggregate across all ratios or pick ratio == 0.5
agg_df = clean_df.groupby(['model', 'method'])[['accuracy', 'ece', 'ood_auroc']].mean().reset_index()

for idx, (col, ylabel, transform, leg_loc) in enumerate(metrics):
    ax = axes[idx]
    
    # Pivot for bar plotting across models & methods
    pivot_data = agg_df.pivot(index='model', columns='method', values=col)
    pivot_data = pivot_data.reindex(model_order) # Order models by scale
    
    # Apply transformation (e.g., percentage for accuracy)
    pivot_data = pivot_data.apply(transform)
    
    pivot_data.plot(kind='bar', ax=ax, colormap='Set2', edgecolor='black', alpha=0.85, width=0.8)
    
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Model Architecture / Scale')
    ax.set_title(f'Impact of Scaling on {col.replace("_", " ").title()}', fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(title='Method', frameon=True, loc=leg_loc)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

plt.suptitle('Figure: Model Architecture Scaling vs. Calibration and Generalization', y=1.03, fontweight='bold', fontsize=14)
plt.savefig('fig_model_scaling_metrics.pdf', bbox_inches='tight')
plt.savefig('fig_model_scaling_metrics.png', bbox_inches='tight')
plt.close()

print("Successfully generated scaling impact figure: fig_model_scaling_metrics.pdf / .png")