import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Publication style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.autolayout': True
})

import glob
import re
import pandas as pd

csv_files = glob.glob("./results2/*results*.csv") + glob.glob("results2/*results*.csv")
df_list = []

for f in csv_files:
    temp_df = pd.read_csv(f)
    
    # Extract model name from filename (e.g., results2/42_0_results_ResNet18.csv -> ResNet18)
    match = re.search(r'results_(.*?)\.csv', f, re.IGNORECASE)
    if match:
        temp_df['model'] = match.group(1)
    else:
        temp_df['model'] = 'Unknown'
        
    df_list.append(temp_df)

df = pd.concat(df_list, ignore_index=True)
df['method'] = df['method'].str.lower().str.strip()

# df_list = [pd.read_csv(f) for f in csv_files]
# df = pd.concat(df_list, ignore_index=True)

# Normalize string formats
df['method'] = df['method'].str.lower().str.strip()
df['model'] = df['model'].str.strip()

# Clean baseline runs (label_noise_rate == 0 or NaN)
clean_df = df[(df['label_noise_rate'].isna()) | (df['label_noise_rate'] == 0.0)]

# ---------------------------------------------------------
# FIGURE 1: Clean Accuracy vs Pruning Ratio per Model
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(11, 4.5), dpi=300)
models_to_plot = ['ResNet18', 'ViTSmall', "ViTMedium"]

for i, m_name in enumerate(models_to_plot):
    ax = axes[i]
    m_data = clean_df[clean_df['model'] == m_name]
    if m_data.empty:
        continue

    grouped = m_data.groupby(['method', 'ratio'])['accuracy'].mean().reset_index()
    
    for method in ['random', 'lsh', 'hybrid', 'coreset']:
        sub = grouped[grouped['method'] == method].sort_values('ratio')
        if not sub.empty:
            fmt = 'o--' if method == 'random' else ('s-' if method == 'lsh' else ('^-.' if method == 'hybrid' else 'd-'))
            lw = 2.5 if method == 'coreset' else 1.8
            color = '#d62728' if method == 'coreset' else None
            ax.plot(sub['ratio'], sub['accuracy'] * 100, fmt, label=method.capitalize(), linewidth=lw, color=color)

    ax.set_xlabel('Dataset Retention Ratio')
    ax.set_ylabel('Clean Accuracy (%)')
    ax.set_title(f'{m_name}: Clean Accuracy Scaling', fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(frameon=True)

plt.suptitle('Figure 1: Clean Accuracy Drop at Low Pruning Ratios', y=1.02, fontweight='bold')
plt.savefig('fig_paper_clean_accuracy.pdf', bbox_inches='tight')
plt.close()

# ---------------------------------------------------------
# FIGURE 2: Instance Noise Enrichment & Z-Score
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

if 'label_noise_type' in df.columns and 'noise_enrichment' in df.columns:
    inst_df = df[df['label_noise_type'] == 'instance']
    
    if not inst_df.empty:
        # Panel A: Enrichment Factor
        enrich_summary = inst_df.groupby(['method', 'ratio'])['noise_enrichment'].mean().reset_index()
        pivot_enrich = enrich_summary.pivot(index='ratio', columns='method', values='noise_enrichment')
        pivot_enrich.plot(kind='bar', ax=axes[0], colormap='Set2', edgecolor='black', alpha=0.85)
        
        axes[0].axhline(1.0, color='red', linestyle='--', linewidth=1.5, label='Neutral (1.0x)')
        axes[0].set_ylabel('Noise Enrichment Factor')
        axes[0].set_xlabel('Retention Ratio')
        axes[0].set_title('(a) Mislabeled Sample Enrichment', fontweight='bold')
        axes[0].grid(axis='y', linestyle='--', alpha=0.5)
        axes[0].legend(title='Method', frameon=True)
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)

        # Panel B: Statistical Significance (Z-Score)
        z_summary = inst_df.groupby(['method', 'ratio'])['noise_enrich_z'].mean().reset_index()
        pivot_z = z_summary.pivot(index='ratio', columns='method', values='noise_enrich_z')
        pivot_z.plot(kind='bar', ax=axes[1], colormap='Set2', edgecolor='black', alpha=0.85)
        
        axes[1].axhline(2.0, color='red', linestyle='--', linewidth=1.5, label='Significance Threshold (|z|=2)')
        axes[1].set_ylabel('Z-Score (Unlikeliness by Chance)')
        axes[1].set_xlabel('Retention Ratio')
        axes[1].set_title('(b) Enrichment Statistical Z-Score', fontweight='bold')
        axes[1].grid(axis='y', linestyle='--', alpha=0.5)
        axes[1].legend(title='Method', frameon=True)
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)

plt.suptitle('Figure 2: Geometric Bias Under Instance Noise', y=1.02, fontweight='bold')
plt.savefig('fig_paper_instance_noise_bias.pdf', bbox_inches='tight')
plt.close()

# ---------------------------------------------------------
# FIGURE 3 (REPLACEMENT): Noise Memorization Rate (Massive Gap)
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)

if 'memorization_rate' in df.columns:
    # Filter for noisy runs (e.g., label_noise_rate > 0)
    noisy_df = df[df['label_noise_rate'] > 0.0]
    
    if not noisy_df.empty:
        mem_summary = noisy_df.groupby(['method', 'ratio'])['memorization_rate'].mean().reset_index()
        pivot_mem = mem_summary.pivot(index='ratio', columns='method', values='memorization_rate')
        
        pivot_mem.plot(kind='bar', ax=ax, colormap='Reds', edgecolor='black', alpha=0.85)
        
        ax.set_ylabel('Noise Memorization Rate (Higher = Worse)')
        ax.set_xlabel('Retention Ratio')
        ax.set_title('Figure 3: Coreset Memorizes Mislabeled Outliers Aggressively', fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.legend(title='Method', frameon=True)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

import glob, pandas as pd
df = pd.concat([pd.read_csv(f) for f in glob.glob("./results2/*results*.csv")], ignore_index=True)
df['method'] = df['method'].str.lower().str.strip()
print(df.groupby(['method', 'ratio'])['memorization_rate'].mean())

plt.savefig('fig_paper_memorization_gap.pdf', bbox_inches='tight')
plt.close()