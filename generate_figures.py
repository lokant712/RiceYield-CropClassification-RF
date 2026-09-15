"""
Generate all publication-quality figures for Experiment 08.
Conforming to Section 5 & Section 8 visualization and captioning guidelines.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Set professional plotting style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#eeeeee'
plt.rcParams['grid.linestyle'] = '--'

os.makedirs("figures", exist_ok=True)

# Load data and artifacts
manifest_df = pd.read_csv("artifacts/split_manifest.csv")
train_df = manifest_df[manifest_df['split'] == 'train'].copy()
val_df = manifest_df[manifest_df['split'] == 'validation'].copy()
test_df = manifest_df[manifest_df['split'] == 'test'].copy()

val_res_df = pd.read_csv("23MID0037_Lab08_Validation_Results.csv")
test_pred_df = pd.read_csv("artifacts/test_predictions.csv")
cls_df = pd.read_csv("data/raw/crop_recommendation.csv")
with open("artifacts/per_class.json", "r") as f:
    cls_summary = json.load(f)

# -------------------------------------------------------------------------
# Figure 1: Training Yield Distribution
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.2), dpi=300)
sns.histplot(train_df['yield_t_ha'], bins=40, kde=True, color='#1f77b4', edgecolor='#0f4c81', alpha=0.65, ax=ax)
mean_val = train_df['yield_t_ha'].mean()
median_val = train_df['yield_t_ha'].median()
ax.axvline(mean_val, color='#d62728', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.2f} t/ha')
ax.axvline(median_val, color='#2ca02c', linestyle='-.', linewidth=1.5, label=f'Median: {median_val:.2f} t/ha')
ax.set_title("Training Set Rice Yield Distribution (1997–2011)", fontsize=11, fontweight='bold', pad=12)
ax.set_xlabel("Rice Yield (tonnes / hectare)", fontsize=10)
ax.set_ylabel("Observation Count", fontsize=10)
ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
plt.tight_layout()
plt.savefig("figures/fig1_training_yield_distribution.png", dpi=300)
plt.close()

# -------------------------------------------------------------------------
# Figure 2: Training Year Coverage
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=300)
yr_counts = train_df['year'].value_counts().sort_index()
bars = ax.bar(yr_counts.index.astype(str), yr_counts.values, color='#3470a3', edgecolor='#1a4366', width=0.7)
ax.set_title("Annual Observation Coverage across Training Period (1997–2011)", fontsize=11, fontweight='bold', pad=12)
ax.set_xlabel("Harvest Year", fontsize=10)
ax.set_ylabel("District-Season Records", fontsize=10)
ax.set_ylim(0, max(yr_counts.values) * 1.15)
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 15, f"{int(yval)}", ha='center', va='bottom', fontsize=7.5, color='#333333')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("figures/fig2_training_year_coverage.png", dpi=300)
plt.close()

# -------------------------------------------------------------------------
# Figure 3: Validation Model Comparison Bar Chart
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.2), dpi=300)
x = np.arange(len(val_res_df))
width = 0.35
rects1 = ax.bar(x - width/2, val_res_df['val_mae_t_ha'], width, label='Validation MAE (t/ha)', color='#2b5c8f', edgecolor='#17395c')
rects2 = ax.bar(x + width/2, val_res_df['val_rmse_t_ha'], width, label='Validation RMSE (t/ha)', color='#e28743', edgecolor='#994d14')
ax.set_title("Validation Performance Across Candidate Models (2012–2013)", fontsize=11, fontweight='bold', pad=12)
ax.set_xlabel("Candidate Model", fontsize=10)
ax.set_ylabel("Error Metric (tonnes / hectare)", fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels([m.replace('_', ' ').title() for m in val_res_df['model']], fontsize=9.5)
ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
ax.set_ylim(0, max(val_res_df['val_rmse_t_ha']) * 1.2)
for r in rects1:
    ax.text(r.get_x() + r.get_width()/2.0, r.get_height() + 0.02, f"{r.get_height():.3f}", ha='center', va='bottom', fontsize=8)
for r in rects2:
    ax.text(r.get_x() + r.get_width()/2.0, r.get_height() + 0.02, f"{r.get_height():.3f}", ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig("figures/fig3_validation_model_comparison.png", dpi=300)
plt.close()

# -------------------------------------------------------------------------
# Figure 4: Locked-Test Actual vs. Predicted Scatter
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)
ax.scatter(test_pred_df['yield_t_ha'], test_pred_df['pred_yield_t_ha'], alpha=0.55, color='#1f77b4', edgecolors='none', s=24)
min_val = 0.0
max_val = max(test_pred_df['yield_t_ha'].max(), test_pred_df['pred_yield_t_ha'].max()) * 1.05
ax.plot([min_val, max_val], [min_val, max_val], color='#d62728', linestyle='--', linewidth=1.5, label='1:1 Ideal Identity (y = x)')
ax.set_xlim(min_val, max_val)
ax.set_ylim(min_val, max_val)
ax.set_title("Locked Test: Actual vs. Predicted Rice Yield (2014–2015)", fontsize=11, fontweight='bold', pad=12)
ax.set_xlabel("Actual Yield (tonnes / hectare)", fontsize=10)
ax.set_ylabel("Predicted Yield (tonnes / hectare)", fontsize=10)
ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper left', fontsize=9)
plt.tight_layout()
plt.savefig("figures/fig4_locked_test_actual_vs_predicted.png", dpi=300)
plt.close()

# -------------------------------------------------------------------------
# Figure 5: Locked-Test Residual Plot
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)
ax.scatter(test_pred_df['pred_yield_t_ha'], test_pred_df['residual_t_ha'], alpha=0.55, color='#4c72b0', edgecolors='none', s=24)
ax.axhline(0.0, color='#d62728', linestyle='--', linewidth=1.5, label='Zero Error Reference (e = 0)')
ax.set_title("Locked Test: Residuals vs. Predicted Yield (2014–2015)", fontsize=11, fontweight='bold', pad=12)
ax.set_xlabel("Predicted Yield (tonnes / hectare)", fontsize=10)
ax.set_ylabel("Residual: Actual - Predicted (tonnes / hectare)", fontsize=10)
ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right', fontsize=9)
plt.tight_layout()
plt.savefig("figures/fig5_locked_test_residual_plot.png", dpi=300)
plt.close()

# -------------------------------------------------------------------------
# Figure 6: Classification Extension - Training Label Distribution
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.2), dpi=300)
cls_counts = cls_df['label'].value_counts().sort_index()
ax.bar([l.title() for l in cls_counts.index], cls_counts.values, color='#2e8540', edgecolor='#1b5527', width=0.65)
ax.set_title("Balanced Crop-Label Class Distribution (22 Categories)", fontsize=11, fontweight='bold', pad=12)
ax.set_xlabel("Target Crop Class", fontsize=10)
ax.set_ylabel("Sample Count", fontsize=10)
ax.set_ylim(0, 130)
plt.xticks(rotation=60, ha='right', fontsize=8)
plt.tight_layout()
plt.savefig("figures/fig6_cls_training_label_distribution.png", dpi=300)
plt.close()

# -------------------------------------------------------------------------
# Figure 7: Classification Extension - Soil pH Distribution
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.2), dpi=300)
sns.histplot(cls_df['ph'], bins=30, kde=True, color='#8c564b', edgecolor='#54332c', alpha=0.65, ax=ax)
ax.axvline(cls_df['ph'].mean(), color='#d62728', linestyle='--', linewidth=1.5, label=f"Mean pH: {cls_df['ph'].mean():.2f}")
ax.axvline(7.0, color='#1f77b4', linestyle=':', linewidth=1.5, label="Neutral pH (7.0)")
ax.set_title("Soil pH Distribution Across Agronomic Crop Profiles", fontsize=11, fontweight='bold', pad=12)
ax.set_xlabel("Soil pH Level", fontsize=10)
ax.set_ylabel("Observation Count", fontsize=10)
ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
plt.tight_layout()
plt.savefig("figures/fig7_cls_soil_ph_distribution.png", dpi=300)
plt.close()

# -------------------------------------------------------------------------
# Figure 8: Classification Extension - Confusion Matrix Heatmap
# -------------------------------------------------------------------------
cls_bundle = joblib.load("models/classification_bundle.joblib")
labels = cls_bundle['classes']
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
X = cls_df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = cls_df['label']
X_tv, X_te, y_tv, y_te = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
y_pred_te = cls_bundle['pipeline'].predict(X_te)
cm = confusion_matrix(y_te, y_pred_te, labels=labels)

fig, ax = plt.subplots(figsize=(9, 7.5), dpi=300)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=[l.title() for l in labels],
            yticklabels=[l.title() for l in labels],
            ax=ax, annot_kws={"size": 7})
ax.set_title("Locked Test Confusion Matrix: Multi-Class Crop Classification", fontsize=11, fontweight='bold', pad=12)
ax.set_xlabel("Predicted Crop Label", fontsize=10)
ax.set_ylabel("True Crop Label", fontsize=10)
plt.xticks(rotation=60, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig("figures/fig8_cls_confusion_matrix.png", dpi=300)
plt.close()

print("All 8 publication figures generated successfully in figures/")
