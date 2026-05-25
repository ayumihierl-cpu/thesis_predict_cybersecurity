import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from matplotlib.patches import Patch
from scipy.stats import spearmanr

#------------------------------------------
# 1 Load fitted models and data 
#------------------------------------------
xgb_pipeline = joblib.load('results/xgb_2stage_pipeline.pkl')
 
from preprocessing import (
    df, X, y, X_trainval, X_test, y_trainval, y_test
)
 
print('Features:', X_trainval.columns.tolist())
print(f'Total features  : {X_trainval.shape[1]}')
print(f'Trainval samples: {X_trainval.shape[0]}')


#------------------------------------------
# 2 Readable feature name mapping 
#------------------------------------------
feature_names_map = {
    'inflation_cpi':                   'Inflation (CPI)',
    'govt_expenditure_pct_gdp':        'Govt Expenditure (% GDP)',
    'gross_capital_formation_pct_gdp': 'Gross Capital Formation',
    'trade_openness':                  'Trade Openness',
    'industry_share_gdp':              'Industry Share (GDP)',
    'gdp_pc_ppp':                      'GDP per Capita (PPP)',
    'comp_edu_duration':               'Compulsory Edu. Duration',
    'primary_enorllment_gross':        'Primary Enrollment',
    'secondary_enrollment_gross':      'Secondary Enrollment',
    'tertiary_enrollment_gross':       'Tertiary Enrollment',
    'govt_expenditure_edu_pct':        'Govt Edu. Expenditure',
    'unemployment_ilo':                'Unemployment Rate',
    'HCI_2020':                        'Human Capital Index',
}
 
feature_cols  = X_trainval.columns.tolist()
feature_names = [feature_names_map.get(c, c) for c in feature_cols]
 
print('\nFeature name mapping:')
for raw, readable in zip(feature_cols, feature_names):
    print(f'  {raw:40s} → {readable}')
#------------------------------------------
# 3 Extract tranformed data from fitted pipelines
#------------------------------------------
X_trainval_xgb = xgb_pipeline['preprocessor'].transform(X_trainval)

print(f'\nTransformed shape (XGBoost preprocessor): {X_trainval_xgb.shape}')

#------------------------------------------
# 4 SHAP values - XGBoost (TreeExplainer)
#   computes SHAP values based on the tree structure directly
#------------------------------------------
print('SHAP:', shap.__version__)

print('\nComputing SHAP values for XGBoost...')
 
explainer_xgb   = shap.TreeExplainer(xgb_pipeline['regressor'])
shap_values_xgb = explainer_xgb.shap_values(X_trainval_xgb)
 
shap_df_xgb = pd.DataFrame(shap_values_xgb, columns=feature_names)
print(f'XGBoost SHAP values computed — shape: {shap_df_xgb.shape}')


#------------------------------------------
# 6 Summary table
#------------------------------------------
def importance_table(shap_df, model_name):
    table = (
        shap_df.abs().mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    table.columns = ['Feature', 'Mean |SHAP|']
    print(f'\n{"="*55}')
    print(f'Mean Absolute SHAP Values — {model_name}')
    print(f'{"="*55}')
    print(table.to_string(index=False))
    return table
 
xgb_importance = importance_table(shap_df_xgb, 'XGBoost')

print(f'\n{"="*55}')


#------------------------------------------
# 7 Visualisations
#------------------------------------------
# ── Plot 1: SHAP summary bar — XGBoost mean absolute SHAP ───────────────────
fig, ax = plt.subplots(figsize=(9, 6))
 
ax.barh(
    xgb_importance['Feature'][::-1],
    xgb_importance['Mean |SHAP|'][::-1],
    color='#4575b4',
    edgecolor='white'
)
ax.tick_params(axis='y', labelsize=13)
ax.set_xlabel('Mean |SHAP Value|', fontsize=14)
ax.set_title(
    'Feature Importance — XGBoost (SHAP)\nMean Absolute SHAP Values',
    fontsize=14, fontweight='bold'
)
plt.tight_layout()
plt.savefig('results_visual/figures/subq3_shap_bar_xgb.pdf', dpi=300, bbox_inches='tight', format='pdf')
print('Saved: subq3_shap_bar_xgb.png and pdf')

# ── Plot 2: SHAP beeswarm — XGBoost ─────────────────────────────────────────
plt.figure(figsize=(10, 7))

shap.summary_plot(
    shap_values_xgb,
    X_trainval_xgb,
    feature_names=feature_names,
    show=False
)

plt.title(
    'SHAP Beeswarm Plot — XGBoost',
    fontsize=12, fontweight='bold'
)

plt.tight_layout()
plt.savefig('results_visual/figures/subq3_shap_beeswarm_xgb.png', dpi=300, bbox_inches='tight')
plt.show()
print('Saved: subq3_shap_beeswarm_xgb.png')


# ═══════════════════════════════════════════════════════════════════════════════
# SUPPLEMENTARY ANALYSIS — Augmented Data
# ═══════════════════════════════════════════════════════════════════════════════

print('\n' + '='*60)
print('SUPPLEMENTARY: SHAP Analysis — Augmented Models')
print('='*60)

# Load augmented models
lr_pipeline_aug  = joblib.load('results/lr_aug_pipeline.pkl')
xgb_pipeline_aug = joblib.load('results/xgb_aug_pipeline.pkl')

# Extract transformed data using augmented preprocessors
X_trainval_xgb_aug = xgb_pipeline_aug['preprocessor'].transform(X_trainval)
X_trainval_lr_aug  = lr_pipeline_aug['preprocessor'].transform(X_trainval)

# SHAP — augmented XGBoost
print('\nComputing SHAP values for XGBoost (augmented)...')
explainer_xgb_aug   = shap.TreeExplainer(xgb_pipeline_aug['regressor'])
shap_values_xgb_aug = explainer_xgb_aug.shap_values(X_trainval_xgb_aug)
shap_df_xgb_aug     = pd.DataFrame(shap_values_xgb_aug, columns=feature_names)

xgb_importance_aug = (
    shap_df_xgb_aug.abs().mean()
    .sort_values(ascending=False)
    .reset_index()
)
xgb_importance_aug.columns = ['Feature', 'Mean |SHAP|']
print('\nMean Absolute SHAP Values — XGBoost (Augmented):')
print(xgb_importance_aug.to_string(index=False))

# ── Rank correlation: original vs augmented ───────────────────────────────────
xgb_orig_ranked = xgb_importance.set_index('Feature')['Mean |SHAP|']
xgb_aug_ranked  = xgb_importance_aug.set_index('Feature')['Mean |SHAP|']
aligned_xgb     = pd.DataFrame({
    'Original': xgb_orig_ranked,
    'Augmented': xgb_aug_ranked
}).dropna()

corr_xgb, p_xgb = spearmanr(aligned_xgb['Original'], aligned_xgb['Augmented'])

print('\n' + '='*55)
print('XGBoost Feature Importance Rank Correlation: Original vs Augmented')
print('='*55)
print(f'Spearman r = {corr_xgb:.4f}, p = {p_xgb:.4f}')
if corr_xgb > 0.7:
    print('→ High agreement — augmentation did not change feature importance rankings')
elif corr_xgb > 0.4:
    print('→ Moderate agreement — some ranking shifts with augmentation')
else:
    print('→ Low agreement — augmentation substantially changed feature importance')

# ── Side-by-side comparison plot ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, importance_df, title, color in zip(
    axes,
    [xgb_importance, xgb_importance_aug],
    ['XGBoost SHAP — Original', 'XGBoost SHAP — Augmented'],
    ['#4575b4', '#d73027']
):
    ax.barh(
        importance_df['Feature'][::-1],
        importance_df['Mean |SHAP|'][::-1],
        color=color, edgecolor='white'
    )
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Mean |SHAP Value|', fontsize=13)
    ax.tick_params(axis='y', labelsize=13)

fig.suptitle(
    'XGBoost Feature Importance: Original vs Augmented Models\n'
    f'Spearman rank correlation: r = {corr_xgb:.3f}',
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig('results_visual/figures/subq3_shap_original_vs_aug.pdf',
            dpi=300, bbox_inches='tight', format='pdf')
plt.show()
print('Saved: subq3_shap_original_vs_aug.png + pdf')

print('\n>>> Sub-question 3 supplementary augmented analysis complete.')