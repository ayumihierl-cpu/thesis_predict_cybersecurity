# subquestion 2

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from itertools import combinations
from scipy import stats
from scipy.stats import kruskal, f_oneway, levene
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from preprocessing import preprocessor_non_tree, preprocessor_tree, X_trainval, X_test, y_trainval, y_test, df
from models import GAMRegressor
from config import RANDOM_STATE

#------------------------------------------
# 1. load saved models and data
#------------------------------------------
# for cross validation prediction
best_gam_params = joblib.load('results/best_gam_params.pkl')
best_svr_params = joblib.load('results/best_svr_params.pkl')
best_xgb_params = joblib.load('results/best_xgb_params.pkl')
 
print('Best GAM params :', best_gam_params)
print('Best SVR params :', best_svr_params)
print('Best XGB params :', best_xgb_params)

# Helper: strip 'regressor__' prefix from GridSearchCV param keys
def strip_prefix(params, prefix='regressor__'):
    return {k.replace(prefix, ''): v for k, v in params.items()}
 
# Unfitted pipeline definitions with best hyperparameters
lr_pipe = Pipeline([
    ('preprocessor', clone(preprocessor_non_tree)),
    ('regressor', LinearRegression())
])
 
gam_pipe = Pipeline([
    ('preprocessor', clone(preprocessor_non_tree)),
    ('regressor', GAMRegressor(**strip_prefix(best_gam_params)))
])
 
svr_pipe = Pipeline([
    ('preprocessor', clone(preprocessor_non_tree)),
    ('regressor', SVR(**strip_prefix(best_svr_params)))
])
 
xgb_pipe = Pipeline([
    ('preprocessor', clone(preprocessor_tree)),
    ('regressor', XGBRegressor(
        objective='reg:squarederror',
        random_state=RANDOM_STATE,
        n_jobs=1,
        **strip_prefix(best_xgb_params)
    ))
])
 
models = {
    'Linear Regression': lr_pipe,
    'GAM':               gam_pipe,
    'SVR':               svr_pipe,
    'XGBoost':           xgb_pipe
}



#------------------------------------------
# 2 Load country metadata
#------------------------------------------
# Pull country metadata for trainval and test sets
df_trainval_meta = df.loc[X_trainval.index, ['Country', 'c_code']].reset_index(drop=True)
 
print(f'\nTrain/val countries : {len(df_trainval_meta)}')

#------------------------------------------
# 3 Load and prepare income groups 
# FY22 = based on 2020 GNI data, consistent with 2020 analysis year
#------------------------------------------
income_raw = pd.read_csv(
    'wb_incomegroup.csv',
    sep=';',
    header=None
)

# Extract fiscal year row to find FY22 column position
fy_row       = income_raw.iloc[4].tolist()
fy_row_clean = [str(x).strip() for x in fy_row]

# Dynamically find FY22 column index
if 'FY22' in fy_row_clean:
    fy22_col = fy_row_clean.index('FY22')
    print(f'FY22 found at column index: {fy22_col}')
else:
    print('FY22 NOT FOUND — unique values in fiscal year row:')
    print(set(fy_row_clean))

# Slice country data — rows 11 to -11 removes metadata header and footer
country_data = income_raw.iloc[11:-11].copy()
country_data = country_data.reset_index(drop=True)

# Keep c_code, country name, and FY22 income group only
income_2020 = country_data[[0, 1, fy22_col]].copy()
income_2020.columns = ['c_code', 'Country_wb', 'income_group']

# Clean whitespace and convert to string
income_2020['c_code']       = income_2020['c_code'].astype(str).str.strip()
income_2020['income_group'] = income_2020['income_group'].astype(str).str.strip()

# Handle Yemen LM* edge case
income_2020['income_group'] = income_2020['income_group'].str.replace('*', '', regex=False)

# Remove rows with no classification (..)
income_2020 = income_2020[income_2020['income_group'] != '..'].copy()

# Map abbreviations to full labels
group_labels = {
    'L':  'Low income',
    'LM': 'Lower middle income',
    'UM': 'Upper middle income',
    'H':  'High income'
}
income_2020['income_group'] = income_2020['income_group'].map(group_labels)

# Check for any unmapped values
unmapped = income_2020['income_group'].isna().sum()
if unmapped > 0:
    print(f'Warning: {unmapped} rows could not be mapped — check raw values:')
    print(income_2020[income_2020['income_group'].isna()])

income_2020 = income_2020[income_2020['income_group'].notna()].copy()

# Ordered category for correct plot ordering
group_order = [
    'Low income',
    'Lower middle income',
    'Upper middle income',
    'High income'
]

income_2020['income_group'] = pd.Categorical(
    income_2020['income_group'], categories=group_order, ordered=True
)

print(f'\nTotal countries with income group: {len(income_2020)}')
print('\nIncome group counts:')
print(income_2020['income_group'].value_counts().reindex(group_order))

#----------------------------------------------

print(f'\ncheck if i should use test set')

from preprocessing import df, X_test, y_test

df_test_meta = df.loc[X_test.index, ['Country', 'c_code']].reset_index(drop=True)
df_test_meta = df_test_meta.merge(income_2020[['c_code', 'income_group']], 
                                   on='c_code', how='left')

print(df_test_meta['income_group'].value_counts().reindex(group_order))
print(f'Total test countries: {len(df_test_meta)}')

# number of obsevations for low income group is too small -> stick to out of fold predition 

#------------------------------------------
# 4. MGenerate out-of-fold predictions
#------------------------------------------
# Simple KFold — one honest prediction per country
oof_cv = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

results = []

for model_name, pipeline in models.items():
    print(f'\nGenerating out-of-fold predictions: {model_name}...')

    y_pred_oof = cross_val_predict(
        pipeline,
        X_trainval,
        y_trainval,
        cv=oof_cv,
        n_jobs=-1
    )

    oof_df = df_trainval_meta.copy()
    oof_df['model']     = model_name
    oof_df['y_true']    = y_trainval.values
    oof_df['y_pred']    = y_pred_oof
    oof_df['residual']  = oof_df['y_true'] - oof_df['y_pred']
    oof_df['abs_error'] = oof_df['residual'].abs()

    results.append(oof_df)
    print(f'  Countries included: {len(oof_df)}')

results_df = pd.concat(results, ignore_index=True)

# Merge income group onto results
results_df = results_df.merge(
    income_2020[['c_code', 'income_group']],
    on='c_code',
    how='left'
)

# Check merge quality
unmatched = results_df[results_df['model'] == 'Linear Regression']['income_group'].isna().sum()
if unmatched > 0:
    print(f'\nWarning: {unmatched} countries not matched to income group:')
    print(
        results_df[
            (results_df['model'] == 'Linear Regression') &
            (results_df['income_group'].isna())
        ][['Country', 'c_code']]
    )

# Drop unmatched countries
results_df = results_df[results_df['income_group'].notna()].copy()

print('\nIncome group distribution in analysis dataset:')
print(
    results_df[results_df['model'] == 'Linear Regression']['income_group']
    .value_counts()
    .reindex(group_order)
)


#------------------------------------------
# 5. Summary error table by income group
#------------------------------------------
print('\n' + '='*60)
print('Prediction Error by Income Group (Out-of-Fold)')
print('='*60)
 
for model_name in models.keys():
    subset = results_df[results_df['model'] == model_name]
 
    summary = (
        subset.groupby('income_group', observed=True)
        .agg(
            n            =('abs_error', 'count'),
            mean_mae     =('abs_error', 'mean'),
            median_mae   =('abs_error', 'median'),
            std          =('abs_error', 'std'),
            mean_residual=('residual',  'mean')
        )
        .reindex(group_order)
    )
 
    print(f'\n--- {model_name} ---')
    print(summary.round(4).to_string())


#------------------------------------------
# 6 Levene's test to check which test to use 
#------------------------------------------
# First test variance assumption
for model_name in models.keys():
    subset = results_df[results_df['model'] == model_name]
    groups = [
        subset[subset['income_group'] == g]['abs_error'].values
        for g in group_order
    ]
    
    # Levene's test for equal variances
    stat, p = levene(*groups)
    print(f'{model_name} - Levene test for equal variances: p = {p:.4f}')
    
    if p < 0.05:
        print('  → Unequal variances confirmed → Kruskal-Wallis appropriate')
        # Run Kruskal-Wallis
        kw_stat, kw_p = kruskal(*groups)
        print(f'  Kruskal-Wallis: H = {kw_stat:.4f}, p = {kw_p:.4f}')
    else:
        print('  → Equal variances → one-way ANOVA is appropriate')
        # Run one-way ANOVA instead
        f_stat, f_p = f_oneway(*groups)
        print(f'  One-way ANOVA: F = {f_stat:.4f}, p = {f_p:.4f}')



##------------------------------------------
# 6 Statistical tests
# Kruskal-Wallis test
#------------------------------------------

print('\n' + '='*60)
print('Kruskal-Wallis Test')
print('='*60)

for model_name in models.keys():
    subset = results_df[results_df['model'] == model_name]

    groups = [
        subset[subset['income_group'] == g]['abs_error'].values
        for g in group_order
        if len(subset[subset['income_group'] == g]) > 0
    ]

    if len(groups) < 2:
        print(f'\n{model_name}: not enough groups for testing')
        continue

    kw_stat, kw_p = kruskal(*groups)
    print(f'\n{model_name}:')
    print(f'  Kruskal-Wallis: H = {kw_stat:.4f}, p = {kw_p:.4f}')

    if kw_p < 0.05:
        print('  → Significant difference in errors across income groups (p < 0.05)')
    else:
        print('  → No significant difference detected (p >= 0.05)')

#------------------------------------------
# 7 Further testing — bootstrap confidence intervals
#------------------------------------------

# bootstrap
def bootstrap_mean_diff(a, b, n_bootstrap=10000, random_state=80):
    rng = np.random.default_rng(random_state)
    diffs = []
    for _ in range(n_bootstrap):
        sample_a = rng.choice(a, size=len(a), replace=True)
        sample_b = rng.choice(b, size=len(b), replace=True)
        diffs.append(np.mean(sample_a) - np.mean(sample_b))
    
    diffs = np.array(diffs)
    ci_lower = np.percentile(diffs, 2.5)
    ci_upper = np.percentile(diffs, 97.5)
    observed_diff = np.mean(a) - np.mean(b)
    return observed_diff, ci_lower, ci_upper

print('\nBootstrap 95% Confidence Intervals for Mean MAE Differences')
print('='*60)

for model_name in models.keys():
    subset = results_df[results_df['model'] == model_name]
    print(f'\n{model_name}:')

    for g1, g2 in combinations(group_order, 2):
        a = subset[subset['income_group'] == g1]['abs_error'].values
        b = subset[subset['income_group'] == g2]['abs_error'].values

        if len(a) == 0 or len(b) == 0:
            continue

        diff, ci_low, ci_high = bootstrap_mean_diff(a, b)
        excludes_zero = '*' if ci_low > 0 or ci_high < 0 else ''
        print(f'  {g1} vs {g2}:')
        print(f'    Mean diff = {diff:.4f}, 95% CI [{ci_low:.4f}, {ci_high:.4f}] {excludes_zero}')

#------------------------------------------
# 8 Visualisations
#------------------------------------------

palette = {
    'Low income':          '#d73027',
    'Lower middle income': '#fc8d59',
    'Upper middle income': '#91bfdb',
    'High income':         '#4575b4'
}
 
short_labels = ['Low', 'Lower Mid.', 'Upper Mid.', 'High']
model_names  = list(models.keys())
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# SUPPLEMENTARY ANALYSIS — Augmented Data
# Primary results above use original unaugmented training data.
# This section repeats the income group error analysis using models trained
# on augmented data, as an additional analysis to assess robustness.
# ═══════════════════════════════════════════════════════════════════════════════

print('\n' + '='*60)
print('SUPPLEMENTARY: Income Group Error Analysis — Augmented Models')
print('='*60)

# Load augmented models — trained on cluster-aware jittered data
aug_models = {
    'Linear Regression': joblib.load('results/lr_aug_pipeline.pkl'),
    'GAM':               joblib.load('results/gam_aug_pipeline.pkl'),
    'SVR':               joblib.load('results/svr_aug_pipeline.pkl'),
    'XGBoost':           joblib.load('results/xgb_aug_pipeline.pkl'),
}

# Load augmented best params for OOF pipeline definitions
best_gam_params_aug = joblib.load('results/best_gam_params_aug.pkl')
best_svr_params_aug = joblib.load('results/best_svr_params_aug.pkl')
best_xgb_params_aug = joblib.load('results/best_xgb_params_aug.pkl')

aug_pipelines = {
    'Linear Regression': Pipeline([
        ('preprocessor', clone(preprocessor_non_tree)),
        ('regressor', LinearRegression())
    ]),
    'GAM': Pipeline([
        ('preprocessor', clone(preprocessor_non_tree)),
        ('regressor', GAMRegressor(**strip_prefix(best_gam_params_aug)))
    ]),
    'SVR': Pipeline([
        ('preprocessor', clone(preprocessor_non_tree)),
        ('regressor', SVR(**strip_prefix(best_svr_params_aug)))
    ]),
    'XGBoost': Pipeline([
        ('preprocessor', clone(preprocessor_tree)),
        ('regressor', XGBRegressor(
            objective='reg:squarederror',
            random_state=RANDOM_STATE,
            n_jobs=1,
            **strip_prefix(best_xgb_params_aug)
        ))
    ]),
}

# Generate OOF predictions using augmented pipeline definitions
results_aug = []

for model_name, pipeline in aug_pipelines.items():
    print(f'\nGenerating OOF predictions (aug hyperparams): {model_name}...')

    y_pred_oof = cross_val_predict(
        pipeline,
        X_trainval,
        y_trainval,
        cv=oof_cv,
        n_jobs=-1
    )

    oof_df = df_trainval_meta.copy()
    oof_df['model']     = model_name
    oof_df['y_true']    = y_trainval.values
    oof_df['y_pred']    = y_pred_oof
    oof_df['residual']  = oof_df['y_true'] - oof_df['y_pred']
    oof_df['abs_error'] = oof_df['residual'].abs()
    results_aug.append(oof_df)

results_aug_df = pd.concat(results_aug, ignore_index=True)
results_aug_df = results_aug_df.merge(
    income_2020[['c_code', 'income_group']], on='c_code', how='left'
)
results_aug_df = results_aug_df[results_aug_df['income_group'].notna()].copy()

# ── Summary table ─────────────────────────────────────────────────────────────
print('\nPrediction Error by Income Group (Augmented Models — OOF)')
for model_name in aug_pipelines.keys():
    subset = results_aug_df[results_aug_df['model'] == model_name]
    summary = (
        subset.groupby('income_group', observed=True)
        .agg(
            n            =('abs_error', 'count'),
            mean_mae     =('abs_error', 'mean'),
            median_mae   =('abs_error', 'median'),
            std          =('abs_error', 'std'),
            mean_residual=('residual',  'mean')
        )
        .reindex(group_order)
    )
    print(f'\n--- {model_name} ---')
    print(summary.round(4).to_string())

# ── Kruskal-Wallis ────────────────────────────────────────────────────────────
print('\nKruskal-Wallis Test (Augmented Models)')
print('='*60)

for model_name in aug_pipelines.keys():
    subset = results_aug_df[results_aug_df['model'] == model_name]
    groups = [
        subset[subset['income_group'] == g]['abs_error'].values
        for g in group_order
        if len(subset[subset['income_group'] == g]) > 0
    ]
    if len(groups) < 2:
        continue
    kw_stat, kw_p = kruskal(*groups)
    print(f'\n{model_name}: H = {kw_stat:.4f}, p = {kw_p:.4f}')
    if kw_p < 0.05:
        print('  → Significant difference across income groups (p < 0.05)')
    else:
        print('  → No significant difference detected (p >= 0.05)')

# ── Bootstrap CIs ─────────────────────────────────────────────────────────────
print('\nBootstrap 95% CIs — Augmented Models')
print('='*60)

for model_name in aug_pipelines.keys():
    subset = results_aug_df[results_aug_df['model'] == model_name]
    print(f'\n{model_name}:')
    for g1, g2 in combinations(group_order, 2):
        a = subset[subset['income_group'] == g1]['abs_error'].values
        b = subset[subset['income_group'] == g2]['abs_error'].values
        if len(a) == 0 or len(b) == 0:
            continue
        diff, ci_low, ci_high = bootstrap_mean_diff(a, b)
        excludes_zero = '*' if ci_low > 0 or ci_high < 0 else ''
        print(f'  {g1} vs {g2}:')
        print(f'    Mean diff = {diff:.4f}, 95% CI [{ci_low:.4f}, {ci_high:.4f}] {excludes_zero}')

# ── Comparison plot: original vs augmented MAE by income group ────────────────
MODEL_COLORS = ['#4C78A8', '#54A24B', '#F58518', '#B279A2']
# '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, df_plot, title in zip(
    axes,
    [results_df, results_aug_df],
    ['Original Models', 'Augmented Models']
):
    mean_errors = (
        df_plot.groupby(['model', 'income_group'], observed=True)['abs_error']
        .mean().reset_index()
    )
    mean_errors['income_group'] = pd.Categorical(
        mean_errors['income_group'], categories=group_order, ordered=True
    )
    x     = np.arange(len(group_order))
    width = 0.2
    for j, (model_name, color) in enumerate(zip(models.keys(), MODEL_COLORS)):
        subset = mean_errors[mean_errors['model'] == model_name].sort_values('income_group')
        ax.bar(x + j * width, subset['abs_error'],
               width=width, label=model_name, color=color,
               edgecolor='white', linewidth=0.5)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(['Low', 'LowMid', 'UpMid', 'High'], fontsize=14)
    ax.set_ylabel('Mean Absolute Error (GCI points)', fontsize=14)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)

fig.suptitle(
    'Mean MAE by Income Group: Original vs Augmented Models',
    fontsize=14, fontweight='bold'
)
plt.tight_layout()
plt.savefig('results_visual/figures/subq2_mae_original_vs_aug.pdf',
            dpi=300, bbox_inches='tight', format='pdf')
print('Saved: subq2_mae_original_vs_aug.png + pdf')

print('\n>>> Sub-question 2 supplementary augmented analysis complete.')


results_df.to_csv('results/sq2_results_df.csv', index=False)

# ── Comparison table: original vs augmented MAE by income group ───────────────
rows = []

for model_name in models.keys():
    for group in group_order:
        orig_mae = (
            results_df[
                (results_df['model'] == model_name) &
                (results_df['income_group'] == group)
            ]['abs_error'].mean()
        )
        aug_mae = (
            results_aug_df[
                (results_aug_df['model'] == model_name) &
                (results_aug_df['income_group'] == group)
            ]['abs_error'].mean()
        )
        rows.append({
            'Model':        model_name,
            'Income Group': group,
            'MAE (orig)':   round(orig_mae, 3),
            'MAE (aug)':    round(aug_mae,  3),
            'Δ MAE':        round(aug_mae - orig_mae, 3),
        })

comparison_df = pd.DataFrame(rows)

# ── LaTeX output ──────────────────────────────────────────────────────────────
latex = r"""\begin{table}
    \caption{Mean absolute error (MAE) by income group and model: 
             original versus Gaussian Copula augmented training data.
             $\Delta$ MAE = augmented $-$ original; negative = augmentation improved performance.}
    \label{tab:sq2_aug_comparison}
    \centering
    \small
    \begin{tabular}{llrrr}
        \toprule
        Model & Income Group & MAE (orig) & MAE (aug) & $\Delta$ MAE \\
        \midrule
"""

for model_name in models.keys():
    latex += f'        \\multicolumn{{5}}{{l}}{{\\textit{{{model_name}}}}} \\\\\n'
    subset = comparison_df[comparison_df['Model'] == model_name]
    for _, row in subset.iterrows():
        g = (row['Income Group']
             .replace('Lower middle income', 'Lower middle')
             .replace('Upper middle income', 'Upper middle')
             .replace(' income', ''))
        sign  = '+' if row['Δ MAE'] >= 0 else ''
        latex += (f'        & {g} & {row["MAE (orig)"]:.3f} & '
                  f'{row["MAE (aug)"]:.3f} & {sign}{row["Δ MAE"]:.3f} \\\\\n')
    latex += '        \\midrule\n'

latex = latex.rstrip()
if latex.endswith('\\midrule'):
    latex = latex[:-len('\\midrule')]

latex += r"""        \bottomrule
    \end{tabular}
\end{table}
"""

with open('results_visual/tables/table_sq2_aug_comparison.tex', 'w') as f:
    f.write(latex)
print('Saved: results_visual/tables/table_sq2_aug_comparison.tex')

# ─────────────────────────────────────────────────────────────────────────────
# THESIS OUTPUT SECTION
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# LaTeX export helper
# Wraps a DataFrame in a full LaTeX table environment with booktabs styling.
# Requires \usepackage{booktabs} in your LaTeX preamble.
# ─────────────────────────────────────────────────────────────────────────────
def save_latex_table(df, filepath, caption, label, col_format=None,
                     midrule_col=None, midrule_val=None):
    """
    Save a DataFrame as a LaTeX table file.
 
    Parameters
    ----------
    df          : pd.DataFrame
    filepath    : str  — output .tex file path
    caption     : str  — table caption
    label       : str  — LaTeX label for \\ref{}
    col_format  : str  — LaTeX column format string, e.g. 'llrrrr'
                         auto-generated if None
    midrule_col : str  — column name to insert \\midrule when value changes
    midrule_val : any  — not used; midrule inserted on every group change
    """
    n_cols = len(df.columns)
    if col_format is None:
        col_format = 'l' + 'r' * (n_cols - 1)
 
    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'    \centering')
    lines.append(r'    \small')
    lines.append(f'    \\caption{{{caption}}}')
    lines.append(f'    \\label{{tab:{label}}}')
    lines.append(f'    \\begin{{tabular}}{{{col_format}}}')
    lines.append(r'        \toprule')
 
    # Header row
    header = ' & '.join(str(c) for c in df.columns) + r' \\'
    lines.append(f'        {header}')
    lines.append(r'        \midrule')
 
    # Data rows — insert \midrule when group column changes
    prev_val = None
    for _, row in df.iterrows():
        if midrule_col and row[midrule_col] != prev_val and prev_val is not None:
            lines.append(r'        \midrule')
        prev_val = row[midrule_col] if midrule_col else None
 
        # Escape % and & in cell values, format numbers
        cells = []
        for val in row:
            if isinstance(val, float):
                cells.append(f'{val:.3f}')
            else:
                cells.append(str(val).replace('%', r'\%').replace('&', r'\&'))
        lines.append('        ' + ' & '.join(cells) + r' \\')
 
    lines.append(r'        \bottomrule')
    lines.append(r'    \end{tabular}')
    lines.append(r'\end{table}')
 
    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))
    print(f'Saved LaTeX: {filepath}')
 
 
# ─────────────────────────────────────────────────────────────────────────────
# TABLE 1: Mean MAE and signed residual by income group and model
# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '='*60)
print('TABLE 1: Mean MAE and Mean Residual by Income Group')
print('='*60)
 
rows = []
for model_name in model_names:
    subset = results_df[results_df['model'] == model_name]
    for group in group_order:
        g = subset[subset['income_group'] == group]
        if len(g) == 0:
            continue
        rows.append({
            'Model':          model_name,
            'Income Group':   group,
            'N':              len(g),
            'Mean MAE':       round(g['abs_error'].mean(), 3),
            'Mean Residual':  round(g['residual'].mean(), 3),
        })
 
table1 = pd.DataFrame(rows)
print(table1.to_string(index=False))
table1.to_csv('results_visual/tables/sq2_table1_mae_by_group.csv', index=False)
print('Saved: results_visual/tables/sq2_table1_mae_by_group.csv')
save_latex_table(
    table1,
    'results_visual/tables/sq2_table1_mae_by_group.tex',
    caption='Mean absolute error and signed residual by income group and model (out-of-fold predictions).',
    label='sq2_mae_by_group',
    col_format='llrrrrrr',
    midrule_col='Model'
)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# TABLE 2: Kruskal-Wallis results per model
# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '='*60)
print('TABLE 2: Kruskal-Wallis Test Results')
print('='*60)
 
kw_rows = []
for model_name in model_names:
    subset = results_df[results_df['model'] == model_name]
    groups = [
        subset[subset['income_group'] == g]['abs_error'].values
        for g in group_order
        if len(subset[subset['income_group'] == g]) > 0
    ]
    h_stat, p_val = kruskal(*groups)
    sig = 'Yes' if p_val < 0.05 else 'No'
    kw_rows.append({
        'Model':       model_name,
        'H statistic': round(h_stat, 3),
        'p-value':     round(p_val, 4),
        'Significant': sig
    })
    print(f'{model_name}: H = {h_stat:.3f}, p = {p_val:.4f} ({sig})')
 
table2 = pd.DataFrame(kw_rows)
table2.to_csv('results_visual/tables/sq2_table2_kruskal_wallis.csv', index=False)
print('Saved: results_visual/tables/sq2_table2_kruskal_wallis.csv')
save_latex_table(
    table2,
    'results_visual/tables/sq2_table2_kruskal_wallis.tex',
    caption='Kruskal-Wallis test results for prediction error differences across income groups.',
    label='sq2_kruskal_wallis',
    col_format='lrrr'
)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# TABLE 3: Bootstrap confidence intervals — pairwise comparisons
# ─────────────────────────────────────────────────────────────────────────────
def bootstrap_mean_diff(a, b, n_bootstrap=10000, random_state=80):
    rng = np.random.default_rng(random_state)
    diffs = [
        np.mean(rng.choice(a, size=len(a), replace=True)) -
        np.mean(rng.choice(b, size=len(b), replace=True))
        for _ in range(n_bootstrap)
    ]
    diffs    = np.array(diffs)
    ci_lower = np.percentile(diffs, 2.5)
    ci_upper = np.percentile(diffs, 97.5)
    obs_diff = np.mean(a) - np.mean(b)
    return obs_diff, ci_lower, ci_upper
 
print('\n' + '='*60)
print('TABLE 3: Bootstrap 95% CI — Pairwise MAE Comparisons')
print('='*60)
 
ci_rows = []
for model_name in model_names:
    subset = results_df[results_df['model'] == model_name]
    for g1, g2 in combinations(group_order, 2):
        a = subset[subset['income_group'] == g1]['abs_error'].values
        b = subset[subset['income_group'] == g2]['abs_error'].values
        if len(a) == 0 or len(b) == 0:
            continue
        diff, ci_low, ci_high = bootstrap_mean_diff(a, b)
        sig = 'Yes' if (ci_low > 0 or ci_high < 0) else 'No'
        ci_rows.append({
            'Model':            model_name,
            'Group 1':          g1,
            'Group 2':          g2,
            'Mean Difference':  round(diff, 3),
            '95% CI Lower':     round(ci_low, 3),
            '95% CI Upper':     round(ci_high, 3),
            'Excludes Zero':    sig
        })
 
table3 = pd.DataFrame(ci_rows)
print(table3.to_string(index=False))
table3.to_csv('results_visual/tables/sq2_table3_bootstrap_ci.csv', index=False)
print('Saved: results_visual/tables/sq2_table3_bootstrap_ci.csv')
save_latex_table(
    table3,
    'results_visual/tables/sq2_table3_bootstrap_ci.tex',
    caption='Bootstrap 95\\% confidence intervals for pairwise mean MAE differences across income groups.',
    label='sq2_bootstrap_ci',
    col_format='llrrrr',
    midrule_col='Model'
)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1: Grouped bar chart — Mean MAE by income group and model
# ─────────────────────────────────────────────────────────────────────────────
mean_errors = (
    results_df.groupby(['model', 'income_group'], observed=True)['abs_error']
    .mean()
    .reset_index()
)
mean_errors['income_group'] = pd.Categorical(
    mean_errors['income_group'], categories=group_order, ordered=True
)
 
fig, ax = plt.subplots(figsize=(11, 5))
x     = np.arange(len(group_order))
width = 0.2
colors = ['#4C78A8', '#54A24B', '#F58518', '#B279A2']
 
for j, (model_name, color) in enumerate(zip(model_names, colors)):
    subset = mean_errors[mean_errors['model'] == model_name].sort_values('income_group')
    ax.bar(
        x + j * width,
        subset['abs_error'],
        width=width,
        label=model_name,
        color=color,
        edgecolor='white',
        linewidth=0.5
    )
 
ax.set_xticks(x + width * (len(model_names) - 1) / 2)
ax.set_xticklabels(group_order, fontsize=14)
ax.set_ylabel('MAE (GCI points)', fontsize=14)
ax.set_title(
    'Mean Absolute Error by Income Group and Model\n(Out-of-Fold Predictions)',
    fontsize=14, fontweight='bold'
)
ax.legend(fontsize=11, framealpha=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('results_visual/figures/sq2_fig1_mae_grouped_bar.pdf', bbox_inches='tight')
print('Saved: sq2_fig1_mae_grouped_bar')
 

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4: Bootstrap CI heatmap — significance of pairwise differences
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
group_short = {
    'Low income':          'L',
    'Lower middle income': 'LM',
    'Upper middle income': 'UM',
    'High income':         'H'
}
pair_labels = [f'{group_short[g1]} vs {group_short[g2]}'
               for g1, g2 in combinations(group_order, 2)]
 
# Build significance matrix for each model
for i, model_name in enumerate(model_names):
    ax      = axes[i]
    model_ci = table3[table3['Model'] == model_name].copy()
    sigs = []
    diffs = []
    for g1, g2 in combinations(group_order, 2):
        row = model_ci[
            (model_ci['Group 1'] == g1) & (model_ci['Group 2'] == g2)
        ]
        if len(row) == 0:
            sigs.append(0)
            diffs.append(0)
        else:
            sigs.append(1 if row['Excludes Zero'].values[0] == 'Yes' else 0)
            diffs.append(row['Mean Difference'].values[0])
 
    # Display as a bar chart of mean differences coloured by significance
    colors_bar = ['#d73027' if s == 1 else '#92c5de' for s in sigs]
    ax.barh(pair_labels, diffs, color=colors_bar, edgecolor='white')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_title(model_name, fontsize=12, fontweight='bold')
    ax.set_xlabel('Mean MAE Difference\n(Group 1 − Group 2)', fontsize=14)
    ax.tick_params(axis='y', labelsize=14) 
    ax.tick_params(axis='x', labelsize=14)   
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
 
# Legend
sig_patch   = mpatches.Patch(color='#d73027', label='Significant (CI excludes 0)')
insig_patch = mpatches.Patch(color='#92c5de', label='Non-significant')
fig.legend(handles=[sig_patch, insig_patch],
           loc='lower center', ncol=2, fontsize=9,
           bbox_to_anchor=(0.5, -0.08))
 
fig.suptitle(
    'Bootstrap 95% CI — Pairwise Mean MAE Differences by Income Group',
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig('results_visual/figures/sq2_fig4_bootstrap_ci.pdf', bbox_inches='tight')
print('Saved: sq2_fig4_bootstrap_ci')
 
 
# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5: Mean signed residual — dot plot across models and income groups
# Compact summary of over/underprediction direction
# ─────────────────────────────────────────────────────────────────────────────
mean_resid = (
    results_df.groupby(['model', 'income_group'], observed=True)['residual']
    .mean()
    .reset_index()
)
mean_resid['income_group'] = pd.Categorical(
    mean_resid['income_group'], categories=group_order, ordered=True
)
 
fig, ax = plt.subplots(figsize=(10, 5))
 
for j, model_name in enumerate(model_names):
    subset = mean_resid[mean_resid['model'] == model_name].sort_values('income_group')
    ax.plot(
        subset['income_group'].astype(str),
        subset['residual'],
        marker='o', linewidth=1.5,
        markersize=7,
        label=model_name,
        color=colors[j]
    )
 
ax.axhline(0, color='black', linewidth=0.9, linestyle='--', alpha=0.6)
ax.fill_between(group_order, 0, 20, alpha=0.04, color='green', label='Underprediction zone')
ax.fill_between(group_order, -20, 0, alpha=0.04, color='red', label='Overprediction zone')
ax.set_ylabel('Mean Signed Residual (GCI points)', fontsize=14)
ax.set_xlabel('Income Group', fontsize=14)
ax.set_xticklabels(group_order, fontsize=12)
ax.set_title(
    'Direction of Misprediction by Income Group and Model',
    fontsize=14, fontweight='bold')
ax.legend(fontsize=12, framealpha=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('results_visual/figures/sq2_fig5_signed_residual_dot.pdf', bbox_inches='tight')
print('Saved: sq2_fig5_signed_residual_dot')
 

#=====================================================================
# SQ2: gci distribution across income levels - discussion
#=====================================================================

results_df = pd.read_csv('results/sq2_results_df.csv')

country_gci = (
    results_df[results_df['model'] == 'Linear Regression']
    [['Country', 'income_group', 'y_true']]
    .sort_values(['income_group', 'y_true'])
)
print(country_gci.to_string())


print(results_df.columns.tolist())

palette = {
    'Low income':          '#1f77b4',
    'Lower middle income': '#ff7f0e',
    'Upper middle income': '#2ca02c',
    'High income':         '#d62728'
}

fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(
    data=results_df.drop_duplicates('Country'),
    x='income_group',
    y='y_true',
    order=group_order,
    palette=palette,
    ax=ax
)
ax.set_xticklabels(['L', 'LM', 'UM', 'H'])
plt.ylabel('GCI Score')
plt.xlabel('Income Group')
plt.title('GCI Score Distribution by Income Group')
plt.tight_layout()

plt.savefig('results_visual/figures/gci_dist_income.pdf',
            dpi=300, bbox_inches='tight', format='pdf')
plt.close()
print('Saved: results_visual/figures/gci_dist_income.pdf')
