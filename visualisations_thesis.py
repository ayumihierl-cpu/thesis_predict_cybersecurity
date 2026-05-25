# visualisations 

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from openpyxl import Workbook
from openpyxl.styles import (
    Font, PatternFill, Alignment, Border, Side
)
from openpyxl.utils import get_column_letter
from scipy.stats import gaussian_kde, skew, kurtosis
import seaborn as sns


os.makedirs('results/figures', exist_ok=True)
os.makedirs('results_visual/figures', exist_ok=True)

data = joblib.load('results/results_no_aug.joblib')

MODEL_NAMES = ['Linear Regression', 'GAM', 'SVR', 'XGBoost']
MODEL_COLORS = ['#2C7BB6', '#1A9641', '#F46D43', '#7B2D8B']

# ─────────────────────────────────────────────────────────────────────────────
# Build data blocks
# ─────────────────────────────────────────────────────────────────────────────
def build_train(data):
    rows = []
    for i, name in enumerate(MODEL_NAMES):
        rows.append([name,
                     round(np.mean(data['train_r2'][i]),  3),
                     round(np.mean(data['train_mae'][i]), 3)])
    return rows

def build_val(data):
    rows = []
    for i, name in enumerate(MODEL_NAMES):
        rows.append([name,
                     round(np.mean(data['cv_r2'][i]),  3),
                     round(np.mean(data['cv_mae'][i]), 3)])
    return rows

def build_test(data):
    rows = []
    for i, name in enumerate(MODEL_NAMES):
        rows.append([name,
                     round(data['test_r2'][i],   3),
                     round(data['test_mae'][i],  3),
                     round(data['test_rmse'][i], 3)])
    return rows

train_headers = ['Model', 'R²', 'MAE']
val_headers   = ['Model', 'R²', 'MAE']
test_headers  = ['Model', 'R²', 'MAE', 'RMSE']

train_data = build_train(data)
val_data   = build_val(data)
test_data  = build_test(data)

# ─────────────────────────────────────────────────────────────────────────────
# Style constants
# ─────────────────────────────────────────────────────────────────────────────
FONT_NAME   = 'Arial'
DARK_BLUE   = '1F3864'
MID_BLUE    = 'BDD7EE'
LIGHT_BLUE  = 'DEEAF1'
WHITE       = 'FFFFFF'
STRIPE      = 'F2F7FB'

thin  = Side(style='thin',   color='BFBFBF')
thick = Side(style='medium', color=DARK_BLUE)

def section_label_style():
    return Font(name=FONT_NAME, bold=True, size=10, color=WHITE)

def cell_font(bold=False):
    return Font(name=FONT_NAME, bold=bold, size=9)

def center():
    return Alignment(horizontal='center', vertical='center', wrap_text=True)

def right():
    return Alignment(horizontal='right', vertical='center')

def left():
    return Alignment(horizontal='left', vertical='center')

# ─────────────────────────────────────────────────────────────────────────────
# Write helper
# ─────────────────────────────────────────────────────────────────────────────
def write_block(ws, start_row, section_title, headers, data):
    n_cols  = len(headers)
    end_col = n_cols

    # Section label row
    ws.merge_cells(start_row=start_row, start_column=1,
                   end_row=start_row,   end_column=end_col)
    cell = ws.cell(row=start_row, column=1, value=section_title)
    cell.font      = section_label_style()
    cell.fill      = PatternFill('solid', fgColor=DARK_BLUE)
    cell.alignment = left()
    cell.border    = Border(left=thick, right=thick,
                            top=thick,  bottom=thick)
    ws.row_dimensions[start_row].height = 18

    # Column header row
    hr = start_row + 1
    for ci, h in enumerate(headers, start=1):
        cell = ws.cell(row=hr, column=ci, value=h)
        cell.font      = Font(name=FONT_NAME, bold=True, size=9,
                              color=DARK_BLUE)
        cell.fill      = PatternFill('solid', fgColor=MID_BLUE
                                     if ci == 1 else LIGHT_BLUE)
        cell.alignment = center() if ci > 1 else left()
        cell.border    = Border(left=thin, right=thin,
                                top=thick, bottom=thick)
    ws.row_dimensions[hr].height = 28

    # Data rows
    for ri, row in enumerate(data):
        r         = hr + 1 + ri
        stripe    = (ri % 2 == 1)
        base_fill = PatternFill('solid', fgColor=STRIPE if stripe else WHITE)
        is_last   = ri == len(data) - 1
        bot_side  = thick if is_last else thin

        for ci, val in enumerate(row, start=1):
            cell       = ws.cell(row=r, column=ci, value=val)
            cell.fill  = base_fill
            cell.font  = cell_font(bold=(ci == 1))
            cell.alignment = left() if ci == 1 else right()
            cell.border    = Border(left=thin, right=thin,
                                    top=thin, bottom=bot_side)
            if isinstance(val, float):
                cell.number_format = '0.000'

        ws.row_dimensions[r].height = 16

    return hr + len(data) + 1

# ─────────────────────────────────────────────────────────────────────────────
# Build workbook
# ─────────────────────────────────────────────────────────────────────────────
wb = Workbook()
ws = wb.active
ws.title = 'Model Performance'

ws.sheet_view.showGridLines = False
ws.freeze_panes = 'B1'

ws.merge_cells('A1:E1')
t = ws['A1']
t.value     = 'Model Performance Summary (No Augmentation)'
t.font      = Font(name=FONT_NAME, bold=True, size=12, color=DARK_BLUE)
t.alignment = left()
ws.row_dimensions[1].height = 22

ws.merge_cells('A2:E2')
s = ws['A2']
s.value     = 'Repeated 10-Fold CV (5 repeats, 50 folds)  |  Test set: 30% held-out'
s.font      = Font(name=FONT_NAME, size=8, italic=True, color='595959')
s.alignment = left()
ws.row_dimensions[2].height = 14

next_row = write_block(ws, 4,  'Training Performance',
                        train_headers, train_data)
next_row = write_block(ws, next_row + 1,
                        'Validation Performance  (CV mean, 50 folds)',
                        val_headers, val_data)
next_row = write_block(ws, next_row + 1,
                        'Test Performance  (held-out set)',
                        test_headers, test_data)

col_widths = {'A': 22, 'B': 10, 'C': 10, 'D': 10, 'E': 10}
for col, w in col_widths.items():
    ws.column_dimensions[col].width = w

ws.page_setup.orientation = 'portrait'
ws.page_setup.fitToPage   = True
ws.page_setup.fitToWidth  = 1
ws.page_margins.left  = 0.5
ws.page_margins.right = 0.5

wb.save('results_visual/tables/table_model_performance.xlsx')
print('Saved: results_visual/tables/table_model_performance.xlsx')
# Build the same data as a pandas DataFrame
rows = []
for i, name in enumerate(MODEL_NAMES):
    rows.append({
        'Model':       name,
        'Train R²':    round(np.mean(data['train_r2'][i]),  3),
        'Train MAE':   round(np.mean(data['train_mae'][i]), 3),
        'Val R²':      round(np.mean(data['cv_r2'][i]),     3),
        'Val MAE':     round(np.mean(data['cv_mae'][i]),    3),
        'Test R²':     round(data['test_r2'][i],            3),
        'Test MAE':    round(data['test_mae'][i],           3),
        'Test RMSE':   round(data['test_rmse'][i],          3),
    })

df = pd.DataFrame(rows)

# ── Table 1: Train and Validation ────────────────────────────────────────────
trainval_tex = r"""
\begin{table}
    \caption{Training and validation performance across models (no augmentation).
             Val metrics are means across 50 repeated 10-fold cross-validation folds.}
    \label{tab:model_performance_trainval}
    \centering
    \small
    \begin{tabular}{lrrrrrr}
        \toprule
        Model & Train R\textsuperscript{2} & Train MAE & Train RMSE & Val R\textsuperscript{2} & Val MAE & Val RMSE \\
        \midrule
"""

for i, name in enumerate(MODEL_NAMES):
    train_r2   = round(np.mean(data['train_r2'][i]),   3)
    train_mae  = round(np.mean(data['train_mae'][i]),  3)
    train_rmse = round(np.mean(data['train_rmse'][i]), 3)
    val_r2     = round(np.mean(data['cv_r2'][i]),      3)
    val_mae    = round(np.mean(data['cv_mae'][i]),     3)
    val_rmse   = round(np.mean(data['cv_rmse'][i]),    3)
    trainval_tex += f'        {name} & {train_r2:.3f} & {train_mae:.3f} & {train_rmse:.3f} & {val_r2:.3f} & {val_mae:.3f} & {val_rmse:.3f} \\\\\n'

trainval_tex += r"""        \bottomrule
    \end{tabular}
\end{table}
"""

with open('results_visual/tables/table_performance_trainval.tex', 'w') as f:
    f.write(trainval_tex)
print('Saved: results2/table_performance_trainval.tex')


# ── Table 2: Test set ─────────────────────────────────────────────────────────
test_tex = r"""
\begin{table}
    \caption{Test set performance across models (no augmentation).
             Results are evaluated on the held-out 30\% test set.}
    \label{tab:model_performance_test}
    \centering
    \small
    \begin{tabular}{lrrr}
        \toprule
        Model & Test R\textsuperscript{2} & Test MAE & Test RMSE \\
        \midrule
"""

for i, name in enumerate(MODEL_NAMES):
    test_r2   = round(data['test_r2'][i],   3)
    test_mae  = round(data['test_mae'][i],  3)
    test_rmse = round(data['test_rmse'][i], 3)
    test_tex += f'        {name} & {test_r2:.3f} & {test_mae:.3f} & {test_rmse:.3f} \\\\\n'

test_tex += r"""        \bottomrule
    \end{tabular}
\end{table}
"""

with open('results_visual/tables/table_performance_test.tex', 'w') as f:
    f.write(test_tex)
print('Saved: results_visual/table_performance_test.tex')
 

# visualisations to compare augmenttaion and non augmentation

no_aug = joblib.load('results/results_no_aug.joblib')
aug    = joblib.load('results/results_aug.joblib')


plt.rcParams.update({
    'font.family':       'serif',
    'font.size':         11,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.3,
    'grid.linestyle':    '--',
    'figure.dpi':        150
})

x     = np.arange(len(MODEL_NAMES))
width = 0.35

# ── Figure 2: Test MAE — original vs augmented ───────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

bars_no  = ax.bar(x - width/2,
                  [no_aug['test_mae'][i] for i in range(len(MODEL_NAMES))],
                  width, label='Original',
                  color=MODEL_COLORS, alpha=0.55,
                  edgecolor='black', linewidth=0.8)
bars_aug = ax.bar(x + width/2,
                  [aug['test_mae'][i] for i in range(len(MODEL_NAMES))],
                  width, label='Augmented',
                  color=MODEL_COLORS, alpha=0.95,
                  edgecolor='black', linewidth=0.8)

for bar in bars_no:
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.1,
            f'{bar.get_height():.2f}',
            ha='center', va='bottom', fontsize=8)
for bar in bars_aug:
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.1,
            f'{bar.get_height():.2f}',
            ha='center', va='bottom', fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(MODEL_NAMES, rotation=15, ha='right')
ax.set_ylabel('Test MAE (GCI points)')
ax.set_title('Test Set MAE: Original vs Augmented')
orig_patch = mpatches.Patch(facecolor='grey', alpha=0.55,
                             edgecolor='black', label='Original')
aug_patch  = mpatches.Patch(facecolor='grey', alpha=0.95,
                             edgecolor='black', label='Augmented')
ax.legend(handles=[orig_patch, aug_patch], fontsize=9)

plt.tight_layout()
plt.savefig('results_visual/figures/fig_aug_test_mae.pdf',
            bbox_inches='tight', format='pdf')
print('Saved: fig_aug_test_mae')


#=====================================================================
# target statistics and distribution
#=====================================================================

df  = pd.read_csv('dataset_2020.csv')
gci = df['GCI_2020'].dropna()

stats = {
    'N':           len(gci),
    'Mean':        round(gci.mean(),   2),
    'Median':      round(gci.median(), 2),
    'Std Dev':     round(gci.std(),    2),
    'Min':         round(gci.min(),    2),
    'Max':         round(gci.max(),    2),
    'Q1 (25\%)':   round(gci.quantile(0.25), 2),
    'Q3 (75\%)':   round(gci.quantile(0.75), 2),
    'IQR':         round(gci.quantile(0.75) - gci.quantile(0.25), 2),
    'Skewness':    round(skew(gci),     2),
    'Kurtosis':    round(kurtosis(gci), 2),
}

# ── Figure: target histogram  ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))

ax.hist(gci, bins=25, color='#4575b4', edgecolor='white',
        alpha=0.75)

kde     = gaussian_kde(gci, bw_method=0.3)
x_range = np.linspace(gci.min() - 2, gci.max() + 2, 300)

ax.axvline(gci.mean(),   color='#1a9641', linewidth=1.5,
           linestyle='--', label=f'Mean ({gci.mean():.1f})')


ax.set_xlabel('GCI Score (0–100)', fontsize=13)
ax.set_ylabel('Count',           fontsize=13)
ax.set_title('Distribution of GCI 2020\n(n = 194 countries)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.set_xlim(gci.min() - 3, gci.max() + 3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('results_visual/figures/gci_distribution.pdf',
            dpi=300, bbox_inches='tight', format='pdf')
plt.close()
print('Saved: gci_distribution.pdf')

# ── Table: LaTeX only ─────────────────────────────────────────────────────────
latex = r"""\begin{table}
    \caption{Descriptive statistics of the GCI 2020 target variable.}
    \label{tab:gci_stats}
    \centering
    \small
    \begin{tabular}{lr}
        \toprule
        Statistic & Value \\
        \midrule
"""

for k, v in stats.items():
    latex += f'        {k} & {v} \\\\\n'

latex += r"""        \bottomrule
    \end{tabular}
\end{table}
"""

with open('results_visual/tables/table_gci_stats.tex', 'w') as f:
    f.write(latex)
print('Saved: results_visual/table_gci_stats.tex')

#=====================================================================
# heatmap
#=====================================================================

drop_cols = ['Country', 'c_code', 'gdp_growth']
df_num = df.drop(columns=drop_cols)

# Readable column name mapping
rename_map = {
    'GCI_2020':                       'GCI',
    'HCI_2020':                       'HCI',
    'gdp_pc_ppp':                     'GDP pc (PPP)',
    'gdp_pc_current_usd':             'GDP pc (USD)',
    'gni_per_capita_ppp':             'GNI pc (PPP)',
    'inflation_cpi':                  'Inflation',
    'govt_expenditure_pct_gdp':       'Govt Exp.',
    'gross_capital_formation_pct_gdp':'Gross Cap.',
    'trade_openness':                 'Trade Open.',
    'industry_share_gdp':             'Industry Share',
    'comp_edu_duration':              'Comp. Edu.',
    'primary_enorllment_gross':       'Primary Enr.',
    'secondary_enrollment_gross':     'Secondary Enr.',
    'tertiary_enrollment_gross':      'Tertiary Enr.',
    'govt_expenditure_edu_pct':       'Edu. Exp.',
    'unemployment_ilo':               'Unemploy.',
    'youth_literacy':                 'Youth Lit.',
}

df_num = df_num.rename(columns=rename_map)

# Compute correlation matrix
corr = df_num.corr(method='pearson')


# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 11))

sns.heatmap(
    corr,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    center=0,
    vmin=-1, vmax=1,
    linewidths=0.4,
    annot_kws={'size': 12},
    ax=ax,
    cbar_kws={'shrink': 0.8, 'label': 'Pearson r'}
)

# Highlight GCI row/column border to make it stand out
ax.add_patch(plt.Rectangle(
    (0, 0), len(corr), 1,
    fill=False, edgecolor='black', lw=2, clip_on=False
))

ax.add_patch(plt.Rectangle(
    (0, 0), 1, len(corr),
    fill=False, edgecolor='black', lw=2, clip_on=False
))

ax.collections[0].colorbar.ax.tick_params(labelsize=11)   # colorbar tick numbers
ax.collections[0].colorbar.set_label('Pearson r', fontsize=14)  # colorbar title


ax.set_title(
    'Correlation Matrix — All Variables',
    fontsize=14, fontweight='bold', pad=15
)

# Italicise dropped variable labels
dropped = [
    'GDP pc (USD)', 'GNI pc (PPP)', 'Youth Lit.'
]
for label in ax.get_xticklabels():
    if label.get_text() in dropped:
        label.set_style('italic')
        label.set_color('#d73027')
for label in ax.get_yticklabels():
    if label.get_text() in dropped:
        label.set_style('italic')
        label.set_color('#d73027')

plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(rotation=0,  fontsize=12)
plt.tight_layout()

plt.savefig('results_visual/figures/correlation_heatmap.pdf',
            dpi=300, bbox_inches='tight', format='pdf')
plt.close()
print('Saved: correlation_heatmap_all_vars.pdf')

#=====================================================================
# visualise augmentation
#=====================================================================
# visualise_augmentation_distribution.py
# Shows GCI distribution before and after Gaussian Copula augmentation

from preprocessing import X_trainval, y_trainval
from augmentation import gaussian_copula_augment, plot_augmentation_check
from config import RANDOM_STATE

# Generate augmented data
X_aug, y_aug = gaussian_copula_augment(
    X_trainval, y_trainval,
    n_synthetic=len(X_trainval),  # doubles the dataset
    random_state=RANDOM_STATE
)

# Plot and save
plot_augmentation_check(
    y_trainval, y_aug
)


#=====================================================================
# hyperparameter table
#=====================================================================

# ── Import grids directly from models.py ─────────────────────────────────────
from models import PARAM_GRID_GAM, PARAM_GRID_SVR, PARAM_GRID_XGB_COMB

# ── Build hyperparameter info from actual grids ───────────────────────────────
def format_values(values):
    return ', '.join(str(v) for v in values)

hyperparameter_info = {
    'Linear Regression': [
        ('No hyperparameters tuned', '—'),
    ],
    'GAM': [
        (param.replace('regressor__', ''), format_values(values))
        for param, values in PARAM_GRID_GAM.items()
    ],
    'SVR': [
        (param.replace('regressor__', ''), format_values(values))
        for param, values in PARAM_GRID_SVR.items()
    ],
    'XGBoost': [
        (param.replace('regressor__', ''), format_values(values))
        for param, values in PARAM_GRID_XGB_COMB.items()
    ],
}

# ── LaTeX output ──────────────────────────────────────────────────────────────
latex = r"""\begin{table}
    \caption{Hyperparameter search spaces used in GridSearchCV. 
             Selection was based on inner 10-fold cross-validation 
             minimising mean absolute error (MAE). Linear Regression 
             has no tunable hyperparameters.}
    \label{tab:hyperparameters}
    \centering
    \small
    \begin{tabular}{lll}
        \toprule
        Model & Parameter & Values Tested \\
        \midrule
"""

for model, params in hyperparameter_info.items():
    latex += f'        \\multicolumn{{3}}{{l}}{{\\textit{{{model}}}}} \\\\\n'
    for param, values in params:
        safe_param = param.replace('_', '\\_')
        latex += f'        & \\texttt{{{safe_param}}} & {values} \\\\\n'
    latex += '        \\midrule\n'

# Remove last midrule and close
latex = latex.rstrip()
if latex.endswith('\\midrule'):
    latex = latex[:-len('\\midrule')]

latex += r"""        \bottomrule
    \end{tabular}
\end{table}
"""

with open('results_visual/tables/table_hyperparameters.tex', 'w') as f:
    f.write(latex)
print('Saved: results_visual/tables/table_hyperparameters.tex')

#=====================================================================
# test MAE comparison across models
#=====================================================================

data = joblib.load('results/results_no_aug.joblib')

test_mae = data['test_mae']

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(
    MODEL_NAMES, test_mae,
    color=MODEL_COLORS,
    edgecolor='white',
    linewidth=0.5,
    width=0.5
)

# Value labels on top of bars
for bar, val in zip(bars, test_mae):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.15,
        f'{val:.2f}',
        ha='center', va='bottom', fontsize=10
    )

# Baseline reference line (Linear Regression MAE)
ax.axhline(test_mae[0], color='grey', linewidth=1,
           linestyle='--', alpha=0.7,
           label=f'LR baseline ({test_mae[0]:.2f})')

ax.set_ylabel('Test MAE (GCI points)', fontsize=14)
ax.set_title('Test Set MAE by Model\n(Original unaugmented data)',
             fontsize=14, fontweight='bold')
ax.set_ylim(0, max(test_mae) * 1.2)
ax.legend(fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('results_visual/figures/test_mae_by_model.pdf',
            dpi=300, bbox_inches='tight', format='pdf')
plt.close()
print('Saved: test_mae_by_model.png + pdf')

#=====================================================================
# Train vs validation vs test MAE chart 
#=====================================================================

train_mae = [np.mean(data['train_mae'][i]) for i in range(len(MODEL_NAMES))]
val_mae   = [np.mean(data['cv_mae'][i])    for i in range(len(MODEL_NAMES))]
test_mae  = data['test_mae']

x     = np.arange(len(MODEL_NAMES))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 5))

bars_train = ax.bar(x - width, train_mae, width,
                    label='Train MAE',
                    color=MODEL_COLORS, alpha=0.4,
                    edgecolor='white', linewidth=0.5)
bars_val   = ax.bar(x,         val_mae,   width,
                    label='Validation MAE (CV mean)',
                    color=MODEL_COLORS, alpha=0.7,
                    edgecolor='white', linewidth=0.5)
bars_test  = ax.bar(x + width, test_mae,  width,
                    label='Test MAE',
                    color=MODEL_COLORS, alpha=1.0,
                    edgecolor='white', linewidth=0.5)

# Value labels
for bars in [bars_train, bars_val, bars_test]:
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            f'{bar.get_height():.1f}',
            ha='center', va='bottom', fontsize=7.5
        )

ax.set_xticks(x)
ax.set_xticklabels(MODEL_NAMES, fontsize=14)
ax.set_ylabel('MAE (GCI points)', fontsize=14)
ax.set_title('Train vs Validation vs Test MAE',
             fontsize=14, fontweight='bold')
ax.set_ylim(0, max(test_mae) * 1.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Custom legend — shade explains train/val/test, color explains model
import matplotlib.patches as mpatches
shade_patches = [
    mpatches.Patch(facecolor='grey', alpha=0.4, label='Train'),
    mpatches.Patch(facecolor='grey', alpha=0.7, label='Validation (CV mean)'),
    mpatches.Patch(facecolor='grey', alpha=1.0, label='Test'),
]
color_patches = [
    mpatches.Patch(facecolor=c, label=n)
    for c, n in zip(MODEL_COLORS, MODEL_NAMES)
]
leg1 = ax.legend(handles=shade_patches, fontsize=10,
                 loc='upper left', title='Set', title_fontsize=10)
ax.add_artist(leg1)
ax.legend(handles=color_patches, fontsize=10,
          loc='upper right', title='Model', title_fontsize=10)

plt.tight_layout()
plt.savefig('results_visual/figures/train_val_test_mae.pdf',
            dpi=300, bbox_inches='tight', format='pdf')
plt.close()
print('Saved: train_val_test_mae.png + pdf')


#=====================================================================
# augmentation main 
#=====================================================================

no_aug = joblib.load('results/results_no_aug.joblib')
aug    = joblib.load('results/results_aug.joblib')

MODEL_NAMES = ['Linear Regression', 'GAM', 'SVR', 'XGBoost']

latex = r"""\begin{table}
    \caption{Effect of Gaussian Copula augmentation on test set performance.
             $\Delta$ = augmented $-$ original. Negative $\Delta$ MAE and 
             positive $\Delta$ R\textsuperscript{2} indicate improvement.}
    \label{tab:augmentation_results}
    \centering
    \small
    \begin{tabular}{lrrrrrr}
        \toprule
        & \multicolumn{2}{c}{R\textsuperscript{2}} 
        & \multicolumn{2}{c}{MAE} 
        & \multicolumn{2}{c}{RMSE} \\
        \cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}
        Model & Orig & Aug & Orig & Aug & Orig & Aug \\
        \midrule
"""

for i, name in enumerate(MODEL_NAMES):
    r2_no    = no_aug['test_r2'][i]
    r2_aug   = aug['test_r2'][i]
    mae_no   = no_aug['test_mae'][i]
    mae_aug  = aug['test_mae'][i]
    rmse_no  = no_aug['test_rmse'][i]
    rmse_aug = aug['test_rmse'][i]

    latex += (f'        {name} & '
              f'{r2_no:.3f} & {r2_aug:.3f} & '
              f'{mae_no:.2f} & {mae_aug:.2f} & '
              f'{rmse_no:.2f} & {rmse_aug:.2f} \\\\\n')

latex += r"""        \bottomrule
    \end{tabular}
\end{table}
"""

with open('results_visual/tables/table_augmentation_results.tex', 'w') as f:
    f.write(latex)
print('Saved: results_visual/tables/table_augmentation_results.tex')

# augmentation effect on test MAE bar chart

test_mae_no  = [no_aug['test_mae'][i] for i in range(len(MODEL_NAMES))]
test_mae_aug = [aug['test_mae'][i]    for i in range(len(MODEL_NAMES))]

x     = np.arange(len(MODEL_NAMES))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))

bars_no  = ax.bar(x - width/2, test_mae_no,  width,
                  color=MODEL_COLORS, alpha=0.5,
                  edgecolor='white', linewidth=0.5)
bars_aug = ax.bar(x + width/2, test_mae_aug, width,
                  color=MODEL_COLORS, alpha=1.0,
                  edgecolor='white', linewidth=0.5)

for bar in list(bars_no) + list(bars_aug):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f'{bar.get_height():.2f}',
            ha='center', va='bottom', fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(MODEL_NAMES, fontsize=14)
ax.set_ylabel('Test MAE (GCI points)', fontsize=14)
ax.set_title('Test MAE: Original vs Gaussian Copula Augmented\n(lower is better)',
             fontsize=14, fontweight='bold')
ax.set_ylim(0, max(test_mae_no + test_mae_aug) * 1.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

orig_patch = mpatches.Patch(facecolor='grey', alpha=0.5,
                             edgecolor='none', label='Original')
aug_patch  = mpatches.Patch(facecolor='grey', alpha=1.0,
                             edgecolor='none', label='Augmented')
ax.legend(handles=[orig_patch, aug_patch], fontsize=10)

plt.tight_layout()

plt.savefig('results_visual/figures/augmentation_effect.pdf',
            dpi=300, bbox_inches='tight', format='pdf')
plt.close()
print('Saved: augmentation_mae.png + pdf')

# augmentation_gap

gap_no  = [np.mean(no_aug['cv_mae'][i]) - np.mean(no_aug['train_mae'][i])
           for i in range(len(MODEL_NAMES))]
gap_aug = [np.mean(aug['cv_mae'][i])    - np.mean(aug['train_mae'][i])
           for i in range(len(MODEL_NAMES))]

x     = np.arange(len(MODEL_NAMES))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))

bars_no  = ax.bar(x - width/2, gap_no,  width,
                  color=MODEL_COLORS, alpha=0.5,
                  edgecolor='white', linewidth=0.5)
bars_aug = ax.bar(x + width/2, gap_aug, width,
                  color=MODEL_COLORS, alpha=1.0,
                  edgecolor='white', linewidth=0.5)

for bar in list(bars_no) + list(bars_aug):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f'{bar.get_height():.2f}',
            ha='center', va='bottom', fontsize=9)

ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels(MODEL_NAMES, fontsize=14)
ax.set_ylabel('Val MAE \u2212 Train MAE (GCI points)', fontsize=14)
ax.set_title('Train\u2013Validation MAE Gap: Original vs Augmented',
             fontsize=14, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

orig_patch = mpatches.Patch(facecolor='grey', alpha=0.5,
                             edgecolor='none', label='Original')
aug_patch  = mpatches.Patch(facecolor='grey', alpha=1.0,
                             edgecolor='none', label='Augmented')
ax.legend(handles=[orig_patch, aug_patch], fontsize=10)

plt.tight_layout()
plt.savefig('results_visual/figures/augmentation_gap.pdf',
            dpi=300, bbox_inches='tight', format='pdf')
plt.close()
print('Saved: augmentation_gap.png + pdf')

