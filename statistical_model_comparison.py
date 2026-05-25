import joblib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from itertools import combinations
import os

os.makedirs('results', exist_ok=True)

data = joblib.load('results/results_no_aug.joblib')

MODEL_NAMES = ['Linear Regression', 'GAM', 'SVR', 'XGBoost']
cv_mae      = [data['cv_mae'][i] for i in range(len(MODEL_NAMES))]

# additional significance analysis for main modeling

for i, name in enumerate(MODEL_NAMES):
    print(f'{name}: {np.mean(cv_mae[i]):.4f}')

def format_p(p):
    if p < 0.001:
        return '<0.001'
    return f'{p:.3f}'

def sig_stars(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    return 'NS'

print(f'CV folds per model: {len(cv_mae[0])}')
print('\nMean CV MAE per model:')
for name, mae in zip(MODEL_NAMES, cv_mae):
    print(f'  {name:20s}: {np.mean(mae):.3f} +/- {np.std(mae):.3f}')

print('\n' + '='*60)
print('Pairwise Wilcoxon Signed-Rank Test — CV MAE (50 folds)')
print('='*60)

rows = []

for (i, m1), (j, m2) in combinations(enumerate(MODEL_NAMES), 2):
    mae1 = np.array(cv_mae[i])
    mae2 = np.array(cv_mae[j])

    try:
        w_stat, w_p = wilcoxon(mae1, mae2, alternative='two-sided')
    except ValueError:
        w_stat, w_p = np.nan, np.nan

    sig    = sig_stars(w_p)
    winner = m1 if np.mean(mae1) < np.mean(mae2) else m2

    print(f'\n{m1} vs {m2}:')
    print(f'  Mean MAE : {np.mean(mae1):.3f} vs {np.mean(mae2):.3f} -> {winner} lower')
    print(f'  W = {w_stat:.1f}, p = {format_p(w_p)} -- {sig}')

    rows.append({
        'Model 1':      m1,
        'Model 2':      m2,
        'Mean MAE (M1)':round(np.mean(mae1), 3),
        'Mean MAE (M2)':round(np.mean(mae2), 3),
        'Delta MAE':    round(np.mean(mae1) - np.mean(mae2), 3),
        'W statistic':  round(w_stat, 1) if not np.isnan(w_stat) else '—',
        'p (Wilcoxon)': format_p(w_p) if not np.isnan(w_p) else '—',
        'Significance': sig,
        'Lower MAE':    winner,
    })

results_df = pd.DataFrame(rows)

print('\n' + '='*60)
print('Summary Table')
print('='*60)
print(results_df.to_string(index=False))

results_df.to_csv('results_visual/tables/table_model_comparison_stats.csv',
                  index=False, sep=';')
print('\nSaved: results_visual/tables/table_model_comparison_stats.csv')

# ── LaTeX ─────────────────────────────────────────────────────────────────────
latex = r"""\begin{table}
    \caption{Pairwise Wilcoxon signed-rank tests comparing cross-validation
             MAE distributions across 50 repeated folds.
             $\Delta$ MAE = Model 1 $-$ Model 2; negative = Model 2 lower.}
    \label{tab:model_comparison_stats}
    \centering
    \small
    \begin{tabular}{llrr}
        \toprule
        Model 1 & Model 2 & $\Delta$ MAE & $p$ \\
        \midrule
"""

for _, row in results_df.iterrows():
    delta  = row['Delta MAE']
    sign   = '+' if delta >= 0 else ''
    stars  = row['Significance']
    p_val  = row['p (Wilcoxon)']
    if stars == 'NS':
        p_cell = f'${p_val}$'
    else:
        p_cell = f'${p_val}^{{\\text{{{stars}}}}}$'
    latex += (f'        {row["Model 1"]} & {row["Model 2"]} & '
              f'${sign}{delta}$ & '
              f'{p_cell} \\\\\n')

latex += r"""        \bottomrule
        \multicolumn{4}{l}{\footnotesize $^{***}p<0.001$,\; $^{**}p<0.01$,\; $^{*}p<0.05$,\; NS\;$p\geq0.05$}
    \end{tabular}
\end{table}
"""

with open('results_visual/tables/table_model_comparison_stats.tex', 'w') as f:
    f.write(latex)
print('Saved: results_visual/tables/table_model_comparison_stats.tex')