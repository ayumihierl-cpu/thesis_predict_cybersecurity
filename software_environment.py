# software_environment

import platform
import sys
import importlib

libraries = [
    ('Python',       'sys'),
    ('numpy',        'numpy'),
    ('pandas',       'pandas'),
    ('scikit-learn', 'sklearn'),
    ('xgboost',      'xgboost'),
    ('shap',         'shap'),
    ('matplotlib',   'matplotlib'),
    ('seaborn',      'seaborn'),
    ('scipy',        'scipy'),
    ('pygam',        'pygam'),
    ('joblib',       'joblib'),
    ('statsmodels',  'statsmodels'),
]

latex = r"""\begin{table}
    \caption{Software environment and library versions used in the analysis.}
    \label{tab:software}
    \centering
    \small
    \begin{tabular}{lr}
        \toprule
        Component & Version \\
        \midrule
        \multicolumn{2}{l}{\textit{Environment}} \\
"""

latex += f'        & Python & {sys.version.split()[0]} \\\\\n'
latex += f'        & Operating System & {platform.system()} {platform.release()} \\\\\n'
latex += '        \\midrule\n'
latex += '        \\multicolumn{2}{l}{\\textit{Libraries}} \\\\\n'

for display_name, import_name in libraries[1:]:
    try:
        mod     = importlib.import_module(import_name)
        version = getattr(mod, '__version__', 'unknown')
    except ImportError:
        version = 'not installed'
    latex += f'        & {display_name} & {version} \\\\\n'

latex += r"""        \bottomrule
    \end{tabular}
\end{table}
"""

with open('results/table_software_environment.tex', 'w') as f:
    f.write(latex)
print('Saved: results/table_software_environment.tex')
