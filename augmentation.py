# Gaussian Copula augmentation for small tabular datasets.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copulas.multivariate import GaussianMultivariate
from sklearn.impute import SimpleImputer


# Features that cannot take negative values — clipped after sampling
NON_NEGATIVE_COLS = [
    'gdp_pc_ppp',
    'unemployment_ilo',
    'tertiary_enrollment_gross',
    'primary_enorllment_gross',
    'govt_expenditure_edu_pct',
    'gross_capital_formation_pct_gdp',
    'comp_edu_duration',
    'trade_openness',
    'industry_share_gdp'
]


def gaussian_copula_augment(X_trainval, y_trainval,
                             n_synthetic=None,
                             random_state=80):
    np.random.seed(random_state)

    if n_synthetic is None:
        n_synthetic = len(X_trainval)

    target_col = 'GCI_2020'
    df_train = X_trainval.copy()
    df_train[target_col] = y_trainval.values

    print('\nApplying Gaussian Copula augmentation...')
    print(f'  Original size    : {len(df_train)}')
    print(f'  Synthetic to add : {n_synthetic}')

    # impute missing values with median before fitting the copula
    # the copula requires complete data — imputation is only applied to
    # the fitting step, not to the original training data which retains
    # its missingness for the main preprocessing pipeline
    imputer = SimpleImputer(strategy='median')
    df_imputed = pd.DataFrame(
        imputer.fit_transform(df_train),
        columns=df_train.columns
    )

    # fit copula on full training set (features + target jointly)
    model = GaussianMultivariate()
    model.fit(df_imputed)

    # sample synthetic observations
    synthetic = model.sample(n_synthetic)

    # clip GCI to valid 0-100 range
    synthetic[target_col] = synthetic[target_col].clip(0, 100)

    # clip non-negative features
    for col in NON_NEGATIVE_COLS:
        if col in synthetic.columns:
            synthetic[col] = synthetic[col].clip(lower=0)

    # align column order to original
    synthetic = synthetic[df_train.columns]

    # combine original and synthetic
    X_synthetic = synthetic.drop(columns=[target_col])
    y_synthetic = synthetic[target_col]

    X_aug = pd.concat(
        [X_trainval.reset_index(drop=True),
         X_synthetic.reset_index(drop=True)],
        ignore_index=True
    )
    y_aug = pd.concat(
        [y_trainval.reset_index(drop=True),
         y_synthetic.reset_index(drop=True)],
        ignore_index=True
    )

    print(f'  Augmented size   : {len(X_aug)}')
    print(f'  GCI range (synthetic): '
          f'{y_synthetic.min():.1f} - {y_synthetic.max():.1f}')

    return X_aug, y_aug


def plot_augmentation_check(y_trainval, y_aug):
    n_synthetic = len(y_aug) - len(y_trainval)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, data, title in zip(
        axes,
        [y_trainval, y_aug],
        ['Original Training Set (141 obs)',
         f'Gaussian Copula Augmented ({len(y_aug)} obs)']
    ):
        ax.hist(data, bins=25, color='#4575b4', edgecolor='white', alpha=0.8)
        ax.set_xlabel('GCI Score', fontsize=13)
        ax.set_ylabel('Count', fontsize=13)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)

    fig.suptitle(
        'GCI Distribution: Original vs Gaussian Copula Augmented',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig('results_visual/figures/augment_distrib.pdf',
            dpi=300, bbox_inches='tight', format='pdf')
    plt.show()
    print(f'Saved: augmented_distribution')