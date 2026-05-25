# config 

import numpy as np
from sklearn.model_selection import RepeatedKFold, KFold

# ----------------------
# Reproducibility
# ----------------------

RANDOM_STATE = 80
np.random.seed(RANDOM_STATE)


# -----------------------
# Data
# ------------------------

DATA_PATH = 'dataset_2020.csv'
TARGET_COL = 'GCI_2020'
DROP_COLS = [
    'youth_literacy',
    'gdp_pc_current_usd',
    'gni_per_capita_ppp',
    'gdp_growth'
]

MAX_MISSING_ROWS = 5 # drop rows with more missing values than 5
TEST_SIZE = 0.2
SKEW_THRESHOLD = 1.0 # if absolute skewness is above this, used skewed pipeline


# ------------------------
# Cross-validation
# ------------------------

RKF = RepeatedKFold(n_splits=10, n_repeats=5, random_state=RANDOM_STATE)
INNER_CV = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

# -------------------------
# Scoring
# -------------------------
SCORING = ['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error']


# -------------------------
# Augmentation (Gaussian Copula)
# Patki, N., Wedge, R., & Veeramachaneni, K. (2016).
# The Synthetic Data Vault. IEEE DSAA.
#
# GC_N_SYNTHETIC: number of synthetic observations to generate.
# Set to None to match the original training set size (doubles the data).
# Set to an integer to generate a specific number of synthetic observations.
# -------------------------
GC_N_SYNTHETIC = None   # None = match original size (doubles dataset)


