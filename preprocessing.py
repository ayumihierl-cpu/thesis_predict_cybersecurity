# preprocessing 

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.model_selection import train_test_split

from config import (DATA_PATH, TARGET_COL, DROP_COLS,
                    MAX_MISSING_ROWS, TEST_SIZE, RANDOM_STATE,
                    SKEW_THRESHOLD)


# -----------------------------
# Load and Clean 
# -----------------------------
df = pd.read_csv(DATA_PATH)
df = df.drop(columns=DROP_COLS)
df = df[df.isna().sum(axis=1) <= MAX_MISSING_ROWS]

df_num = df.drop(columns=['Country', 'c_code'])

print(len(df))

# -----------------------------
# Feature and Target 
# -----------------------------
X = df_num.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# -----------------------------
# Train Test SPlit (stratified)
# -----------------------------

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

print(f'Trainval GCI — mean: {y_trainval.mean():.1f}  std: {y_trainval.std():.1f}')
print(f'Test GCI     — mean: {y_test.mean():.1f}  std: {y_test.std():.1f}')
print(f'\nTrainval size: {len(X_trainval)}')
print(f'Test size    : {len(X_test)}')

print(len(X_trainval))
# -----------------------------
# Skewness
# -----------------------------
skewed_feats = X_trainval.skew().sort_values(ascending=False)
high_skew = skewed_feats[abs(skewed_feats) > SKEW_THRESHOLD].index.tolist()

skewed_features = [col for col in X.columns if col in high_skew]
non_skewed_features = [col for col in X.columns if col not in high_skew]

print('High-skew features:', skewed_features)
print('Non-skew features:', non_skewed_features)


# -----------------------------
# Tranformers 
# -----------------------------
skew_transformer = Pipeline([
    ('imputer', IterativeImputer(random_state=RANDOM_STATE,
                                 initial_strategy='median',
                                 max_iter=20)),
    ('scaler', PowerTransformer(method='yeo-johnson',
                                standardize=True))
])

standard_transformer = Pipeline([
    ('imputer', IterativeImputer(random_state=RANDOM_STATE,
                                 initial_strategy='median',
                                 max_iter=20)),
    ('scaler', StandardScaler())
])

# for non tree models (LR, GAM, SVR)
preprocessor_non_tree = ColumnTransformer(
    transformers=[
        ('skew', skew_transformer, skewed_features),
        ('num', standard_transformer, non_skewed_features)
    ],
    remainder='passthrough'
)

# for tree-based models (XGBoost)
preprocessor_tree = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), X.columns.tolist())
    ],
    remainder='passthrough'
)
