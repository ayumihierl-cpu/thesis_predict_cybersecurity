# models 

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.svm import SVR
from xgboost import XGBRegressor
from pygam import LinearGAM

from config import RANDOM_STATE
from preprocessing import preprocessor_non_tree, preprocessor_tree

# -----------------------------
# GAM wrapper (makes pyGAM sklearn-compatible)
# -----------------------------
class GAMRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_splines=10, lam=0.6):
        self.n_splines = n_splines
        self.lam = lam

    def fit(self, X, y):
        self.model_ = LinearGAM(n_splines=self.n_splines, lam=self.lam)
        self.model_.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model_.predict(X)
    

# -----------------------------
# Pipelines 
# -----------------------------
lr_pipeline = Pipeline([
    ('preprocessor', clone(preprocessor_non_tree)),
    ('regressor', LinearRegression())
])

gam_pipeline = Pipeline([
    ('preprocessor', clone(preprocessor_non_tree)),
    ('regressor', GAMRegressor(n_splines=10, lam=0.6))
])

svr_pipeline = Pipeline([
    ('preprocessor', clone(preprocessor_non_tree)),
    ('regressor', SVR())
])

xgboost_pipeline = Pipeline([
    ('preprocessor',clone(preprocessor_tree)),
    ('regressor', XGBRegressor(
        objective='reg:squarederror',
        random_state=RANDOM_STATE,
        n_jobs=1
    ))
])

# -----------------------------
# Hyperparameter grids
# -----------------------------
PARAM_GRID_GAM = {
    'regressor__n_splines': [3, 4, 5, 10, 15, 20],
    'regressor__lam': [0.01, 0.1, 1, 10, 100]
}


PARAM_GRID_SVR = {
    'regressor__kernel': ['rbf'],
    'regressor__C': [0.01, 0.1, 1, 10, 50, 100],
    'regressor__epsilon': [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10],
    'regressor__gamma': ['scale', 'auto']
}

PARAM_GRID_XGB = {
    'regressor__n_estimators':  [100, 150, 200],
    'regressor__max_depth':     [1, 2, 3, 4],
    'regressor__learning_rate': [0.01, 0.03, 0.05]
}


PARAM_GRID_XGB2 = {
    'regressor__min_child_weight': [3, 5, 8, 10],
    'regressor__gamma':            [0.3, 0.5, 1.0],
    'regressor__subsample':        [0.6, 0.8],
    'regressor__colsample_bytree': [0.6, 0.8],
    'regressor__reg_lambda':       [5, 10, 20]
}


PARAM_GRID_XGB_COMB = {
    'regressor__n_estimators': [100, 150, 200],
    'regressor__max_depth': [1, 2, 3, 4],
    'regressor__learning_rate': [0.01, 0.03, 0.05],
    'regressor__min_child_weight': [3, 5, 8, 10],
    'regressor__gamma': [0.3, 0.5, 1.0],
    'regressor__subsample': [0.6, 0.8],
    'regressor__colsample_bytree': [0.6, 0.8],
    'regressor__reg_lambda': [5, 10, 20]
}

