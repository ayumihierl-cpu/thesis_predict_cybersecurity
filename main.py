# main
# Runs both augmented and non-augmented conditions in a single pass.

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.base import clone
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


from config import (RKF, INNER_CV, SCORING, RANDOM_STATE, 
                    GC_N_SYNTHETIC)
from preprocessing import (
    X_trainval, X_test, y_trainval, y_test, preprocessor_tree, df
)
from models import (
    lr_pipeline, gam_pipeline, svr_pipeline, xgboost_pipeline,
    PARAM_GRID_GAM, PARAM_GRID_SVR, PARAM_GRID_XGB, PARAM_GRID_XGB2, PARAM_GRID_XGB_COMB
)

from augmentation import gaussian_copula_augment, plot_augmentation_check

os.makedirs('results', exist_ok=True)
os.makedirs('results/figures', exist_ok=True)
os.makedirs('results_visual/figures', exist_ok=True)
os.makedirs('results_visual/tables', exist_ok=True)


# -----------------------------
# Helper: print CV summary
# -----------------------------
def print_cv_summary(name, r2, mae, rmse, train_r2=None, train_mae=None, train_rmse=None):
    print(f"\n{'='*50}")
    print(f'{name} - Train/Validation CV Results')
    print(f"{'='*50}")

    if train_r2 is not None:
        print('\n[Train vs Validation]')
        print(f' R2 : train={np.mean(train_r2):.4f} val={np.mean(r2):.4f}')
        print(f' MAE : train={np.mean(train_mae):.4f} val={np.mean(mae):.4f}')
        print(f' RMSE : train={np.mean(train_rmse):.4f} val={np.mean(rmse):.4f}')

    for metric, scores in [('R2', r2), ('MAE', mae), ('RMSE', rmse)]:
        print(f'\n{metric}:')
        print(f' Mean : {np.mean(scores):.4f} ± {np.std(scores):.4f}')
        print(f' Median : {np.median(scores):.4f}')
        print(f' Pctiles : {np.percentile(scores, [10, 25, 75, 90])}')


def extract_scores(cv_results, return_train=False):
    r2 = cv_results['test_r2']
    mae = -cv_results['test_neg_mean_absolute_error']
    rmse = np.sqrt(-cv_results['test_neg_mean_squared_error'])
    if return_train:
        tr2 = cv_results['train_r2']
        tmae = -cv_results['train_neg_mean_absolute_error']
        trmse = np.sqrt(-cv_results['train_neg_mean_squared_error'])
        return r2, mae, rmse, tr2, tmae, trmse
    return r2, mae, rmse

def evaluate_on_test(model, name):
    y_pred = model.predict(X_test)
    r2     = r2_score(y_test, y_pred)
    mae    = mean_absolute_error(y_test, y_pred)
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f'\n  {name}')
    print(f'    R2   : {r2:.4f}')
    print(f'    MAE  : {mae:.4f}')
    print(f'    RMSE : {rmse:.4f}')
    return y_pred, r2, mae, rmse


# -----------------------------
# Prepare Gaussian Copula augmented training data
# -----------------------------

X_aug, y_aug = gaussian_copula_augment(
    X_trainval, y_trainval,
    n_synthetic=GC_N_SYNTHETIC,
    random_state=RANDOM_STATE
)
 
plot_augmentation_check(y_trainval, y_aug)
 
print(f'\nOriginal training set  : {len(X_trainval)} observations')
print(f'Augmented training set : {len(X_aug)} observations')
print(f'Net synthetic added    : {len(X_aug) - len(X_trainval)}')
print(f'Test set (never augmented): {len(X_test)} observations')



# -----------------------------
# Condition 1: no augmentation 
# hyperparameter tuned on original data 
# -----------------------------
print('\n' + '#'*60)
print('# CONDITION 1: NO AUGMENTATION')
print('#'*60)
 
# ── Linear Regression ─────────────────────────────────────────────────────────
print('\n>>> Linear Regression [no aug]')
 
lr_no = clone(lr_pipeline)
cv_lr_no = cross_validate(lr_no, X_trainval, y_trainval,
                          cv=RKF, scoring=SCORING, n_jobs=-1,
                          return_train_score=True)
r2_lr_no, mae_lr_no, rmse_lr_no, tr2_lr_no, tmae_lr_no, trmse_lr_no = \
    extract_scores(cv_lr_no, return_train=True)
print_cv_summary('Linear Regression [no aug]',
                 r2_lr_no, mae_lr_no, rmse_lr_no,
                 tr2_lr_no, tmae_lr_no, trmse_lr_no)
 
# ── GAM ───────────────────────────────────────────────────────────────────────
print('\n>>> GAM [no aug] - hyperparameter tuning')
 
gs_gam_no = GridSearchCV(clone(gam_pipeline), PARAM_GRID_GAM,
                         cv=INNER_CV, scoring='neg_mean_absolute_error',
                         n_jobs=-1, refit=True)
gs_gam_no.fit(X_trainval, y_trainval)
print(f'  Best params : {gs_gam_no.best_params_}')
print(f'  Best CV MAE : {-gs_gam_no.best_score_:.4f}')
 
best_gam_no = gs_gam_no.best_estimator_
cv_gam_no = cross_validate(best_gam_no, X_trainval, y_trainval,
                           cv=RKF, scoring=SCORING, n_jobs=-1,
                           return_train_score=True)
r2_gam_no, mae_gam_no, rmse_gam_no, tr2_gam_no, tmae_gam_no, trmse_gam_no = \
    extract_scores(cv_gam_no, return_train=True)
print_cv_summary('GAM [no aug]',
                 r2_gam_no, mae_gam_no, rmse_gam_no,
                 tr2_gam_no, tmae_gam_no, trmse_gam_no)
 
# ── SVR ───────────────────────────────────────────────────────────────────────
print('\n>>> SVR [no aug] - hyperparameter tuning')
 
gs_svr_no = GridSearchCV(clone(svr_pipeline), PARAM_GRID_SVR,
                         cv=INNER_CV, scoring='neg_mean_absolute_error',
                         n_jobs=-1, refit=True)
gs_svr_no.fit(X_trainval, y_trainval)
print(f'  Best params : {gs_svr_no.best_params_}')
print(f'  Best CV MAE : {-gs_svr_no.best_score_:.4f}')
 
best_svr_no = gs_svr_no.best_estimator_
cv_svr_no = cross_validate(best_svr_no, X_trainval, y_trainval,
                           cv=RKF, scoring=SCORING, n_jobs=-1,
                           return_train_score=True)
r2_svr_no, mae_svr_no, rmse_svr_no, tr2_svr_no, tmae_svr_no, trmse_svr_no = \
    extract_scores(cv_svr_no, return_train=True)
print_cv_summary('SVR [no aug]',
                 r2_svr_no, mae_svr_no, rmse_svr_no,
                 tr2_svr_no, tmae_svr_no, trmse_svr_no)
 

# ── XGBoost [no aug] hyperparameter tuning ────────────────────────────
print('\n>>> XGBoost [no aug] - hyperparameter tuning')

gs_xgb2_no = GridSearchCV(clone(xgboost_pipeline), PARAM_GRID_XGB_COMB,
                          cv=INNER_CV, scoring='neg_mean_absolute_error',
                          n_jobs=-1, refit=True)
gs_xgb2_no.fit(X_trainval, y_trainval)
print(f'  Best params : {gs_xgb2_no.best_params_}')
print(f'  Best CV MAE : {-gs_xgb2_no.best_score_:.4f}')

best_xgb_no = gs_xgb2_no.best_estimator_
cv_xgb_no = cross_validate(best_xgb_no, X_trainval, y_trainval,
                            cv=RKF, scoring=SCORING, n_jobs=-1,
                            return_train_score=True)
r2_xgb_no, mae_xgb_no, rmse_xgb_no, tr2_xgb_no, tmae_xgb_no, trmse_xgb_no = \
    extract_scores(cv_xgb_no, return_train=True)
print_cv_summary('XGBoost [no aug]',
                 r2_xgb_no, mae_xgb_no, rmse_xgb_no,
                 tr2_xgb_no, tmae_xgb_no, trmse_xgb_no)
 
# ── Fit final no-aug models on full trainval set ──────────────────────────────
print('\n>>> Fitting final models on full trainval set [no aug]')
 
lr_no_final = clone(lr_pipeline)
lr_no_final.fit(X_trainval, y_trainval)
best_gam_no.fit(X_trainval, y_trainval)
best_svr_no.fit(X_trainval, y_trainval)
best_xgb_no.fit(X_trainval, y_trainval)
 
# ── Test set evaluation [no aug] ──────────────────────────────────────────────
print('\n>>> Final Test Set Evaluation [no aug]')
 
pred_lr_no,  r2_lr_no_test,  mae_lr_no_test,  rmse_lr_no_test  = evaluate_on_test(lr_no_final, 'Linear Regression')
pred_gam_no, r2_gam_no_test, mae_gam_no_test, rmse_gam_no_test = evaluate_on_test(best_gam_no, 'GAM')
pred_svr_no, r2_svr_no_test, mae_svr_no_test, rmse_svr_no_test = evaluate_on_test(best_svr_no, 'SVR')
pred_xgb_no, r2_xgb_no_test, mae_xgb_no_test, rmse_xgb_no_test = evaluate_on_test(best_xgb_no, 'XGBoost')
 
# ── Save no-aug results ────────────────────────────────────────────────────────
joblib.dump({
    'cv_r2':    [r2_lr_no,   r2_gam_no,   r2_svr_no,   r2_xgb_no],
    'cv_mae':   [mae_lr_no,  mae_gam_no,  mae_svr_no,  mae_xgb_no],
    'cv_rmse':  [rmse_lr_no, rmse_gam_no, rmse_svr_no, rmse_xgb_no],
    'train_r2': [tr2_lr_no,  tr2_gam_no,  tr2_svr_no,  tr2_xgb_no],
    'train_mae':[tmae_lr_no, tmae_gam_no, tmae_svr_no, tmae_xgb_no],
    'train_rmse':[trmse_lr_no, trmse_gam_no, trmse_svr_no, trmse_xgb_no],
    'test_r2':  [r2_lr_no_test,   r2_gam_no_test,   r2_svr_no_test,   r2_xgb_no_test],
    'test_mae': [mae_lr_no_test,  mae_gam_no_test,  mae_svr_no_test,  mae_xgb_no_test],
    'test_rmse':[rmse_lr_no_test, rmse_gam_no_test, rmse_svr_no_test, rmse_xgb_no_test],
    'test_preds': {
        'Linear Regression': pred_lr_no,
        'GAM':               pred_gam_no,
        'SVR':               pred_svr_no,
        'XGBoost':    pred_xgb_no,
    },
    'y_test':    y_test.values,
    'augmented': False,
    'multiplier': None,
}, 'results/results_no_aug.joblib')
print('Saved: results/results_no_aug.joblib')
 
# ── Save fitted models and metadata for SQ2 and SQ3 ──────────────────────────
joblib.dump(lr_no_final, 'results/lr_pipeline.pkl')
joblib.dump(best_gam_no, 'results/gam_pipeline.pkl')
joblib.dump(best_svr_no, 'results/svr_pipeline.pkl')
joblib.dump(best_xgb_no, 'results/xgb_2stage_pipeline.pkl')
 
joblib.dump(gs_gam_no.best_params_,  'results/best_gam_params.pkl')
joblib.dump(gs_svr_no.best_params_,  'results/best_svr_params.pkl')
joblib.dump(gs_xgb2_no.best_params_, 'results/best_xgb_params.pkl')
 
X_test.to_csv('results/X_test.csv', index=False)
pd.Series(y_test, name='GCI_2020').to_csv('results/y_test.csv', index=False)
df.loc[X_test.index, ['Country', 'c_code']].reset_index(drop=True).to_csv('results/df_test.csv', index=False)
 
print('No-aug models and metadata saved for SQ2 and SQ3.')
 

# -----------------------------
# Condition 2: augmented
# wider hyperparameter grids - 2x tarining data supports broader search 
# used for SQ1 augmentation comparison only
# -----------------------------

print('\n' + '#'*60)
print('# CONDITION 2: AUGMENTED')
print('#'*60)
 
# ── Linear Regression ─────────────────────────────────────────────────────────
print('\n>>> Linear Regression [aug]')
 
lr_aug = clone(lr_pipeline)
cv_lr_aug = cross_validate(lr_aug, X_aug, y_aug,
                           cv=RKF, scoring=SCORING, n_jobs=-1,
                           return_train_score=True)
r2_lr_aug, mae_lr_aug, rmse_lr_aug, tr2_lr_aug, tmae_lr_aug, trmse_lr_aug = \
    extract_scores(cv_lr_aug, return_train=True)
print_cv_summary('Linear Regression [aug]',
                 r2_lr_aug, mae_lr_aug, rmse_lr_aug,
                 tr2_lr_aug, tmae_lr_aug, trmse_lr_aug)
 
# ── GAM ───────────────────────────────────────────────────────────────────────
print('\n>>> GAM [aug] - hyperparameter tuning')
 
gs_gam_aug = GridSearchCV(clone(gam_pipeline), PARAM_GRID_GAM,
                          cv=INNER_CV, scoring='neg_mean_absolute_error',
                          n_jobs=-1, refit=True)
gs_gam_aug.fit(X_aug, y_aug)
print(f'  Best params : {gs_gam_aug.best_params_}')
print(f'  Best CV MAE : {-gs_gam_aug.best_score_:.4f}')
 
best_gam_aug = gs_gam_aug.best_estimator_
cv_gam_aug = cross_validate(best_gam_aug, X_aug, y_aug,
                            cv=RKF, scoring=SCORING, n_jobs=-1,
                            return_train_score=True)
r2_gam_aug, mae_gam_aug, rmse_gam_aug, tr2_gam_aug, tmae_gam_aug, trmse_gam_aug = \
    extract_scores(cv_gam_aug, return_train=True)
print_cv_summary('GAM [aug]',
                 r2_gam_aug, mae_gam_aug, rmse_gam_aug,
                 tr2_gam_aug, tmae_gam_aug, trmse_gam_aug)
 
# ── SVR ───────────────────────────────────────────────────────────────────────
print('\n>>> SVR [aug] - hyperparameter tuning')
 
gs_svr_aug = GridSearchCV(clone(svr_pipeline), PARAM_GRID_SVR,
                          cv=INNER_CV, scoring='neg_mean_absolute_error',
                          n_jobs=-1, refit=True)
gs_svr_aug.fit(X_aug, y_aug)
print(f'  Best params : {gs_svr_aug.best_params_}')
print(f'  Best CV MAE : {-gs_svr_aug.best_score_:.4f}')
 
best_svr_aug = gs_svr_aug.best_estimator_
cv_svr_aug = cross_validate(best_svr_aug, X_aug, y_aug,
                            cv=RKF, scoring=SCORING, n_jobs=-1,
                            return_train_score=True)
r2_svr_aug, mae_svr_aug, rmse_svr_aug, tr2_svr_aug, tmae_svr_aug, trmse_svr_aug = \
    extract_scores(cv_svr_aug, return_train=True)
print_cv_summary('SVR [aug]',
                 r2_svr_aug, mae_svr_aug, rmse_svr_aug,
                 tr2_svr_aug, tmae_svr_aug, trmse_svr_aug)
 

# ── XGBoost [aug] — hyperparameter tuning ───────────────────────────────
print('\n>>> XGBoost [aug] - hyperparameter tuning')

gs_xgb2_aug = GridSearchCV(clone(xgboost_pipeline), PARAM_GRID_XGB_COMB,
                           cv=INNER_CV, scoring='neg_mean_absolute_error',
                           n_jobs=-1, refit=True)
gs_xgb2_aug.fit(X_aug, y_aug)
print(f'  Best params : {gs_xgb2_aug.best_params_}')
print(f'  Best CV MAE : {-gs_xgb2_aug.best_score_:.4f}')

best_xgb_aug = gs_xgb2_aug.best_estimator_
cv_xgb_aug = cross_validate(best_xgb_aug, X_aug, y_aug,
                             cv=RKF, scoring=SCORING, n_jobs=-1,
                             return_train_score=True)
r2_xgb_aug, mae_xgb_aug, rmse_xgb_aug, tr2_xgb_aug, tmae_xgb_aug, trmse_xgb_aug = \
    extract_scores(cv_xgb_aug, return_train=True)
print_cv_summary('XGBoost [aug]',
                 r2_xgb_aug, mae_xgb_aug, rmse_xgb_aug,
                 tr2_xgb_aug, tmae_xgb_aug, trmse_xgb_aug)

# ── Fit final aug models on full augmented set ────────────────────────────────
print('\n>>> Fitting final models on full augmented training set [aug]')
 
lr_aug_final = clone(lr_pipeline)
lr_aug_final.fit(X_aug, y_aug)
best_gam_aug.fit(X_aug, y_aug)
best_svr_aug.fit(X_aug, y_aug)
best_xgb_aug.fit(X_aug, y_aug)
 
# ── Test set evaluation [aug] ─────────────────────────────────────────────────
print('\n>>> Final Test Set Evaluation [aug]')
 
pred_lr_aug,  r2_lr_aug_test,  mae_lr_aug_test,  rmse_lr_aug_test  = evaluate_on_test(lr_aug_final, 'Linear Regression')
pred_gam_aug, r2_gam_aug_test, mae_gam_aug_test, rmse_gam_aug_test = evaluate_on_test(best_gam_aug, 'GAM')
pred_svr_aug, r2_svr_aug_test, mae_svr_aug_test, rmse_svr_aug_test = evaluate_on_test(best_svr_aug, 'SVR')
pred_xgb_aug, r2_xgb_aug_test, mae_xgb_aug_test, rmse_xgb_aug_test = evaluate_on_test(best_xgb_aug, 'XGBoost')
 
# ── Save aug results ──────────────────────────────────────────────────────────
joblib.dump({
    'cv_r2':    [r2_lr_aug,   r2_gam_aug,   r2_svr_aug,   r2_xgb_aug],
    'cv_mae':   [mae_lr_aug,  mae_gam_aug,  mae_svr_aug,  mae_xgb_aug],
    'cv_rmse':  [rmse_lr_aug, rmse_gam_aug, rmse_svr_aug, rmse_xgb_aug],
    'train_r2': [tr2_lr_aug,  tr2_gam_aug,  tr2_svr_aug,  tr2_xgb_aug],
    'train_mae':[tmae_lr_aug, tmae_gam_aug, tmae_svr_aug, tmae_xgb_aug],
    'train_rmse':[trmse_lr_aug, trmse_gam_aug, trmse_svr_aug, trmse_xgb_aug],
    'test_r2':  [r2_lr_aug_test,   r2_gam_aug_test,   r2_svr_aug_test,   r2_xgb_aug_test],
    'test_mae': [mae_lr_aug_test,  mae_gam_aug_test,  mae_svr_aug_test,  mae_xgb_aug_test],
    'test_rmse':[rmse_lr_aug_test, rmse_gam_aug_test, rmse_svr_aug_test, rmse_xgb_aug_test],
    'test_preds': {
        'Linear Regression': pred_lr_aug,
        'GAM':               pred_gam_aug,
        'SVR':               pred_svr_aug,
        'XGBoost':    pred_xgb_aug,
    },
    'y_test':    y_test.values,
    'augmented': True
}, 'results/results_aug.joblib')
print('Saved: results/results_aug.joblib')
 
print('\n>>> Both conditions complete.')

# Add to the aug model saving block in main.py
joblib.dump(lr_aug_final,  'results/lr_aug_pipeline.pkl')
joblib.dump(best_gam_aug,  'results/gam_aug_pipeline.pkl')
joblib.dump(best_svr_aug,  'results/svr_aug_pipeline.pkl')
joblib.dump(best_xgb_aug,  'results/xgb_aug_pipeline.pkl')

joblib.dump(gs_gam_aug.best_params_,  'results/best_gam_params_aug.pkl')
joblib.dump(gs_svr_aug.best_params_,  'results/best_svr_params_aug.pkl')
joblib.dump(gs_xgb2_aug.best_params_,  'results/best_xgb_params_aug.pkl')
