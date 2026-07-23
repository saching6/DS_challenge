# Borrowed from an existing codebase

import pandas as pd
import numpy as np 
import joblib
from tqdm import tqdm


import xgboost as xgb
from sklearn.model_selection import ParameterSampler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV

from utils.metrics import * 

from config.global_vars import DEFAULT_PARAMS, GRID_SEARCH_SPACE



class LogisticPredictor:
    """Fitted logistic baseline + the one-hot column layout it was trained on.
    `columns` is frozen at fit time so inference reindexes to the same layout.
    """
    CATS = ['line_primary', 'symptom', 'hardware_type']  
    # station excluded on purpose
 
    def __init__(self, model, columns, numeric):
        self.model = model
        self.columns = columns
        self.numeric = numeric
 
    def preprocess(self, df):
        dummies = pd.get_dummies(df[self.CATS].astype(str), drop_first=True)
        X = pd.concat([dummies, df[self.numeric]], axis=1)
        return X.reindex(columns=self.columns, fill_value=0)
 
    def predict_proba(self, df):
        return self.model.predict_proba(self.preprocess(df))[:, 1]
 
    def __call__(self, df):
        return self.predict_proba(df)

class XGBPredictor:
    """Fitted XGBoost + its feature spec, in one self-contained object.
    Carrying `features` and `categorical` on the object means inference at load
    time doesn't re-import training config.
    """
 
    def __init__(self, model, features, categorical):
        self.model = model
        self.features = features
        self.categorical = categorical
 
    def preprocess(self, df):
        X = df[self.features].copy()
        for c in self.categorical:
            X[c] = X[c].astype('category')
        return X
 
    def predict_proba(self, df):
        return self.model.predict_proba(self.preprocess(df))[:, 1]
 
    def __call__(self, df):
        return self.predict_proba(df)

# logistic regression
def fit_logistic(X_train, y_train, numeric_features, calibration_method='sigmoid'):
    '''
    `station` is dropped from the linear model: one-hot at 216 levels leads to overfitting
    '''
    dummies = pd.get_dummies(X_train[LogisticPredictor.CATS].astype(str), drop_first=True)
    X = pd.concat([dummies, X_train[numeric_features]], axis=1)
    columns = X.columns

    estimator = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight='balanced'))
    model = CalibratedClassifierCV(estimator, method=calibration_method, cv=10)
    model.fit(X, y_train)

    return LogisticPredictor(model, columns , numeric_features)


def fit_xgb(X_train,y_train,features, categorical,
            params = DEFAULT_PARAMS, calibrated=True, calibration_method='sigmoid'):
    """XGBoost, optionally calibrated. Returns a self-contained XGBPredictor.
    """
    estimator = xgb.XGBClassifier(**params, tree_method='hist',
                             enable_categorical=True, eval_metric='aucpr',
                             random_state=0)
    if calibrated:
        estimator = CalibratedClassifierCV(estimator, method=calibration_method, cv=10)
        estimator.fit(X_train, y_train)
    else:
        estimator.fit(X_train, y_train)
 
    return XGBPredictor(estimator, features, categorical)

def tune_xgb(X_train,y_train, X_val_select, y_val_select, features, categorical,
             n_trials=50, seed=0):
    """Random search, scored on the selection half of validation.
    X_train: training data
    y_train: training labels

    X_val_select: validation data for hyperparameter tuning
    y_val_select: validation labels for hyperparameter tuning
 
    Random rather than a full grid: a full grid is too expensive, random search covers each
    one's range more efficiently.
 
    Configs are scored uncalibrated. skipping it here cheaper. 
    Caveat: CalibratedClassifierCV(cv=10) refits on 90% subsets and averages, so the
    final model is not literally the object the search scored.
    """
    param_samples = list(ParameterSampler(GRID_SEARCH_SPACE, n_iter=n_trials, random_state=seed))
    trials = []
    for params in tqdm(param_samples):
        predict = fit_xgb(X_train,y_train, features, categorical, params, calibrated=False)
        score = average_precision_score(y_val_select, predict(X_val_select))
        trials.append({'pr_auc': score, **params})
 
    results = pd.DataFrame(trials).sort_values('pr_auc', ascending=False)
    best = results.iloc[0].drop('pr_auc').to_dict()
    best = {k: (float(v) if k == 'learning_rate' else int(v))
            for k, v in best.items()}
    return best, results