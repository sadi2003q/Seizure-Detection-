# models/xgboost_model.py
import xgboost as xgb
from base import Model, ModelFamily, check_params


class _XGB(Model):
    name = 'XGBoost'

    def __init__(self, **params):
        self.params = params
        self.m = xgb.XGBClassifier(**params, eval_metric='logloss',
                                   tree_method='hist', n_jobs=-1, verbosity=0)

    def fit(self, X_train, y_train):
        self.m.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.m.predict_proba(X)[:, 1]


class XGBoostModel(ModelFamily):
    name = 'XGBoost'

    @classmethod
    def search_space(cls, trial):
        return dict(
            n_estimators=trial.suggest_int('n_estimators', 200, 800),
            max_depth=trial.suggest_int('max_depth', 3, 9),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            subsample=trial.suggest_float('subsample', 0.6, 1.0),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.3, 1.0),
            min_child_weight=trial.suggest_int('min_child_weight', 5, 50),
            gamma=trial.suggest_float('gamma', 0.0, 2.0),
            reg_alpha=trial.suggest_float('reg_alpha', 0.01, 10, log=True),
            reg_lambda=trial.suggest_float('reg_lambda', 0.01, 10, log=True))

    @classmethod
    def build(cls, **params):
        check_params('XGBoost', params, cls._build)    # clear error on wrong names
        return cls._build(**params)

    @classmethod
    def _build(cls, n_estimators=500, max_depth=8, learning_rate=0.05, subsample=0.9,
               colsample_bytree=0.6, min_child_weight=10, gamma=0.0,
               reg_alpha=0.1, reg_lambda=1.0, random_state=42):
        return _XGB(n_estimators=n_estimators, max_depth=max_depth,
                    learning_rate=learning_rate, subsample=subsample,
                    colsample_bytree=colsample_bytree, min_child_weight=min_child_weight,
                    gamma=gamma, reg_alpha=reg_alpha, reg_lambda=reg_lambda,
                    random_state=random_state)
