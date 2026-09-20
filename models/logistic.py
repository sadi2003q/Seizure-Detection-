# models/logistic.py
from sklearn.linear_model import LogisticRegression
from base import Model, ModelFamily, check_params


class _Logistic(Model):
    name = 'LogReg'

    def __init__(self, **params):
        self.params = params
        self.m = LogisticRegression(**params)

    def fit(self, X_train, y_train):
        self.m.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.m.predict_proba(X)[:, 1]


class LogisticModel(ModelFamily):
    name = 'LogReg'

    @classmethod
    def search_space(cls, trial):
        return dict(C=trial.suggest_float('C', 1e-3, 10, log=True),
                    max_iter=trial.suggest_int('max_iter', 200, 1000))

    @classmethod
    def build(cls, **params):
        check_params('LogReg', params, cls._build)     # clear error on wrong names
        return cls._build(**params)

    @classmethod
    def _build(cls, C=0.1, penalty='l2', solver='lbfgs', max_iter=300,
               class_weight=None, random_state=42):
        return _Logistic(C=C, penalty=penalty, solver=solver, max_iter=max_iter,
                         class_weight=class_weight, random_state=random_state)
