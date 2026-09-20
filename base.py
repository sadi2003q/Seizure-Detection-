# base.py
# Shared contract for every model file.
#   Model        : what build() returns (has .fit and .predict)
#   ModelFamily  : one per model file (has build, search_space, best_param)
#   check_params : raises a clear error when a wrong parameter name is passed
import inspect
from abc import ABC, abstractmethod
import optuna
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

optuna.logging.set_verbosity(optuna.logging.WARNING)


def check_params(family, given, build_fn):
    """Raise a clear error if `given` contains names that `build_fn` does not accept.

    family   : model name, e.g. 'XGBoost'
    given    : dict of parameters the user passed
    build_fn : the function whose signature lists the allowed names
    """
    allowed = list(inspect.signature(build_fn).parameters)
    bad = [k for k in given if k not in allowed]
    if bad:
        raise ValueError(
            f"{family} does not accept parameter(s): {bad}.\n"
            f"Valid parameters for {family}: {allowed}")


class Model(ABC):
    """A built model. Every model file returns an object like this from build()."""
    name = 'model'
    params = {}

    @abstractmethod
    def fit(self, X_train, y_train):
        """Train in place. Must return self."""

    @abstractmethod
    def predict(self, X):
        """Return the probability of the seizure class, shape (n,)."""


class ModelFamily(ABC):
    """One per model file. Provides build() and best_param()."""
    name = 'model'

    @classmethod
    @abstractmethod
    def search_space(cls, trial):
        """Return a dict of parameters suggested by the Optuna trial."""

    @classmethod
    @abstractmethod
    def build(cls, **params):
        """Return an UNTRAINED Model. Every parameter is an argument."""

    @classmethod
    def best_param(cls, X_train, y_train, g_train, n_trials=15, n_folds=4, seed=42):
        """Optuna with patient-grouped CV (a whole patient is held out per fold).
        Prints and returns the best parameters, ready for: Family.build(**best)"""
        folds = list(GroupKFold(n_splits=n_folds).split(X_train, y_train, groups=g_train))

        def objective(trial):
            params = cls.search_space(trial)
            scores = []
            for tr, va in folds:
                m = cls.build(**params).fit(X_train[tr], y_train[tr])
                scores.append(roc_auc_score(y_train[va], m.predict(X_train[va])))
            return sum(scores) / len(scores)

        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        print(f'[{cls.name}] best CV ROC-AUC = {study.best_value:.4f}')
        print(f'[{cls.name}] best parameters : {study.best_params}')
        return study.best_params
