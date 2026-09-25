# models/__init__.py
# Each model is optional. If its library is missing, only that model is disabled and
# calling it gives a clear message. Logistic Regression needs only scikit-learn.
from models.logistic import LogisticModel


def _unavailable(model_name, error):
    """Placeholder class used when a model's library is not installed."""
    message = f'{model_name} is unavailable: {error}. Install its library to use it.'

    class _Missing:
        name = model_name

        @classmethod
        def _fail(cls, *args, **kwargs):
            raise ImportError(message)

        build = search_space = best_param = _fail

    return _Missing


try:
    from models.xgboost_model import XGBoostModel
except Exception as e:
    XGBoostModel = _unavailable('XGBoost', e)

try:
    from models.mamba_model import MambaModel
except Exception as e:
    MambaModel = _unavailable('Mamba', e)
