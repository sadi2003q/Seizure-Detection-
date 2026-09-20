# models/__init__.py
from models.logistic import LogisticModel
from models.xgboost_model import XGBoostModel

# Mamba needs PyTorch. If it is not installed, the rest of the project still works,
# and calling MambaModel gives a clear message instead of breaking the import.
try:
    from models.mamba_model import MambaModel
except Exception as _err:
    _MAMBA_ERROR = str(_err)      # saved: Python deletes `_err` when this block ends

    class MambaModel:             # placeholder that explains what is missing
        name = 'Mamba'

        @classmethod
        def _fail(cls, *args, **kwargs):
            raise ImportError(f'Mamba is unavailable: {_MAMBA_ERROR}. '
                              f'Install PyTorch to use it.')

        build = search_space = best_param = _fail