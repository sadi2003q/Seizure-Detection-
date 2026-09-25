# preprocessing.py
import warnings
import numpy as np, pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

N_CH, N_FEAT = 22, 42
TEST_PATIENTS = ['chb16', 'chb17', 'chb18', 'chb19', 'chb20', 'chb22']
VAL_PATIENTS  = ['chb14', 'chb15']

# ── per-patient normalisation settings ─────────────────────────────────
NORM_MODES = (None, 'interictal', 'all')
NORM_GROUP = 'subject_id'   # finest grouping in the CSV (chb17a/b/c and chb21 are separate sessions)
CLIP = 20.0                 # after per-patient scaling, values are clipped to [-CLIP, CLIP]
FEATURE_SETS = ('all', 'conventional', 'snn')


def _patient(subject_id):
    if subject_id.startswith('chb17'): return 'chb17'   # chb17a/b/c = one patient
    if subject_id == 'chb21':          return 'chb01'   # same patient as chb01
    return subject_id


def _normalise_per_group(F, groups, label, mode):
    """Centre and scale every group's features with that group's OWN median and IQR.

    mode 'interictal': statistics come from the group's label-0 (seizure-free) windows only.
                       Assumes a seizure-free calibration recording exists for each new patient.
    mode 'all'       : statistics come from all of the group's windows. No labels are used, but the
                       result depends on how many seizure windows the group happens to contain.
    Returns (normalised float32 array, size of the smallest reference set).
    """
    out = np.empty(F.shape, dtype=np.float32)
    smallest = None
    for grp in np.unique(groups):
        rows = groups == grp
        ref = (rows & (label == 0)) if mode == 'interictal' else rows
        n_ref = int(ref.sum())
        if n_ref < 30:
            raise ValueError(f"Group '{grp}' has only {n_ref} reference windows for mode "
                             f"'{mode}'; at least 30 are needed for stable statistics.")
        smallest = n_ref if smallest is None else min(smallest, n_ref)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)      # all-NaN columns
            q1, med, q3 = np.nanpercentile(F[ref], [25, 50, 75], axis=0)
        scale = q3 - q1
        scale = np.where(scale > 1e-12, scale, 1.0)              # NaN or ~0 spread: leave unscaled
        out[rows] = np.clip((F[rows] - med) / scale, -CLIP, CLIP)
    return out, smallest


def load_raw(csv_path, per_patient_norm=None):
    """Read the CSV ONCE. The returned dict is reused by make_split() and run_loso().

    per_patient_norm : None (default)  no per-patient step (one global scaler is fitted later)
                       'interictal'    scale each patient with the median/IQR of that patient's
                                       seizure-free windows (needs a seizure-free calibration recording)
                       'all'           same, but using all of the patient's windows
    Each patient is scaled using ONLY that patient's own rows, so no patient's data
    influences another patient's features.
    """
    if per_patient_norm is False:
        per_patient_norm = None
    if per_patient_norm not in NORM_MODES:
        raise ValueError(f"per_patient_norm must be one of {NORM_MODES}, got {per_patient_norm!r}. "
                         "Use 'interictal' (recommended) or 'all'.")

    df = pd.read_csv(csv_path)
    df['patient'] = df['subject_id'].map(_patient)

    # Metadata columns to exclude from the feature matrix. Older CSVs only have the first four;
    # newer ones also carry window_start_sample / window_center_sample / in_core_ictal.
    meta = ['label', 'subject_id', 'event_id', 'patient',
            'window_start_sample', 'window_center_sample', 'in_core_ictal']
    cols = [c for c in df.columns if c not in meta]
    assert len(cols) == N_CH * N_FEAT, f'expected {N_CH * N_FEAT} features, got {len(cols)}'

    feats = df[cols].replace([np.inf, -np.inf], np.nan)
    label, patient = df['label'].values, df['patient'].values

    if per_patient_norm:
        arr, smallest = _normalise_per_group(feats.to_numpy(dtype=np.float64),
                                             df[NORM_GROUP].values, label, per_patient_norm)
        feats = pd.DataFrame(arr, columns=cols)
        how = ("statistics from each patient's seizure-free windows (assumes a seizure-free "
               "calibration recording per new patient)" if per_patient_norm == 'interictal'
               else "statistics from all of each patient's windows (depends on seizure fraction)")
        print(f"Per-patient normalisation '{per_patient_norm}': {how}; "
              f"smallest reference set = {smallest} windows")

    return dict(feats=feats, label=label, patient=patient, cols=cols,
                per_patient_norm=per_patient_norm)


def select_features(raw, features='all'):
    """Column indices for the ablation study.
    'all' (924), 'conventional' (no spike features) or 'snn' (spike features only)."""
    if features not in FEATURE_SETS:
        raise ValueError(f"features must be one of {FEATURE_SETS}, got {features!r}")
    is_snn = np.array([c.split('_', 1)[1].startswith('snn_') for c in raw['cols']])
    if features == 'all':
        return np.arange(len(raw['cols']))
    return np.where(is_snn if features == 'snn' else ~is_snn)[0]


def make_split(raw, train_patients, val_patients, test_patients, seed=42, verbose=True):
    """Split by PATIENT, fit imputer + scaler on the TRAIN patients only, shuffle train.
    Returns X, y, g: dicts keyed 'train' / 'validation' / 'test'."""
    parts = {'train': list(train_patients), 'validation': list(val_patients), 'test': list(test_patients)}
    seen = [p for ps in parts.values() for p in ps]
    assert len(seen) == len(set(seen)), 'a patient appears in more than one partition'
    known = set(raw['patient'])
    missing = [p for p in seen if p not in known]
    if missing:
        raise ValueError(f'unknown patient(s): {missing}. Available: {sorted(known)}')

    feats, label, patient = raw['feats'], raw['label'], raw['patient']
    masks = {k: np.isin(patient, v) for k, v in parts.items()}

    imp = SimpleImputer(strategy='median').fit(feats[masks['train']])
    sc  = StandardScaler().fit(imp.transform(feats[masks['train']]))

    X, y, g = {}, {}, {}
    for part, m in masks.items():
        Xp = sc.transform(imp.transform(feats[m])).astype(np.float32)
        yp, gp = label[m], patient[m]
        if part == 'train':
            order = np.random.default_rng(seed).permutation(len(yp))
            Xp, yp, gp = Xp[order], yp[order], gp[order]
        X[part], y[part], g[part] = Xp, yp, gp

    assert set(g['train']).isdisjoint(g['validation']) \
       and set(g['train']).isdisjoint(g['test']) \
       and set(g['validation']).isdisjoint(g['test']), 'PATIENT LEAK between partitions'

    if verbose:
        for part in ['train', 'validation', 'test']:
            print(f'{part:10s} n={len(y[part]):6d}  seizure={y[part].mean():.1%}  '
                  f'patients={sorted(set(g[part]))}')
    return X, y, g


def prepare_data(csv_path, seed=42, per_patient_norm=None):
    """The fixed split used so far (test: chb16-22, validation: chb14/15, train: the rest).
    Returns X, y, g dicts keyed 'train' / 'validation' / 'test'."""
    raw = load_raw(csv_path, per_patient_norm)
    held = set(TEST_PATIENTS) | set(VAL_PATIENTS)
    train = [p for p in sorted(set(raw['patient'])) if p not in held]
    return make_split(raw, train, VAL_PATIENTS, TEST_PATIENTS, seed=seed)
