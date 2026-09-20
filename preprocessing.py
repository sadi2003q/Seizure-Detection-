# preprocessing.py
import numpy as np, pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

N_CH, N_FEAT = 22, 42
TEST_PATIENTS = ['chb16', 'chb17', 'chb18', 'chb19', 'chb20', 'chb22']
VAL_PATIENTS  = ['chb14', 'chb15']


def _patient(subject_id):
    if subject_id.startswith('chb17'): return 'chb17'   # chb17a/b/c = one patient
    if subject_id == 'chb21':          return 'chb01'   # same patient as chb01
    return subject_id


def prepare_data(csv_path, seed=42):
    """
    Returns three dicts keyed by 'train' / 'validation' / 'test':
        X[...]  features (normalised)     y[...]  labels     g[...]  patient ids
    Order of operations (this prevents leakage):
        1) split by PATIENT   2) fit imputer + scaler on train only   3) shuffle train
    """
    df = pd.read_csv(csv_path)
    df['patient'] = df['subject_id'].map(_patient)

    meta = ['label', 'subject_id', 'event_id', 'patient']
    cols = [c for c in df.columns if c not in meta]
    assert len(cols) == N_CH * N_FEAT, f'expected 924 features, got {len(cols)}'

    feats = df[cols].replace([np.inf, -np.inf], np.nan)
    label, patient = df['label'].values, df['patient'].values

    masks = {'test':       df['patient'].isin(TEST_PATIENTS).values,
             'validation': df['patient'].isin(VAL_PATIENTS).values}
    masks['train'] = ~(masks['test'] | masks['validation'])

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

    for part in ['train', 'validation', 'test']:
        print(f'{part:10s} n={len(y[part]):6d}  seizure={y[part].mean():.1%}  '
              f'patients={sorted(set(g[part]))}')
    return X, y, g
