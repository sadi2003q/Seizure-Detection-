# annotation.py
"""Seizure-label definition test: does the +10 s / +5 s annotation extension change performance?

Compares two label definitions on the SAME preprocessing pipeline:
  'extended'  : label == 1  (the definition used in the paper; includes pre/post-ictal margins)
  'core'      : in_core_ictal == True  (only the true seizure onset-to-offset interval)

Rows that are label==1 but NOT in_core_ictal (the pre/post-ictal-only windows) are
DROPPED for the 'core' definition -- they are neither its positive nor its negative class.
"""
import numpy as np


def load_raw_annotation(csv_path, definition='extended', per_patient_norm=None):
    """Same as preprocessing.load_raw(), but lets you choose the positive-class definition.

    definition : 'extended' (default, as in the paper) or 'core' (true seizure interval only)
    Returns a `raw` dict with the SAME shape as load_raw(), so it plugs straight into
    make_split() / run_loso() with no other code changes.
    """
    import pandas as pd
    from preprocessing import N_CH, N_FEAT, _patient, _normalise_per_group

    if definition not in ('extended', 'core'):
        raise ValueError(f"definition must be 'extended' or 'core', got {definition!r}")

    df = pd.read_csv(csv_path)
    if 'in_core_ictal' not in df.columns:
        raise ValueError("This CSV has no 'in_core_ictal' column; the label-definition comparison needs it.")

    if definition == 'core':
        # drop the pre/post-ictal-only windows (label==1 & not core); keep label==0 and core-ictal
        keep = (df['label'] == 0) | df['in_core_ictal']
        n_dropped = int((~keep).sum())
        print(f"definition='core': dropping {n_dropped:,} pre/post-ictal-only windows "
              f"(label==1 & in_core_ictal==False); {int(keep.sum()):,} windows remain")
        df = df.loc[keep].reset_index(drop=True)
        label = df['in_core_ictal'].astype(int).values
    else:
        label = df['label'].values

    df = df.copy()
    df['patient'] = df['subject_id'].map(_patient)
    meta = ['label', 'subject_id', 'event_id', 'patient',
            'window_start_sample', 'window_center_sample', 'in_core_ictal']
    cols = [c for c in df.columns if c not in meta]
    assert len(cols) == N_CH * N_FEAT, f'expected {N_CH * N_FEAT} features, got {len(cols)}'

    feats = df[cols].replace([np.inf, -np.inf], np.nan)
    patient = df['patient'].values

    if per_patient_norm:
        arr, smallest = _normalise_per_group(feats.to_numpy(dtype=np.float64),
                                             df['subject_id'].values, label, per_patient_norm)
        feats = pd.DataFrame(arr, columns=cols)
        print(f"Per-patient normalisation '{per_patient_norm}' applied; "
              f"smallest reference set = {smallest} windows")

    for pid in np.unique(patient):
        n_pos = int(label[patient == pid].sum())
        if n_pos < 10:
            print(f"WARNING: patient {pid} has only {n_pos} positive windows under "
                  f"definition='{definition}'; its LOSO fold may be unstable.")

    return dict(feats=feats, label=label, patient=patient, cols=cols,
                per_patient_norm=per_patient_norm, definition=definition)
