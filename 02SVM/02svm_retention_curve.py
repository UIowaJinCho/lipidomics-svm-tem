from glob import glob
import os
from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef

"""
EXPERIMENTAL / PRELIMINARY ANALYSIS

This script is an exploratory attempt to investigate what drives the
family-level linear SVM classifiers. It is intended for hypothesis generation
and model interpretation, not for confirmatory statistical inference or
validated feature selection. Ranking, subset selection and evaluation all use
the same 24 samples, so reduced-signature performance is optimistic by
construction and the recommended subsets themselves are not stable.

Within-family lipid stability and reduced-signature retention for the family SVM.
Produces the three supplementary tables: per-lipid coefficient stability, the
retention curve, and the permutation test of the recommended reduced signature.
Matches the main SVM pipeline: linear kernel, fixed C = 1.0, no tuning, and
standardization refit inside every training fold.
"""

DATA_DIR = './data_log'
OUT_DIR = './results/family_svm_results'

SVM_C = 1.0
MCC_TOLERANCE = 0.0
CLUSTER_THRESHOLD = 0.9

os.makedirs(OUT_DIR, exist_ok=True)

N_SAMPLES = 24
category_index = np.repeat(np.arange(8), 3)
treatment_index = category_index % 4
TRUE_Y = np.where(category_index < 4, 0, 1)

PERM_LABELS = np.array([np.isin(category_index, c).astype(int)
                        for c in combinations(range(8), 4)])
N_PERM = len(PERM_LABELS)
OBS_PERM_IDX = int(np.where((PERM_LABELS == TRUE_Y).all(axis=1))[0][0])

family_paths = sorted(glob(os.path.join(DATA_DIR, '*.npy')))
family_paths = [p for p in family_paths
                if 'sample_names' not in os.path.basename(p)
                and 'labels' not in os.path.basename(p)]


stability_rows = []
curve_rows = []
reduced_rows = []

for path in family_paths:

    family = os.path.splitext(os.path.basename(path))[0]
    X_full = np.asarray(np.load(path, allow_pickle=True), dtype=np.float64).T

    if X_full.shape[0] != N_SAMPLES or np.isnan(X_full).any() or np.isinf(X_full).any():
        continue

    n_lipids = X_full.shape[1]
    lipid_names = ['%s_%03d' % (family, i) for i in range(n_lipids)]

    if n_lipids < 2:
        continue


    coef_by_scheme = {}

    for scheme_name, groups in [('LOOCV', None), ('LOTO', treatment_index)]:

        splitter = LeaveOneOut() if groups is None else LeaveOneGroupOut()
        splits = (list(splitter.split(X_full)) if groups is None
                  else list(splitter.split(X_full, TRUE_Y, groups=groups)))

        fold_coefs = []

        for train_idx, _ in splits:
            scaler = StandardScaler().fit(X_full[train_idx])
            clf = SVC(kernel='linear', C=SVM_C, class_weight='balanced')
            clf.fit(scaler.transform(X_full[train_idx]), TRUE_Y[train_idx])

            fold_coefs.append(clf.coef_.ravel())

        coef_by_scheme[scheme_name] = np.vstack(fold_coefs)

    loocv_coefs = coef_by_scheme['LOOCV']
    loto_coefs = coef_by_scheme['LOTO']
    mean_loocv = loocv_coefs.mean(axis=0)
    sd_loocv = loocv_coefs.std(axis=0, ddof=0)
    sign_loocv = np.abs(np.sign(loocv_coefs).mean(axis=0))
    mean_loto = loto_coefs.mean(axis=0)
    sd_loto = loto_coefs.std(axis=0, ddof=0)
    sign_loto = np.abs(np.sign(loto_coefs).mean(axis=0))

    NR, R = X_full[TRUE_Y == 0], X_full[TRUE_Y == 1]
    pooled_sd = np.sqrt((NR.var(axis=0, ddof=1) + R.var(axis=0, ddof=1)) / 2.0)
    cohens_d = np.where(pooled_sd > 0,
                        (R.mean(axis=0) - NR.mean(axis=0)) / pooled_sd, np.nan)

    corr = np.corrcoef(X_full, rowvar=False)
    cluster_id = np.full(n_lipids, -1, dtype=int)
    next_cluster = 0
    for i in range(n_lipids):
        if cluster_id[i] >= 0:
            continue
        cluster_id[i] = next_cluster
        for j in range(i + 1, n_lipids):
            if cluster_id[j] < 0 and abs(corr[i, j]) >= CLUSTER_THRESHOLD:
                cluster_id[j] = next_cluster
        next_cluster += 1

    rank_score = np.abs(mean_loto) * sign_loto / (sd_loto + 1e-12)
    rank_within = (-rank_score).argsort().argsort() + 1
    order = np.argsort(-rank_score)

    y_pred_full = np.full(N_SAMPLES, -1, dtype=int)
    for train_idx, test_idx in LeaveOneGroupOut().split(
            X_full, TRUE_Y, groups=treatment_index):
        scaler = StandardScaler().fit(X_full[train_idx])
        clf = SVC(kernel='linear', C=SVM_C, class_weight='balanced')
        clf.fit(scaler.transform(X_full[train_idx]), TRUE_Y[train_idx])
        y_pred_full[test_idx] = clf.predict(scaler.transform(X_full[test_idx]))

    reference_mcc = matthews_corrcoef(TRUE_Y, y_pred_full)
    reference_bacc = balanced_accuracy_score(TRUE_Y, y_pred_full)

    ablation_delta = np.full(n_lipids, np.nan)
    for i in range(n_lipids):
        keep_idx = [j for j in range(n_lipids) if j != i]
        X_drop = X_full[:, keep_idx]
        y_pred = np.full(N_SAMPLES, -1, dtype=int)
        for train_idx, test_idx in LeaveOneGroupOut().split(
                X_drop, TRUE_Y, groups=treatment_index):
            scaler = StandardScaler().fit(X_drop[train_idx])
            clf = SVC(kernel='linear', C=SVM_C, class_weight='balanced')
            clf.fit(scaler.transform(X_drop[train_idx]), TRUE_Y[train_idx])
            y_pred[test_idx] = clf.predict(scaler.transform(X_drop[test_idx]))
        ablation_delta[i] = balanced_accuracy_score(TRUE_Y, y_pred) - reference_bacc

    family_curve = []
    for k in range(1, n_lipids + 1):

        keep = order[:k]
        X_sub = X_full[:, keep]
        y_pred = np.full(N_SAMPLES, -1, dtype=int)

        for train_idx, test_idx in LeaveOneGroupOut().split(
                X_sub, TRUE_Y, groups=treatment_index):
            scaler = StandardScaler().fit(X_sub[train_idx])
            clf = SVC(kernel='linear', C=SVM_C, class_weight='balanced')
            clf.fit(scaler.transform(X_sub[train_idx]), TRUE_Y[train_idx])
            y_pred[test_idx] = clf.predict(scaler.transform(X_sub[test_idx]))

        entry = {
            'family': family,
            'n_lipids_in_family': n_lipids,
            'k': k,
            'lipids_retained': ';'.join(lipid_names[i] for i in keep),
            'loto_mcc': matthews_corrcoef(TRUE_Y, y_pred),
            'loto_balanced_accuracy': balanced_accuracy_score(TRUE_Y, y_pred),
            'full_feature_loto_mcc': reference_mcc
        }
        entry['meets_tolerance_criterion'] = bool(
            entry['loto_mcc'] >= reference_mcc - MCC_TOLERANCE)
        family_curve.append(entry)

    curve_rows.extend(family_curve)

    meeting = [r for r in family_curve if r['meets_tolerance_criterion']]
    k_star = min(r['k'] for r in meeting) if meeting else n_lipids
    keep = order[:k_star]

    for i in range(n_lipids):
        stability_rows.append({
            'family': family,
            'lipid': lipid_names[i],
            'n_lipids_in_family': n_lipids,
            'mean_weight_LOOCV': mean_loocv[i],
            'weight_sd_LOOCV': sd_loocv[i],
            'sign_consistency_LOOCV': sign_loocv[i],
            'mean_weight_LOTO': mean_loto[i],
            'weight_sd_LOTO': sd_loto[i],
            'sign_consistency_LOTO': sign_loto[i],
            'cohens_d_NR_vs_R': cohens_d[i],
            'corr_cluster_id': int(cluster_id[i]),
            'rank_score': rank_score[i],
            'rank_within_family': int(rank_within[i]),
            'ablation_delta_balanced_accuracy_LOTO': ablation_delta[i]
        })

    X_sub = X_full[:, keep]
    null_mcc = np.empty(N_PERM)

    for perm_index, y_perm in enumerate(PERM_LABELS):
        y_pred = np.full(N_SAMPLES, -1, dtype=int)
        for train_idx, test_idx in LeaveOneGroupOut().split(
                X_sub, y_perm, groups=treatment_index):
            scaler = StandardScaler().fit(X_sub[train_idx])
            clf = SVC(kernel='linear', C=SVM_C, class_weight='balanced')
            clf.fit(scaler.transform(X_sub[train_idx]), y_perm[train_idx])
            y_pred[test_idx] = clf.predict(scaler.transform(X_sub[test_idx]))
        null_mcc[perm_index] = matthews_corrcoef(y_perm, y_pred)

    observed = null_mcc[OBS_PERM_IDX]

    reduced_rows.append({
        'family': family,
        'n_lipids_in_family': n_lipids,
        'recommended_k': k_star,
        'is_reduced_vs_full': k_star < n_lipids,
        'lipids_in_reduced_signature': ';'.join(lipid_names[i] for i in keep),
        'observed_mcc_reduced': observed,
        'full_feature_loto_mcc': reference_mcc,
        'p_value_mcc': float(np.mean(null_mcc >= observed)),
        'n_permutations': N_PERM
    })


reduced_df = pd.DataFrame(reduced_rows)

p = reduced_df['p_value_mcc'].to_numpy(float)
order_p = np.argsort(p)
scaled = p[order_p] * len(p) / (np.arange(len(p)) + 1)
scaled = np.minimum.accumulate(scaled[::-1])[::-1]
q = np.empty(len(p))
q[order_p] = np.clip(scaled, 0, 1)
reduced_df['q_value_mcc'] = q

pd.DataFrame(stability_rows).to_csv(
    os.path.join(OUT_DIR, 'family_lipid_stability.csv'), index=False)
pd.DataFrame(curve_rows).to_csv(
    os.path.join(OUT_DIR, 'family_retention_curve.csv'), index=False)
reduced_df.to_csv(
    os.path.join(OUT_DIR, 'family_reduced_signature_permutation_test.csv'), index=False)
