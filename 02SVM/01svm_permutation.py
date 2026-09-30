"""
Exact permutation test for family-wise linear SVM classifiability under
LOOCV / LOCO / LOTO. 

Labels are permuted at the block level rather than per sample. Samples are not
individually exchangeable -- each condition is 3 replicates and NR/R status is
fully crossed with 4 treatments -- so shuffling all 24 labels would break
replicate grouping. The default scheme enumerates which 4 of the 8 conditions
are labelled R, giving all C(8,4) = 70 arrangements, one of which is the true
labelling; the minimum attainable p is therefore 1/70 = 0.014.
"""

from glob import glob
import os
import itertools
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, roc_auc_score

DATA_DIR = "./data_log"
OUT_DIR = "./results/family_svm_results"
os.makedirs(OUT_DIR, exist_ok=True)

PERMUTATION_SCHEME = "condition"  # "condition" or "treatment_paired"


N_SAMPLES = 24
category_index = np.repeat(np.arange(8), 3)          # 0-7, condition id
treatment_index = category_index % 4                  # 0-3, treatment arm
TRUE_Y = np.where(category_index < 4, 0, 1)            # 0=NR, 1=R (ground truth)

CV_SCHEMES = {
    "LOOCV": (LeaveOneOut(), None),
    "LOCO": (LeaveOneGroupOut(), category_index),
    "LOTO": (LeaveOneGroupOut(), treatment_index),
}


if PERMUTATION_SCHEME == "condition":
    perm_labels = [np.array([1 if c in r_conditions else 0 for c in category_index])
                   for r_conditions in itertools.combinations(range(8), 4)]

elif PERMUTATION_SCHEME == "treatment_paired":
    perm_labels = []
    for flips in itertools.product([0, 1], repeat=4):
        y_perm = np.empty(N_SAMPLES, dtype=int)
        for cond in range(8):
            t = cond % 4
            is_original_r = int(cond >= 4)
            y_perm[category_index == cond] = (1 - is_original_r) if flips[t] else is_original_r
        perm_labels.append(y_perm)

if not any(np.array_equal(p, TRUE_Y) for p in perm_labels):
    raise RuntimeError(
        "true labeling not found among enumerated permutations - design "
        "assumptions (category_index / treatment_index) do not match TRUE_Y."
    )

n_perm = len(perm_labels)
observed_idx = next(i for i, p in enumerate(perm_labels) if np.array_equal(p, TRUE_Y))

results = []

for path in sorted(glob(os.path.join(DATA_DIR, "*.npy"))):
    if os.path.basename(path) == "sample_names.npy":
        continue

    family = os.path.splitext(os.path.basename(path))[0]

    arr = np.load(path, allow_pickle=True).astype(np.float64)
    X = arr.T

    for scheme_name, (splitter, groups) in CV_SCHEMES.items():
        null_mcc = np.empty(n_perm)
        null_bal_acc = np.empty(n_perm)
        null_auc = np.empty(n_perm)

        for i, y_perm in enumerate(perm_labels):

            y_pred = np.full(N_SAMPLES, -1, dtype=int)
            margins = np.full(N_SAMPLES, np.nan, dtype=float)

            split_iter = (
                splitter.split(X, y_perm, groups=groups) if groups is not None
                else splitter.split(X)
            )

            for train_idx, test_idx in split_iter:
                X_train, X_test = X[train_idx], X[test_idx]
                y_train = y_perm[train_idx]
                scaler = StandardScaler().fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)
                clf = SVC(kernel="linear", C=1.0, class_weight="balanced")
                clf.fit(X_train, y_train)
                y_pred[test_idx] = clf.predict(X_test)
                margins[test_idx] = clf.decision_function(X_test)


            try:
                null_mcc[i] = matthews_corrcoef(y_perm, y_pred)
            except Exception:
                null_mcc[i] = np.nan
            null_bal_acc[i] = balanced_accuracy_score(y_perm, y_pred)
            try:
                null_auc[i] = roc_auc_score(y_perm, margins)
            except ValueError:
                null_auc[i] = np.nan

        obs_mcc = null_mcc[observed_idx]
        obs_bal_acc = null_bal_acc[observed_idx]
        obs_auc = null_auc[observed_idx]


        results.append({
            "family": family,
            "cv_scheme": scheme_name,
            "n_lipids": X.shape[1],
            "n_permutations": n_perm,
            "observed_mcc": obs_mcc,
            "null_mean_mcc": np.mean(null_mcc),
            "null_std_mcc": np.std(null_mcc),
            "p_value_mcc": np.mean(null_mcc >= obs_mcc),
            "observed_balanced_accuracy": obs_bal_acc,
            "null_mean_balanced_accuracy": np.mean(null_bal_acc),
            "p_value_balanced_accuracy": np.mean(null_bal_acc >= obs_bal_acc),
            "observed_roc_auc": obs_auc,
            "p_value_roc_auc": np.mean(np.nan_to_num(null_auc, nan=-np.inf) >= obs_auc),
        })


results_df = pd.DataFrame(results)

results_df["q_value_mcc"] = np.nan
for scheme_name, grp in results_df.groupby("cv_scheme"):
    pvals = np.asarray(grp["p_value_mcc"].values, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    q = pvals[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q_full = np.empty(n)
    q_full[order] = np.clip(q, 0, 1)
    results_df.loc[grp.index, "q_value_mcc"] = q_full

results_df = results_df.sort_values(
    by=["cv_scheme", "p_value_mcc", "observed_mcc"],
    ascending=[True, True, False],
)

out_path = os.path.join(OUT_DIR, "family_svm_permutation_test.csv")
results_df.to_csv(out_path, index=False)

loto = results_df[results_df.cv_scheme == "LOTO"][
    ["family", "n_lipids", "observed_mcc", "p_value_mcc", "q_value_mcc"]
]