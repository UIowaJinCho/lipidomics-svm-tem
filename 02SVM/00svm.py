"""
Family-wise linear SVM classifiability under three cross-validation schemes of
increasing strictness: leave-one-sample-out (LOOCV), leave-one-condition-out
(LOCO), and leave-one-treatment-out (LOTO). LOTO holds out an entire treatment
arm from both NR and R, so it tests whether the resistance signature generalizes
to a pharmacological context never seen in training, and is the scheme reported
as primary. Standardization is refit inside every training fold.
"""

from glob import glob
import os
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
)

DATA_DIR = "./data_log"
OUT_DIR = "./results/family_svm_results"
os.makedirs(OUT_DIR, exist_ok=True)


category_index = np.repeat(np.arange(8), 3)
y = np.where(category_index < 4, 0, 1)  # 0 = NR, 1 = R

category_names = np.array([
    "NR-C", "NR-C", "NR-C",
    "NR-ST", "NR-ST", "NR-ST",
    "NR-TMZ", "NR-TMZ", "NR-TMZ",
    "NR-TMZ-ST", "NR-TMZ-ST", "NR-TMZ-ST",
    "R-C", "R-C", "R-C",
    "R-ST", "R-ST", "R-ST",
    "R-TMZ", "R-TMZ", "R-TMZ",
    "R-TMZ-ST", "R-TMZ-ST", "R-TMZ-ST",
])

# Treatment arm (C, ST, TMZ, TMZ-ST), independent of NR/R status.
# category_index runs 0..7 as [NR-C, NR-ST, NR-TMZ, NR-TMZ-ST, R-C, R-ST, R-TMZ, R-TMZ-ST],
# so treatment repeats every 4 -> category_index % 4 recovers the treatment arm.
treatment_index = category_index % 4
treatment_names_lookup = np.array(["C", "ST", "TMZ", "TMZ-ST"])
treatment_names = treatment_names_lookup[treatment_index]

N_SAMPLES = 24
assert category_index.shape[0] == N_SAMPLES
assert len(np.unique(category_index)) == 8
assert len(np.unique(treatment_index)) == 4

CV_SCHEMES = {
    "LOOCV": (LeaveOneOut(), None),
    "LOCO": (LeaveOneGroupOut(), category_index),
    "LOTO": (LeaveOneGroupOut(), treatment_index),
}


summary_rows = []
sample_rows = []

for path in sorted(glob(os.path.join(DATA_DIR, "*.npy"))):

    if os.path.basename(path) == "sample_names.npy":
        continue

    family = os.path.splitext(os.path.basename(path))[0]

    arr = np.load(path, allow_pickle=True).astype(np.float64)

    X = arr.T

    final_clf = SVC(kernel="linear", C=1.0, class_weight="balanced")
    final_clf.fit(StandardScaler().fit_transform(X), y)
    n_support_total = int(np.sum(final_clf.n_support_))
    n_support_NR = int(final_clf.n_support_[0])
    n_support_R = int(final_clf.n_support_[1])

    for scheme_name, (splitter, groups) in CV_SCHEMES.items():

        y_pred = np.full(N_SAMPLES, -1, dtype=int)
        margins = np.full(N_SAMPLES, np.nan, dtype=float)

        split_iter = (
            splitter.split(X, y, groups=groups) if groups is not None
            else splitter.split(X)
        )

        for train_idx, test_idx in split_iter:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train = y[train_idx]
            scaler = StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
            clf = SVC(kernel="linear", C=1.0, class_weight="balanced")
            clf.fit(X_train, y_train)
            y_pred[test_idx] = clf.predict(X_test)
            margins[test_idx] = clf.decision_function(X_test)


        tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()

        try:
            auc = roc_auc_score(y, margins)
        except ValueError:
            auc = np.nan

        nr_margins = margins[y == 0]
        r_margins = margins[y == 1]

        n_folds = N_SAMPLES if groups is None else len(np.unique(groups))

        summary_rows.append({
            "family": family,
            "cv_scheme": scheme_name,
            "n_folds": n_folds,
            "n_lipids": X.shape[1],
            "accuracy": accuracy_score(y, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "sensitivity": recall_score(y, y_pred, zero_division=0),
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
            "f1": f1_score(y, y_pred, zero_division=0),
            "mcc": matthews_corrcoef(y, y_pred),
            "roc_auc": auc,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "mean_margin_NR": np.mean(nr_margins),
            "mean_margin_R": np.mean(r_margins),
            "mean_abs_margin_NR": np.mean(np.abs(nr_margins)),
            "mean_abs_margin_R": np.mean(np.abs(r_margins)),
            "std_margin_NR": np.std(nr_margins, ddof=1),
            "std_margin_R": np.std(r_margins, ddof=1),
            "margin_gap_minR_minus_maxNR": np.min(r_margins) - np.max(nr_margins),
            "n_support_total": n_support_total,
            "n_support_NR": n_support_NR,
            "n_support_R": n_support_R,
        })

        for i in range(N_SAMPLES):
            sample_rows.append({
                "family": family,
                "cv_scheme": scheme_name,
                "sample_index": i,
                "category_index": int(category_index[i]),
                "category_name": category_names[i],
                "treatment": treatment_names[i],
                "status": "NR" if y[i] == 0 else "R",
                "true_label": int(y[i]),
                "predicted_label": int(y_pred[i]),
                "correct": int(y[i] == y_pred[i]),
                "svm_margin": float(margins[i]),
                "abs_margin": float(abs(margins[i])),
            })

summary_df = pd.DataFrame(summary_rows)
sample_df = pd.DataFrame(sample_rows)

summary_df = summary_df.sort_values(
    by=["cv_scheme", "mcc", "balanced_accuracy", "roc_auc"],
    ascending=[True, False, False, False],
)

summary_path = os.path.join(OUT_DIR, "family_svm_cv_summary.csv")
sample_path = os.path.join(OUT_DIR, "family_svm_cv_sample_margins.csv")

summary_df.to_csv(summary_path, index=False)
sample_df.to_csv(sample_path, index=False)


ladder = summary_df.pivot(index="family", columns="cv_scheme", values="mcc")
ladder = ladder[["LOOCV", "LOCO", "LOTO"]]
ladder = ladder.sort_values("LOTO", ascending=False)

ladder_path = os.path.join(OUT_DIR, "family_svm_mcc_ladder.csv")
ladder.to_csv(ladder_path)