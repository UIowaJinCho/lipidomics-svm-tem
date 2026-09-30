from glob import glob
import os
import re
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.metrics import silhouette_score

"""
Measures NR vs R separation for every lipid family in the raw feature space and in
each embedding space, using centroid distance, dispersion, silhouette and PERMANOVA.

Raw-space values are the reporting statistics, the embedding rows are for exploratory purposes only.
"""

FEATURE_DIR = './data_znorm'

EMBEDDING_DIRS = {
    'PCA': './pca_csv_znorm',
    'ISOMAP': './isomap_csv_znorm',
    'UMAP': './umap_csv_znorm',
    'LLE': './lle_csv_znorm'
}

OUT_DIR = './results/statistics'
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUT_DIR, 'family_separation_statistics.csv')

PERMUTATIONS = 9999
RANDOM_STATE = 42

# 8 groups x 3 replicates = 24 samples, the first four groups are NR
categories = np.repeat(np.arange(8), 3)
resistance_labels = np.where(categories < 4, 0, 1)

# every feature matrix and every embedding csv is appended here as one row
rows = []

# the same matrix is assembled for the feature space and for each embedding space,
# so both are collected first and measured in a single pass below
targets = []

feature_files = sorted(glob(os.path.join(FEATURE_DIR, '*.npy')))

for file in feature_files:

    base = os.path.basename(file)

    # label arrays, the sample-name array, and the two families too small to embed
    if (base.endswith('_labels.npy')
            or 'sample_names' in base
            or 'free_cholesterol' in base
            or 'oxidized_phospholipids' in base):
        continue

    family = base.replace('.npy', '')

    # stored as lipid species x samples, the statistics need samples x species
    X = np.load(file).T

    targets.append((family, 'Feature_Log2Z', np.nan, X.shape[1], X, resistance_labels))

for space, directory in EMBEDDING_DIRS.items():

    for file in sorted(glob(os.path.join(directory, '*.csv'))):

        base = os.path.basename(file)
        family = re.sub(r'_(isomap|umap|lle|pca)(_k\d+)?\.csv$', '', base)

        # PCA files carry no k, the neighbour-based ones encode it in the filename
        match = re.search(r'_k(\d+)\.csv$', base)
        k = int(match.group(1)) if match else np.nan

        df = pd.read_csv(file)
        X = df[['component_1', 'component_2', 'component_3']].values

        # the csv carries its own category column, fall back to the global design
        if 'category_index' in df.columns:
            labels = np.where(df['category_index'].values < 4, 0, 1)
        else:
            labels = resistance_labels

        targets.append((family, space, k, np.nan, X, labels))

for family, space, k, n_species, X, labels in targets:

    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)

    if X.shape[0] != len(labels):
        raise ValueError('%s %s: X rows %i != labels %i'
                         % (family, space, X.shape[0], len(labels)))

    NR = X[labels == 0]
    R = X[labels == 1]

    centroid_NR = NR.mean(axis=0)
    centroid_R = R.mean(axis=0)
    centroid_distance = float(np.linalg.norm(centroid_NR - centroid_R))

    # pseudo-F responds to spread as well as to centroid displacement, so both
    # dispersions are carried through and read alongside it
    within_NR = float(np.mean(cdist(NR, centroid_NR.reshape(1, -1), metric='euclidean')))
    within_R = float(np.mean(cdist(R, centroid_R.reshape(1, -1), metric='euclidean')))
    mean_within = (within_NR + within_R) / 2.0

    # centroid distance in units of within-group spread, so spaces with different
    # coordinate scales stay comparable
    normalized_centroid_distance = (centroid_distance / mean_within
                                    if mean_within > 0 else np.nan)

    between_distance = float(np.mean(cdist(NR, R, metric='euclidean')))
    dispersion_ratio = between_distance / mean_within if mean_within > 0 else np.nan

    try:
        silhouette = float(silhouette_score(X, labels, metric='euclidean'))
    except Exception:
        silhouette = np.nan

    # one-way PERMANOVA on euclidean distances, partitioning total sum of squares
    # into within-group and between-group components
    rng = np.random.default_rng(RANDOM_STATE)
    unique_groups = np.unique(labels)
    n = X.shape[0]
    g = len(unique_groups)
    D = squareform(pdist(X, metric='euclidean'))

    observed_F = np.nan
    observed_R2 = np.nan
    permanova_p = np.nan

    try:
        permutation_F = np.zeros(PERMUTATIONS)

        # the observed labelling is measured first, then the same statistic is
        # recomputed under PERMUTATIONS random relabellings to build the null
        for permutation_index in range(PERMUTATIONS + 1):

            current_labels = labels if permutation_index == 0 else rng.permutation(labels)

            ss_total = np.sum(D ** 2) / n
            ss_within = 0.0

            for group in unique_groups:
                idx = np.where(current_labels == group)[0]
                if len(idx) <= 1:
                    continue
                Dg = D[np.ix_(idx, idx)]
                ss_within += np.sum(Dg ** 2) / len(idx)

            ss_between = ss_total - ss_within
            ms_between = ss_between / (g - 1)
            ms_within = ss_within / (n - g)

            F = ms_between / ms_within if ms_within > 0 else np.nan

            if permutation_index == 0:
                observed_F = F
                observed_R2 = ss_between / ss_total if ss_total > 0 else np.nan
            else:
                permutation_F[permutation_index - 1] = F

        permanova_p = (np.sum(permutation_F >= observed_F) + 1) / (PERMUTATIONS + 1)



    rows.append({
        'Family': family,
        'Space': space,
        'k': k,
        'n_species': n_species,
        'n_samples': X.shape[0],
        'n_NR': int(np.sum(labels == 0)),
        'n_R': int(np.sum(labels == 1)),
        'n_dimensions': X.shape[1],
        'Centroid_Distance': centroid_distance,
        'Normalized_Centroid_Distance': normalized_centroid_distance,
        'Within_NR_Dispersion': within_NR,
        'Within_R_Dispersion': within_R,
        'Between_NR_R_Distance': between_distance,
        'Dispersion_Ratio': dispersion_ratio,
        'Silhouette_NR_R': silhouette,
        'PERMANOVA_F': observed_F,
        'PERMANOVA_R2': observed_R2,
        'PERMANOVA_p': permanova_p
    })

results = pd.DataFrame(rows)

results = results.sort_values(
    by=['Space', 'PERMANOVA_R2', 'Normalized_Centroid_Distance'],
    ascending=[True, False, False])

results.to_csv(OUT_CSV, index=False)