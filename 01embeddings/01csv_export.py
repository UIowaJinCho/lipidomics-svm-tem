from glob import glob
import os
import csv
import numpy as np
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import connected_components
from sklearn.decomposition import PCA
import umap

"""
Embeds every lipid family into 3 components with ISOMAP, LLE, UMAP and PCA,
and writes one csv per family per method per neighbourhood size.
Neighbour-based methods use the smallest k that leaves the kNN graph connected.
"""

DATA_DIR = './data_znorm'
ISOMAP_OUT = './isomap_csv_znorm'
LLE_OUT = './lle_csv_znorm'
UMAP_OUT = './umap_csv_znorm'
PCA_OUT = './pca_csv_znorm'

for directory in [ISOMAP_OUT, LLE_OUT, UMAP_OUT, PCA_OUT]:
    os.makedirs(directory, exist_ok=True)

category_names = [
    'NR-C', 'NR-ST', 'NR-TMZ', 'NR-TMZ-ST',
    'R-C', 'R-ST', 'R-TMZ', 'R-TMZ-ST'
]

# 8 categories x 3 samples each = 24 samples
categories = np.repeat(np.arange(8), 3)

# UMAP neighbourhood size per family. Each k was fixed before any NR/R colouring
# was applied, by taking the value that maximised the unsupervised trustworthiness
# of the 3-component embedding for that family; resistance status played no part
# in the choice. Each family is then swept across k-1, k, k+1 so no figure rests
# on a single neighbourhood size.
umap_k = {
    'acylcarnitines': 2,
    'ceramides': 3,
    'cerebrosides': 2,
    'cholesteryl_esters': 3,
    'diglycerides': 4,
    'etherplasmalogen_pc': 2,
    'etherplasmalogen_pe': 3,
    'free_fatty_acids': 3,
    'gangliosides': 3,
    'globosides': 5,
    'lysophosphatidylcholine': 4,
    'lysophosphatidylethanolamine': 3,
    'phosphatidylcholine': 3,
    'phosphatidylethanolamine': 2,
    'phosphatidylglycerol': 3,
    'phosphatidylinositol': 3,
    'phosphatidylserine': 3,
    'sphingomyelins': 2,
    'triglycerides': 3
}

data_files = sorted(glob(os.path.join(DATA_DIR, '*.npy')))
labels = np.load(os.path.join(DATA_DIR, 'sample_names.npy'), allow_pickle=True)

for file in data_files:

    base = os.path.basename(file)

    # label arrays, the sample-name array, and the two families too small to embed
    if (base.endswith('_labels.npy')
            or 'sample_names' in base
            or 'free_cholesterol' in base
            or 'oxidized_phospholipids' in base):
        continue

    lipid_name = base.replace('.npy', '')
    X = np.load(file).transpose()

    if len(labels) != X.shape[0] or len(categories) != X.shape[0]:
        continue


    # smallest k that leaves the neighbourhood graph in one piece, ISOMAP and LLE
    # both fail or fragment below this. k is set by graph connectivity alone, so
    # it is fixed before the labels are ever applied
    k0 = None
    for k in range(1, X.shape[0]):
        graph = kneighbors_graph(X, n_neighbors=k, mode='connectivity', include_self=False)
        n_components, _ = connected_components(graph, directed=False)
        if n_components == 1:
            k0 = k
            break

    if k0 is None:
        continue

    neighbour_ks = [k for k in [k0, k0 + 1, k0 + 2] if k < X.shape[0]]

    # UMAP is swept around its own dictionary value instead of the connectivity k
    if lipid_name in umap_k:
        best_k = umap_k[lipid_name]
        if best_k == 2:
            umap_ks = [2, 3, 4]
        else:
            umap_ks = [best_k - 1, best_k, best_k + 1]
        umap_ks = sorted(set(k for k in umap_ks if 2 <= k < X.shape[0]))
    else:
        umap_ks = []

    # each method writes the same csv layout so the plotting scripts are interchangeable
    embeddings = []

    for k in neighbour_ks:
        reduced = Isomap(n_components=3, n_neighbors=k).fit_transform(X)
        embeddings.append((ISOMAP_OUT, '%s_isomap_k%i.csv' % (lipid_name, k), reduced))

        reduced = LocallyLinearEmbedding(n_components=3, n_neighbors=k).fit_transform(X)
        embeddings.append((LLE_OUT, '%s_lle_k%i.csv' % (lipid_name, k), reduced))

    for k in umap_ks:
        reducer = umap.UMAP(n_components=3, n_neighbors=k, random_state=42)
        reduced = reducer.fit_transform(X)
        embeddings.append((UMAP_OUT, '%s_umap_k%i.csv' % (lipid_name, k), reduced))

    pca = PCA(n_components=3)
    reduced = pca.fit_transform(X)
    embeddings.append((PCA_OUT, '%s_pca.csv' % lipid_name, reduced))
    for out_dir, filename, reduced in embeddings:
        out_path = os.path.join(out_dir, filename)
        with open(out_path, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['label', 'component_1', 'component_2', 'component_3',
                             'category_index', 'category_name'])
            for i, label in enumerate(labels):
                cat_idx = int(categories[i])
                writer.writerow([str(label),
                                 float(reduced[i, 0]),
                                 float(reduced[i, 1]),
                                 float(reduced[i, 2]),
                                 cat_idx,
                                 category_names[cat_idx]])