import numpy as np
from glob import glob
import os

"""
Converts raw lipid abundances into the two matrices the rest of the pipeline uses:
log2 abundances for the SVM, and per-lipid z-scored log2 abundances for the embeddings.
"""

DATA_DIR = './data'
LOG_DIR = './data_log'
ZNORM_DIR = './data_znorm'

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(ZNORM_DIR, exist_ok=True)

npy_files = glob(os.path.join(DATA_DIR, '*.npy'))

for npy_file in npy_files:

    # label and sample-name arrays are carried through untouched
    base = os.path.basename(npy_file)
    if 'labels' in base or 'sample_names' in base:
        continue

    data = np.load(npy_file).astype(np.float64)

    # lipid abundance -> log abundance, for SVM
    log_data = np.log2(data + 1e-8)
    np.save(os.path.join(LOG_DIR, base), log_data)

    # z-score each lipid across the 24 samples, for embeddings
    mean = log_data.mean(axis=1, keepdims=True)
    sd = log_data.std(axis=1, ddof=0, keepdims=True)

    # a lipid with no variance would divide by zero
    sd[sd == 0] = 1.0

    znorm_data = (log_data - mean) / sd
    np.save(os.path.join(ZNORM_DIR, base), znorm_data)
