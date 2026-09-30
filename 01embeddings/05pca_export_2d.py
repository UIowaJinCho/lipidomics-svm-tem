from glob import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""
Plots every PCA embedding csv as a two-panel figure: samples coloured by
resistance status, and samples coloured by treatment with status as the marker shape.
"""

CSV_DIR = './pca_csv_znorm'
SAVE_DIR = './results/pca_embeddings'

os.makedirs(SAVE_DIR, exist_ok=True)
sns.set_theme(style='whitegrid', context='paper', font_scale=1.2)

status_palette = {
    'NR': '#D55E00',
    'R': '#0072B2'
}

treatment_palette = {
    'C': '#4C72B0',
    'ST': '#55A868',
    'TMZ': '#E69F00',
    'TMZ-ST': '#8172B2'
}

treatment_names = {
    0: 'C',
    1: 'ST',
    2: 'TMZ',
    3: 'TMZ-ST'
}

csv_files = sorted(glob(os.path.join(CSV_DIR, '*.csv')))

for csv_file in csv_files:

    df = pd.read_csv(csv_file)

    # category_index runs 0-7, the first four are NR and the treatment repeats every four
    df['status'] = np.where(df['category_index'] < 4, 'NR', 'R')
    df['treatment_id'] = df['category_index'] % 4
    df['treatment'] = df['treatment_id'].map(treatment_names)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    # left panel, resistance status only
    sns.scatterplot(
        data=df,
        x='component_1',
        y='component_2',
        hue='status',
        hue_order=['NR', 'R'],
        palette=status_palette,
        s=85,
        edgecolor='black',
        linewidth=0.5,
        alpha=0.9,
        ax=ax[0]
    )

    ax[0].set_title('Non-resistant vs Resistant')
    ax[0].set_xlabel('PCA Component 1')
    ax[0].set_ylabel('PCA Component 2')
    ax[0].legend(title='', frameon=True)

    # right panel, treatment as colour and status as marker so both factors are visible
    sns.scatterplot(
        data=df,
        x='component_1',
        y='component_2',
        hue='treatment',
        hue_order=['C', 'ST', 'TMZ', 'TMZ-ST'],
        palette=treatment_palette,
        style='status',
        style_order=['NR', 'R'],
        markers={'NR': 'o', 'R': '^'},
        s=90,
        edgecolor='black',
        linewidth=0.5,
        alpha=0.9,
        ax=ax[1]
    )

    ax[1].set_title('Treatment-matched categories')
    ax[1].set_xlabel('PCA Component 1')
    ax[1].set_ylabel('PCA Component 2')
    ax[1].legend(title='', frameon=True, fontsize=9)

    plt.suptitle(os.path.basename(csv_file), y=1.03)
    plt.tight_layout()

    filename = os.path.splitext(os.path.basename(csv_file))[0] + '.png'
    save_path = os.path.join(SAVE_DIR, filename)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
