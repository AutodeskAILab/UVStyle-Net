from ast import literal_eval

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from figures_for_paper.single_letter_hits.single_letter_hits_plot import heatmap, annotate_heatmap

if __name__ == '__main__':
    df = pd.read_csv('gaussian_weights_scores_font_51.csv')
    df['mu'], df['sigma'] = zip(*df['mu_sigma'].apply(literal_eval))

    sigmas = list(reversed(df['sigma'].unique()))
    mus = df['mu'].unique()

    arr = np.zeros([df['sigma'].nunique(), df['mu'].nunique()])
    for i, sigma in enumerate(sigmas):
        for j, mu in enumerate(mus):
            matching_sigma = df[df['sigma'] == sigma]
            matching_both = matching_sigma[matching_sigma['mu'] == mu]
            arr[i, j] = matching_both['score']

    fig, ax = plt.subplots(squeeze=True)  # type: plt.Figure, plt.Axes
    im, cbar = heatmap(data=arr,
                       row_labels=sigmas,
                       col_labels=mus,
                       label_top=False)
    annotate_heatmap(im=im,
                     valfmt='{x:.1f}',
                     textcolors=['white', 'black'])
    ax.set_title(f'Hits@10 (Font 51)')
    ax.set_xlabel('mu')
    ax.set_ylabel('sigma')
    fig.set_size_inches(6, 5)
    fig.tight_layout()
    fig.savefig('all_fonts_weight_layers_gaussian_plot_font_51.pdf')
    fig.show()

