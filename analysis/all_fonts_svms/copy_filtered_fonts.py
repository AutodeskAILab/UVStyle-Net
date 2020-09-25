import os

import pandas as pd
import numpy as np

from util import Grams

if __name__ == '__main__':
    source_dir = '../uvnet_data/solidmnist_all'
    font_filter_labels_file = 'filtered_font_labels.csv'
    target_dir = '../uvnet_data/solidmnist_filtered_font_families'

    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    fonts_to_keep = pd.read_csv(font_filter_labels_file, header=None)

    grams = Grams(data_root=source_dir)

    df = pd.DataFrame({
        'graph_file': grams.graph_files
    })

    df['font'] = df['graph_file'].apply(lambda f: f[2:-10])
    df['keep'] = df['font'].apply(lambda f: f in fonts_to_keep[0].values.tolist())

    keep_idx = df[df['keep']].index.tolist()
    df[df['keep']]['graph_file'].to_csv(target_dir + '/graph_files.txt', index=None, header=False)

    for i, layer in enumerate(grams.layer_names):
        filtered_grams = grams[i][keep_idx]
        np.save(f'{target_dir}/{layer}_grams', filtered_grams)