import os

import pandas as pd
import numpy as np


def get_summary(directory):
    categories = os.listdir(os.path.join(root, directory))

    count = np.zeros([len(categories), len(categories)], dtype=np.int)

    for i, cat_i in enumerate(categories):
        for j, cat_j in enumerate(categories):
            imgs_i = set(os.listdir(os.path.join(root, directory, cat_i)))
            imgs_j = set(os.listdir(os.path.join(root, directory, cat_j)))
            count[i, j] = len(imgs_i.intersection(imgs_j))

    return pd.DataFrame(count, index=categories, columns=categories)


if __name__ == '__main__':
    root = './abc_quant_data'
    summaries = [get_summary(directory) for directory in os.listdir(root)]
    for summary in summaries:
        print(summary.to_markdown())
