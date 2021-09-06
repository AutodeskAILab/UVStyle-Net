import os
import sys

import pandas as pd
from sklearn.decomposition import PCA

file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(file_dir)
sys.path.append(project_root)
from utils import Grams, probe_score

if __name__ == '__main__':
    grams = Grams(os.path.join(project_root, 'data', 'SolidLETTERS', 'grams', 'subset'))
    df = pd.DataFrame(grams.layer_names, columns=['layer'])
    labels = grams.labels
    df['linear_probe'], df['linear_probe_err'] = zip(
        *[probe_score(stat=i,
                      reduced=PCA(70).fit_transform(x) if x.shape[-1] > 70 else x,
                      labels=labels,
                      err=True)
          for i, x in enumerate(grams)])
    df.to_csv('linear_probes.csv', index=False)
    print(df)
