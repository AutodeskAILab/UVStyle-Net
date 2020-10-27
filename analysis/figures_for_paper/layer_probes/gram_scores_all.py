import pandas as pd

from layer_stats_analysis import probe_score
from util import Grams

if __name__ == '__main__':
    grams = {
        'uvnet': Grams('../../uvnet_data/solidmnist_font_subset_face_norm_only'),
        'pointnet': Grams('../../pointnet2_data/solidmnist_font_subset'),
        'meshcnn': Grams('../../meshcnn_data/solidmnist_font_subset')
    }
    for model, gram in grams.items():
        df = pd.DataFrame(gram.layer_names, columns=['layer'])
        labels = gram.labels
        df['linear_probe'], df['linear_probe_err'] = zip(
            *list(map(lambda x: probe_score(None, x, labels, err=True), gram)))
        df.to_csv(f'{model}_layer_probe_scores_with_err.csv', index=False)
