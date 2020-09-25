import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torchvision
from sklearn.decomposition import PCA
from sklearn.neighbors._kd_tree import KDTree
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity

from constrained_optimization import optimize, optimize_with_gaussian
from figures_for_paper.low_mid_upper.low_mid_high import spherize
from util import Grams, get_pca_3_70, weight_layers, IdMap, Images


def cosine_kernel(x0: np.ndarray, x1: np.ndarray):
    return 1 - np.dot(x0, x1) / (np.linalg.norm(x0) * np.linalg.norm(x1))


if __name__ == '__main__':

    metric = 'cosine'

    models = {
        'uvnet': Grams(data_root='../../uvnet_data/solidmnist_single_letter'),
        # 'pointnet': Grams(data_root='../../pointnet2_data/solidmnist_single_letter')
    }

    images = {
        'uvnet': Images(data_root='../../uvnet_data/solidmnist_single_letter',
                        img_root='/Users/t_meltp/uvnet-img/png/test',
                        cache_file='../../cache/single_letter_pngs'),
        'pointnet': Images(data_root='../../pointnet2_data/solidmnist_single_letter',
                           img_root='/Users/t_meltp/uvnet-img/png/test',
                           cache_file='../../cache/single_letter_pngs_pointnet2')
    }
    styles = [
        'slanted',
        'serif',
        'nocurve'
    ]

    for model, grams in models.items():
        labels = pd.read_csv('../../extra_labels.csv')
        labels.set_index(f'{model}_id', inplace=True)
        labels.sort_index(inplace=True)

        if model == 'pointnet':
            id_map = IdMap(src_file='../../uvnet_data/solidmnist_single_letter/graph_files.txt',
                           dest_file='../../pointnet2_data/solidmnist_single_letter/graph_files.txt')
        else:
            id_map = lambda x: x

        for t in range(10):
            pca_3, pca_70 = get_pca_3_70(grams, cache_file=f'../../cache/{model}_solidmnist_single_letter')

            for style in styles:
                positives = labels[labels[style]].index.dropna()
                negatives = labels[labels[style] == False].index.dropna()
                positives = np.random.choice(positives, 5, replace=False).astype(np.int)
                negatives = np.random.choice(negatives, 5, replace=False).astype(np.int)

                df = pd.DataFrame()
                for p in range(1, len(positives) + 1):
                    res = []
                    for n in range(len(negatives) + 1):
                        # perform weight optimization
                        grams_ = list(pca_70.values())
                        d = 65 if model == 'uvnet' else 6
                        for i in range(len(grams_)):
                            pca = PCA(d)
                            grams_[i] = pca.fit_transform(grams_[i]) if grams_[i].shape[-1] > 65 else grams_[i]
                            # print(pca.explained_variance_ratio_)
                            # print(pca.n_components)
                        weights, _, _ = optimize_with_gaussian(positive_idx=positives[:p],
                                           negative_idx=negatives[:n],
                                           grams=grams_,
                                           )

                        combined = weight_layers(grams_, weights)

                        dist = cosine_similarity(combined, combined)
                        cls = SVC(kernel='precomputed', class_weight='balanced')
                        y = labels[style].loc[0:len(combined) - 1]
                        score = cross_val_score(estimator=cls,
                                                X=dist,
                                                y=y.values.astype(np.int),
                                                scoring='roc_auc').mean()
                        res.append(score)
                    df[str(p)] = res
                print(df)
                df.to_csv(f'{model}_{style}_svm_{t}.txt', sep=',')

    # COLLATE SCORES
    def load_df(model, style, i):
        df = pd.read_csv(f'{model}_{style}_svm_{i}.txt', sep=',')
        df['trial'] = i
        return df

    for model in models.keys():
        for style in styles:
            df = pd.concat([load_df(model, style, i) for i in range(10)])
            df = df.groupby('Unnamed: 0').mean().drop('trial', axis=1)
            df.to_csv(f'{model}_{style}_svm.txt', sep=',')