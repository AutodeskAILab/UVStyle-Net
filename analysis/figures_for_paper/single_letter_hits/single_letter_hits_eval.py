import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torchvision
from sklearn.neighbors._kd_tree import KDTree

from constrained_optimization import optimize
from figures_for_paper.low_mid_upper.low_mid_high import spherize
from util import Grams, get_pca_3_70, weight_layers, IdMap, Images

if __name__ == '__main__':

    metric = 'cosine'

    models = {
        'uvnet': Grams(data_root='../../uvnet_data/solidmnist_single_letter'),
        'pointnet': Grams(data_root='../../pointnet2_data/solidmnist_single_letter')
    }

    images = {
        'uvnet': Images(data_root='../../uvnet_data/solidmnist_single_letter',
                        img_root='/Users/t_meltp/uvnet-img/png/test',
                        cache_file='../../cache/single_letter_pngs'),
        'pointnet': Images(data_root='../../pointnet2_data/solidmnist_single_letter',
                           img_root='/Users/t_meltp/uvnet-img/png/test',
                           cache_file='../../cache/single_letter_pngs_pointnet2')
    }

    plot_when = [(3, 1, 'nocurve')]
    to_plot = {}

    for model, grams in models.items():
        labels = pd.read_csv('../../extra_labels.csv')
        labels.set_index(f'{model}_id', inplace=True)
        labels.sort_index(inplace=True)

        if model == 'pointnet':
            id_map = IdMap(src_file='../../uvnet_data/solidmnist_single_letter/graph_files.txt',
                           dest_file='../../pointnet2_data/solidmnist_single_letter/graph_files.txt')
        else:
            id_map = lambda x: x
        examples = {
            'slanted': {
                'positive': id_map([7, 27, 87, 147, 300]),
                'negative': id_map([34, 35, 40, 68, 125])
            },
            'serif': {
                'positive': id_map([11, 25, 52, 137, 208]),
                'negative': id_map([0, 5, 24, 34, 46])
            },
            'nocurve': {
                'positive': id_map([29, 40, 68, 78, 99]),
                'negative': id_map([23, 43, 51, 53, 107])
            }
        }

        pca_3, pca_70 = get_pca_3_70(grams, cache_file=f'../../cache/{model}_solidmnist_single_letter')

        for style, style_egs in examples.items():
            positives = style_egs['positive']
            negatives = style_egs['negative']

            df = pd.DataFrame()
            for p in range(1, len(positives) + 1):
                res = []
                for n in range(len(negatives) + 1):
                    # perform weight optimization
                    weights = optimize(positive_idx=positives[:p],
                                       negative_idx=negatives[:n],
                                       grams=list(pca_70.values()),
                                       metric=metric)

                    combined = weight_layers(pca_70.values(), weights)

                    if metric == 'cosine':
                        combined = spherize(combined)

                    # QUERIES
                    # get center of positive examples
                    query_idx = positives
                    queries = combined[query_idx, :].mean(0, keepdims=True)

                    kd_tree = KDTree(combined)
                    results = kd_tree.query(queries, k=10, return_distance=False).flatten()

                    r = labels.loc[results][style].sum() / 10
                    # print(f'style: {style}, p: {positives[:p]}, n: {negatives[:n]} -- {r}')
                    res.append(r)

                    # PLOTTING
                    if (p, n, style) in plot_when:
                        if (p, n, style) not in to_plot:
                            to_plot[p, n, style] = {}
                        to_plot[p, n, style][model] = results
                df[str(p)] = res
            print(df)
            df.to_csv(f'{model}_{style}.txt')


    titles = {
        'uvnet': 'UV-Net',
        'pointnet': 'Pointnet++'
    }
    # DO PLOTS
    for i, ((p, n, style), both_model_results) in enumerate(to_plot.items()):
        fig, axes = plt.subplots(2, squeeze=True)
        for model_i, (model, result) in enumerate(both_model_results.items()):
            ax = axes[model_i]
            imgs_to_show = map(lambda r: images[model][r], result)
            imgs_to_show = map(torchvision.transforms.Resize((256, 256)), imgs_to_show)
            img_tensors = list(map(torchvision.transforms.ToTensor(), imgs_to_show))
            grid = torchvision.utils.make_grid(img_tensors, nrow=5).permute((1, 2, 0))
            ax.imshow(grid)
            ax.set_title(titles[model], size='large')

            # ax.set_ylabel('Layer')
            ax.set_yticks([128, 128+256])
            ax.set_yticklabels(['1-5', '6-10'])
            #
            ax.set_xticks([])
            # ax.set_xticklabels(['Q', '1', '2', '3', '4', '5'])
        fig.tight_layout()
        fig.savefig(f'single_letter_hits_selection_{i}_cosine.pdf')
        fig.show()