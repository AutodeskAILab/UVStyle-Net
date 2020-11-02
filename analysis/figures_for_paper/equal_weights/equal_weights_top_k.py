import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors._kd_tree import KDTree

from figures_for_paper.low_mid_upper.low_mid_high import spherize
from util import Grams, weight_layers, KNNGrid, Images, IdMap, OnTheFlyImages

if __name__ == '__main__':
    uv_net_data_root = '../../uvnet_data/solidmnist_sub_mu_only'
    pointnet_data_root = '../../pointnet2_data/solidmnist_font_subset'
    meshcnn_data_root = '../../meshcnn_data/solidmnist_font_subset'
    img_path = '/home/pete/brep_style/solidmnist/test_pngs'
    grams = {
        'UV-Net': Grams(data_root=uv_net_data_root),
        'Pointnet++': Grams(data_root=pointnet_data_root),
        'MeshCNN': Grams(data_root=meshcnn_data_root)
    }

    images = {
        'UV-Net': OnTheFlyImages(data_root=uv_net_data_root,
                                 img_root=img_path),
        'Pointnet++': OnTheFlyImages(data_root=pointnet_data_root,
                                     img_root=img_path),
        'MeshCNN': OnTheFlyImages(data_root=meshcnn_data_root,
                                  img_root=img_path)
    }

    letter_idx = {
        name: i for i, name in enumerate(map(lambda n: n[:-4], grams['UV-Net'].graph_files))
    }

    fig, axes = plt.subplots(3, squeeze=True)
    for i, (model, gram) in enumerate(grams.items()):
        query_idx = np.array([
            letter_idx['y_Abhaya Libre_upper'],
            letter_idx['n_Zhi Mang Xing_lower'],
            letter_idx['l_Seaweed Script_upper'],
            letter_idx['e_Turret Road_upper']
        ])
        if model == 'Pointnet++':
            id_map = IdMap(src_file=uv_net_data_root + '/graph_files.txt',
                           dest_file=pointnet_data_root + '/graph_files.txt')
            query_idx = id_map(query_idx)
        elif model == 'MeshCNN':
            id_map = IdMap(src_file=uv_net_data_root + '/graph_files.txt',
                           dest_file=meshcnn_data_root + '/graph_files.txt')
            query_idx = id_map(query_idx)

        # select lower half of layers
        n = len(gram) // 2
        weights = np.zeros(len(gram))
        weights[:n] = 1

        layer = weight_layers(gram.grams, weights)
        layer = spherize(layer)

        knn_grid = KNNGrid(KDTree(layer), images[model])

        queries = layer[query_idx]
        im = knn_grid._get_image(queries, k=6, label_fn=lambda idx: gram.labels[idx])
        ax = axes[i]
        ax.imshow(im)
        ax.set_yticks([])
        # ax.set_title(model, size='large')
        ax.set_ylabel(model)
        ax.set_xticks(np.arange(6) * 256 + 128)
        ax.set_xticklabels(['Q', '1', '2', '3', '4', '5'])
    fig.set_size_inches(4, 8)
    fig.tight_layout()
    fig.savefig(f'equal_weights_topk.pdf')
    plt.show()
