import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors._kd_tree import KDTree

from figures_for_paper.low_mid_upper.low_mid_high import spherize
from util import Grams, weight_layers, KNNGrid, Images, IdMap, OnTheFlyImages

if __name__ == '__main__':
    uv_net_data_root = '../../uvnet_data/solidmnist_font_subset'
    pointnet_data_root = '../../pointnet2_data/solidmnist_font_subset'
    meshcnn_data_root = '../../meshcnn_data/solidmnist_font_subset'
    grams = {
        'UV-Net': Grams(data_root=uv_net_data_root),
        'Pointnet++': Grams(data_root=pointnet_data_root),
        'MeshCNN': Grams(data_root=meshcnn_data_root)
    }

    images = {
        'UV-Net': OnTheFlyImages(data_root=uv_net_data_root,
                         img_root='/Users/t_meltp/solid-mnist/mesh/test_pngs'),
        'Pointnet++': OnTheFlyImages(data_root=pointnet_data_root,
                             img_root='/Users/t_meltp/solid-mnist/mesh/test_pngs'),
        'MeshCNN': OnTheFlyImages(data_root=meshcnn_data_root,
                             img_root='/Users/t_meltp/solid-mnist/mesh/test_pngs')
    }


    fig, axes = plt.subplots(3, squeeze=True)
    for i, (model, gram) in enumerate(grams.items()):
        query_idx = np.array([124, 94, 121, 95])
        if model == 'Pointnet++':
            id_map = IdMap(src_file=uv_net_data_root + '/graph_files.txt',
                           dest_file=pointnet_data_root + '/graph_files.txt')
            query_idx = id_map(query_idx)
        elif model == 'MeshCNN':
            id_map = IdMap(src_file=uv_net_data_root + '/graph_files.txt',
                           dest_file=meshcnn_data_root + '/graph_files.txt')
            query_idx = id_map(query_idx)

        layer = weight_layers(gram.grams, np.ones(len(gram)))
        layer = spherize(layer)

        knn_grid = KNNGrid(KDTree(layer), images[model])

        queries = layer[query_idx]
        im = knn_grid._get_image(queries, k=6)
        ax = axes[i]
        ax.imshow(im)
        ax.set_yticks([])
        ax.set_title(model, size='large')
        ax.set_xticks(np.arange(6) * 256 + 128)
        ax.set_xticklabels(['Q', '1', '2', '3', '4', '5'])
    fig.set_size_inches(4, 9)
    fig.tight_layout()
    fig.savefig(f'equal_weights_topk.pdf')
    plt.show()
