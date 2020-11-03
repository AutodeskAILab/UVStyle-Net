import math

import numpy as np

from util import Grams

if __name__ == '__main__':
    grams = {
        'UV-Net': Grams('../../uvnet_data/solidmnist_all_fnorm'),
        'Pointnet++': Grams('../../pointnet2_data/solidmnist_font_subset'),
        'MeshCNN': Grams('../../meshcnn_data/solidmnist_font_subset')
    }

    for model, gram in grams.items():
        sizes = []
        for layer in gram:
            if model == 'UV-Net':
                sizes.append(layer.shape[-1])
            else:
                s = math.sqrt(layer.shape[-1])
                sizes.append(int((s * (s + 1)) / 2))
        print(f'model: {model}, total size: {int(np.sum(sizes)) * 4:3,}, mean size: {int(np.mean(sizes)) * 4:3,}, max size: {int(np.max(sizes))}')
