import torchvision
import matplotlib.pyplot as plt
import numpy as np

from util import Images

if __name__ == '__main__':
    all_imgs = Images(data_root='../../uvnet_data/solidmnist_single_letter',
                      img_root='/Users/t_meltp/uvnet-img/png/test',
                      cache_file='../../cache/single_letter_pngs')

    examples = {
        'Slanted': {
            'positive': [7, 27, 87, 147, 300],
            'negative': [34, 35, 40, 68, 125]
        },
        'Serif': {
            'positive': [11, 25, 52, 137, 208],
            'negative': [0, 5, 24, 34, 46]
        },
        'No Curves': {
            'positive': [29, 40, 68, 78, 99],
            'negative': [23, 43, 51, 53, 107]
        }
    }

    fig, axes = plt.subplots(3, squeeze=True)
    for i, (style, style_egs) in enumerate(examples.items()):
        egs = style_egs['positive'] + style_egs['negative']
        eg_imgs = map(lambda i: all_imgs[i], egs)
        eg_imgs = map(torchvision.transforms.Resize((256, 256)), eg_imgs)
        img_tensors = list(map(torchvision.transforms.ToTensor(), eg_imgs))

        grid = torchvision.utils.make_grid(img_tensors, nrow=5).permute((1, 2, 0))

        ax = axes[i]
        ax.imshow(grid)
        ax.set_title(style, size='large')

        ax.set_yticks(np.arange(2) * 256 + 128)
        ax.set_yticklabels(['+ve', '-ve'])

        ax.set_xticks([])
    fig.set_size_inches((4, 6))
    fig.tight_layout()
    fig.savefig('positives_and_negatives.pdf')
    fig.show()
