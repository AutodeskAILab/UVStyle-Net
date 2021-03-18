import os
import sys

import PIL
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.utils import make_grid


def open_image(img_path):
    try:
        img = PIL.Image.open(img_path)
        arr = np.array(img.convert('RGB'))

        black_x, black_y = np.where((arr == [0, 0, 0]).all(axis=-1))
        green_x, green_y = np.where((arr == [0, 255, 0]).all(axis=-1))

        arr = (arr + 50).astype(np.uint8)

        arr[black_x, black_y, :] = [255, 255, 255]
        arr[green_x, green_y, :] = [0, 0, 0]

        img = PIL.Image.fromarray(arr)
    except FileNotFoundError as e:
        print(f'WARNING cannot find {img_path}, using blank image - {e}', file=sys.stderr)
        img = PIL.Image.new(mode='P', size=(512, 512), color=(255, 255, 255))

    return img

if __name__ == '__main__':
    pngs_root = 'abc_quant_data/image_cats'
    cats = [
        'flat',
        'electric',
        'free_form',
        'pipe',
        'angular',
        'rounded',
    ]

    names = [
        'Flat', 'Electric', 'FreeForm', 'Tubular', 'Angular', 'Rounded'
    ]

    files = {
        cat: os.listdir(f'{pngs_root}/{cat}') for cat in cats
    }

    imgs = {
        cat: [open_image(f'{pngs_root}/{cat}/{file}') for file in cat_files]
        for cat, cat_files in files.items()
    }
    img_size=512
    t = torchvision.transforms.Compose([
        torchvision.transforms.Resize((img_size, img_size)),
        torchvision.transforms.ToTensor()
    ])
    tensors = {
        cat: [t(img) for img in cat_imgs]
        for cat, cat_imgs in imgs.items()
    }

    selection = []
    for cat, cat_tensors in tensors.items():
        selection += cat_tensors[-5:]

    selection = torch.stack(selection)

    grid = make_grid(selection, nrow=5).permute([1, 2, 0])
    fig, ax = plt.subplots()
    ax.imshow(grid)

    ax.set_yticks(np.arange(len(cats)) * (img_size + 2) + (img_size / 2))
    ax.set_yticklabels(names, Fontsize=12)
    ax.set_xticks([])
    plt.tight_layout()
    plt.savefig('abc_quant_egs.pdf')
    plt.show()

