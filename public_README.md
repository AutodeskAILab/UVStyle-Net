UVStyle-Net: Unsupervised Few-shot Learning of 3D Style Similarity Measure for B-Reps
=====================================================================================

This repository contains the authors' implementation of
[UVStyle-Net: Unsupervised Few-shot Learning of 3D Style Similarity Measure for B-Reps](https://arxiv.org/abs/2105.02961).

## About UVStyle-Net

## Citing this Work

If you use any of the code or techniques from the paper, please cite the following:

> Meltzer, P., Shayani, H., Khasahmadi, A., Jayaraman, P. K., Sanghi, A., & Lambourne, J. (2021). UVStyle-Net: Unsupervised Few-shot Learning of 3D Style Similarity Measure for B-Reps. _arXiv preprint arXiv:2105.02961_.

```text
@misc{meltzer2021uvstylenet,
      title={UVStyle-Net: Unsupervised Few-shot Learning of 3D Style Similarity Measure for B-Reps}, 
      author={Peter Meltzer and Hooman Shayani and Amir Khasahmadi and Pradeep Kumar Jayaraman and Aditya Sanghi and Joseph Lambourne},
      year={2021},
      eprint={2105.02961},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Quickstart

### Environment Setup

We recommend using a virtual environment such as conda:

```bash
$ conda create --name uvstylenet python=3.7
WINDOWS:       $ activate uvstylenet
LINUX, macOS:  $ source activate uvstylenet
```

- swap `dgl-cu101` in `requirmenets.txt:3` for the correct cuda version for your system. i.e. `cdl-cu102`, `dgl-cu100`, etc. (for cpu only use `dgl`)
- for gpu use you may need to install cudatoolkit/set environment variable `LD_LIBRARY_PATH` if you have not done so already

Install the remaining requirements:

```bash
$ pip install -r requirements.txt
```

### Download the Data & Pre-trained Models

To get started quickly and interact with the models, we recommend downloading only
the pre-computed Gram matrices for the SolidLETTERS test set along with the pre-trained
SolidLETTERS model and the SolidLETTERS test set meshes (to assist visualization).

```bash
# TODO
```

OPTIONAL: If you wish to modify the normalisation or the way in which the Grams
are computed you will need the pre-computed DGL binary files from which the Gram
matrices are computed (details on how to do this [below](#Compute-the-Gram-Matrices)):

```bash
# TODO
```

OPTIONAL: If you wish to retrain UVNet, or use SolidLETTERS for training your own models, the complete dataset
is available in DGL binary files, mesh (edge numbers), point cloud, :

```bash
# TODO
```

OPTIONAL: For the ABC dataset, 

### Top-k Queries

### Visualize Style Loss Gradients

### Optimize Layer Weights (Few-shot Learning)

## Other Experiments from Paper

### Linear Probes (Fig. 3)

### Precision@10 for Few-shot Learning (Fig. 8)

### Classification on ABC Subsets (Table 2)

### Ablation (Fig. 10 & 11)

## Custom Datasets

### Feature Pipeline for B-Rep to DGL

### Compute the Gram Matrices

## License
Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg