UVStyle-Net: Unsupervised Few-shot Learning of 3D Style Similarity Measure for B-Reps
=====================================================================================

This repository contains the authors' implementation of
[UVStyle-Net: Unsupervised Few-shot Learning of 3D Style Similarity Measure for B-Reps](https://arxiv.org/abs/2105.02961).

### Contents

1. [About UVStyle-Net](#About-UVStyle-Net)
2. [Citing this Work](#Citing-this-Work)
3. [Quickstart](#Quickstart)
    1. [Environment Setup](#Environment-Setup)
    2. [Download the Data & Pre-trained Model](#Download-the-Data-&-Pre-trained-Model)
    3. [Explore our Interactive Dashboards](#Explore-our-Interactive-Dashboards)
        - [Top-k Queries](#Top-k-Queries)
        - [Visualize Style Loss Gradients](#Visualize-Style-Loss-Gradients)
        - [Optimize Layer Weights](#Optimize-Layer-Weights)
4. [Full Datasets](#Full-Datasets)
    1. [SolidLETTERS](#SolidLETTERS)
    2. [ABC](#ABC)
5. [Other Experiments](#Other-Experiments)
    1. [Linear Probes](#Linear-Probes)
    2. [Precision@10 for Few-shot Learning](#Precision@10-for-Few-shot-Learning)
    3. [Classification on ABC Subsets](#Classification-on-ABC-Subsets)
    4. [Ablation](#Ablation)
6. [Using Your Own Data](#Using-Your-Own-Data)
    1. [Feature Pipeline for B-Rep to DGL](#Feature-Pipeline-for-B-Rep-to-DGL)
    2. [Training the Model](#Training-the-Model)
    3. [Compute the Gram Matrices](#Training-the-Model)
7. [License](#License)

## About UVStyle-Net

![Overview of UVStyle-Net](demo_imgs/overview.png?raw=true "Overview of UVStyle-Net")

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

### Explore our Interactive Dashboards

All dashboards use streamlit and can be run from the project root.

#### Top-k

Experiment with manually adjusting the layer weights used to compute the style loss,
and observe their effect on the nearest neighbours. Available for SolidLETTERS and ABC datasets.

```bash
$ streamlit run dashboards/top_k.py
```

![Top-k Dashboard](demo_imgs/top-k.png?raw=true "Top-k Dashboard")

#### Visualize Style Loss Gradients

Visualize the xyz position gradients of the style loss between a pair of solids.
Black lines indicate the direction and magnitude of the gradient of the loss with
respect to each of the individual sampled points (i.e. which direction to move each
sampled point to better match the style between the solids). Available for SolidLETTERS.

```bash
$ streamlit run dashboards/visualize_style_loss.py
```

![Gradients Visualization](demo_imgs/gradients.gif?raw=true "Gradients Visualization")

#### Optimize Layer Weights

Experiment with different positive and negative examples for few-shot optimization of
the user defined style loss. Select negatives manually, or use randomly drawn examples
from the remaining dataset. Available for SolidLETTERS and ABC datasets.

```bash
streamlit run dashboards/few_shot_optimization.py
```

![Few-shot Optimization](demo_imgs/few_shot.png?raw=true "Few-shot Optimization")

## Full Datasets

## Other Experiments

### Linear Probes

Ensure you have the Gram matrices for SolidLETTERS subset in 
`PROJECT_ROOT/data/SolidLETTERS/grams/subset`.

```text
$ python experiments/linear_probes.py
```

### Precision@10 for Few-shot Learning

Ensure you have the Gram matrices for the complete SolidLETTERS test set in
`PROJECT_ROOT/data/SolidLETTERS/grams/all`.

First perform the optimizations and hits@10 scoring (if you run into memory problems
please reduce `--num_threads`): 

```text
$ python experiments/font_selection_optimize.py

usage: font_selection_optimize.py [-h] [--exp_name EXP_NAME]
                                  [--num_threads NUM_THREADS]

optional arguments:
  -h, --help            show this help message and exit
  --exp_name EXP_NAME   experiment name - results for each font/trial will be
                        saved into this directory (default: SolidLETTERS-all)
  --num_threads NUM_THREADS
                        number of concurrent threads (default: 6)
```
 
Next collate all results and produce the heatmaps (figures will be saved to
the experiments directory):

```text
$ python experiments/font_selection_optimize_collate_and_plot.py

usage: font_selection_optimize_collate_and_plot.py [-h] [--exp_name EXP_NAME]

optional arguments:
  -h, --help           show this help message and exit
  --exp_name EXP_NAME  experiment name - font scores will be read fromthis
                       directory (default: SolidLETTERS-all)
```

### Classification on ABC Subsets

Ensure you have the UVStyle-Net Gram matrices for the complete ABC dataset in
`PROJECT_ROOT/data/ABC/grams/all` as well as the PSNet* Gram matrices in
`PROJECT_ROOT/psnet_data/ABC/grams/all`. Finally, you will need the labeled subset
pngs in `PROJECT_ROOT/data/ABC/labeled_pngs`.

First perform the logistic regression and log the results for each trial (if
you run into memory problems please reduce `--num_threads`):

```text
$ python experiments/abc_logistic_regression.py

usage: abc_logistic_regression.py [-h] [--num_threads NUM_THREADS]

optional arguments:
  -h, --help            show this help message and exit
  --num_threads NUM_THREADS
                        number of concurrent threads (default: 5)
```

Next collate all results and produce the comparison table:

```text
$ python experiments/abc_logistic_regression_collate_scores.py
```

### Ablation

#### Normalization Comparison & Content Embedding

Ensure you have all versions of  Gram matrices for the complete SolidLETTERS
test set (`all_raw`, `all_inorm_only`, `all_fnorm_only`, `all`) - see
[above](#SolidLETTERS).
Then run the logistic regression probes on each version of the gram and log
the results:

```text
python experiments/compare_normalization.py
```

Next collate all results and produce the comparison chart (plot saved to the
experiments directory):

```text
python experiments/compare_normalization_plot.py
```

#### Dimension Reduction Probes

First run the PCA/probes for each dimension:

```text
$ python experiments/dimension_reduction_probes.py
```

Next process the results and create the plot (figure will be saved to the experiments
directory):

```text
$ python experiments/dimension_reduction_plot.py

usage: dimension_reduction_plot.py [-h] [--include_content]

optional arguments:
  -h, --help         show this help message and exit
  --include_content  include content embeddings in plot (default: False)
```

## Using Your Own Data

### Feature Pipeline for B-Rep to DGL

### Training the Model

### Compute the Gram Matrices

## License
Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg