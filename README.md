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
5. [Other Experiments](#Other-Experiments)
    1. [Linear Probes](#Linear-Probes)
    2. [Precision@10 for Few-shot Learning](#Precision@10-for-Few-shot-Learning)
    3. [Classification on ABC Subsets](#Classification-on-ABC-Subsets)
    4. [Ablation](#Ablation)
        - [Normalization Comparison & Content Embedding](#Normalization-Comparison-&-Content-Embedding)
        - [Dimension Reduction Probes](#Dimension-Reduction-Probes)
6. [Using Your Own Data](#Using-Your-Own-Data)
    1. [Feature Pipeline for B-Rep to DGL](#Feature-Pipeline-for-B-Rep-to-DGL)
    2. [Training the Model](#Training-the-Model)
    3. [Compute the Gram Matrices](#Training-the-Model)
7. [License](#License)

## About UVStyle-Net

<p align="center">
    <img src="demo_imgs/overview.png?raw=true" alt="Overview of UVStyle-Net">
</p>

UVStyle-Net is an unsupervised style similarity learning method for Boundary
Representations (B-Reps)/CAD models, which can be tailored to an end-user's
interpretation of style by supplying only a few examples.

The way it works can be summarized as follows:

1. Train the B-Rep encoder (we use [UV-Net](https://github.com/AutodeskAILab/UV-Net)
 without edge features) using content classification (supervised), or point cloud
 reconstruction (unsupervised)
2. Extract the Gram matrices of the activations in each layer of the encoder
3. Define the style distance for two solids as a weighted sum of distances (Euclidean, cosine, etc.)
 between each layer's Gram matrix using uniform weights
4. OPTIONAL: Determine the best weights for the style distance according to a few
 user selected examples that share a common style

For a 5 minute overview, take a look at our ICCV presentation video/poster below, or for full details, see the [paper](https://arxiv.org/abs/2105.02961).

[UVStyle-Net ICCV Poster.pdf](https://github.com/AutodeskAILab/UVStyle-Net/files/7452640/UVStyle-Net.ICCV.2021.5.min.pdf)

[![Screenshot 2021-11-01 at 11 08 30](https://user-images.githubusercontent.com/23616959/139662655-11eb6cbd-347f-4f92-a580-49647f4c0ebf.png)](https://www.youtube.com/watch?v=E6fMVjyqtGs)

	
## Citing this Work

If you use any of the code or techniques from the paper, please cite the following:

> Meltzer, Peter, Hooman Shayani, Amir Khasahmadi, Pradeep Kumar Jayaraman, Aditya Sanghi, and Joseph Lambourne. ‘UVStyle-Net: Unsupervised Few-Shot Learning of 3D Style Similarity Measure for B-Reps’. In IEEE/CVF International Conference on Computer Vision (ICCV), 2021. http://arxiv.org/abs/2105.02961.

```text
@inproceedings{meltzer_uvstyle-net_2021,
	title = {{UVStyle}-{Net}: {Unsupervised} {Few}-shot {Learning} of {3D} {Style} {Similarity} {Measure} for {B}-{Reps}},
	shorttitle = {{UVStyle}-{Net}},
	url = {http://arxiv.org/abs/2105.02961},
	booktitle = {{IEEE}/{CVF} {International} {Conference} on {Computer} {Vision} ({ICCV})},
	author = {Meltzer, Peter and Shayani, Hooman and Khasahmadi, Amir and Jayaraman, Pradeep Kumar and Sanghi, Aditya and Lambourne, Joseph},
	year = {2021},
	note = {arXiv: 2105.02961},
	keywords = {Computer Science - Computer Vision and Pattern Recognition, 68T07, 68T10, Computer Science - Machine Learning, I.3.5, I.5.1},
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

- swap `dgl` in `requirmenets.txt:3` for the correct cuda version for your system. i.e. `cdl-cu102`, `dgl-cu100`, etc. (for cpu only use `dgl`)
- for gpu use you may need to install cudatoolkit/set environment variable `LD_LIBRARY_PATH` if you have not done so already

Install the remaining requirements:

```bash
$ pip install -r requirements.txt
```

To use the download script requires a 7z extractor to be installed:

```bash
# Linux
$ sudo apt-get install p7zip-full

# MacOS
$ brew install p7zip
```

or for windows, visit: <https://www.7-zip.org/download.html>.

### Download the Data & Pre-trained Models

To get started quickly and interact with the models, we recommend downloading only
the pre-computed Gram matrices for the SolidLETTERS test set along with the pre-trained
SolidLETTERS model, the test set dgl binary files, and the test set meshes (to assist visualization).
You will need a little bit over 7GB of space for all this data. 
All of these necessary files can be downloaded into the correct directory structure with:

```bash
$ python download_data.py quickstart
```

### Explore our Interactive Dashboards

All dashboards use streamlit and can be run from the project root.

#### Top-k

Experiment with manually adjusting the layer weights used to compute the style loss,
and observe their effect on the nearest neighbours. Available for SolidLETTERS and ABC datasets.

```bash
$ streamlit run dashboards/top_k.py
```

<p align="center">
    <img src="demo_imgs/top-k.png?raw=true" alt="Top-k Dashboard"
    width="600px">
</p>

#### Visualize Style Loss Gradients

Visualize the xyz position gradients of the style loss between a pair of solids.
Black lines indicate the direction and magnitude of the gradient of the loss with
respect to each of the individual sampled points (i.e. which direction to move each
sampled point to better match the style between the solids). Available for SolidLETTERS.

```bash
$ streamlit run dashboards/visualize_style_loss.py
```

<p align="center">
    <img src="demo_imgs/gradients.gif?raw=true" alt="Gradients Visualization"
    width="600px">
</p>

#### Optimize Layer Weights

Experiment with different positive and negative examples for few-shot optimization of
the user defined style loss. Select negatives manually, or use randomly drawn examples
from the remaining dataset. Available for SolidLETTERS and ABC datasets.

```bash
streamlit run dashboards/few_shot_optimization.py
```

<p align="center">
    <img src="demo_imgs/few_shot.png?raw=true" alt="Few-shot Optimization" 
    width="600px">
</p>

## Full Datasets

To download the full datasets in all formats you will need approximately 111 GB (+ approx. 10 GB for
temporary files during extraction). A breakdown is given below:

- SolidLETTERS (46 GB for complete set)
  - grams: 7.5 GB
  - dgl binary files: 7.7 GB
  - mesh: 9.9 GB
  - point clouds: 3.3 GB
  - smt: 18 GB
  - pngs: 84 MB
- ABC (65 GB for complete set)
  - grams: 12.6 GB (uv-net: 8 GB, psnet: 4.6 GB)
  - dgl binary files: 20 GB
  - mesh: 13 GB
  - point clouds: 2.3 GB
  - smt: 18 GB
  - pngs: 513 MB
  - subset labels: 16 MB

All data/parts you require may be downloaded using the following script:

```text
$ python download_data.py

usage: download_data.py [-h] [--all] [--grams] [--dgl] [--mesh] [--pc] [--smt]
                        [--pngs] [--labels]
                        {quickstart,models,solid_letters,abc}

positional arguments:
  {quickstart,models,solid_letters,abc}

optional arguments:
  -h, --help            show this help message and exit

solid_letters, abc:
  --all
  --grams
  --dgl
  --mesh
  --pc
  --smt
  --pngs

abc only:
  --labels
```

To only download files necessary for the quickstart above use command `quickstart`. For the pre-trained
models use command `models`. To download SolidLETTERS or ABC datasets, use command
`solid_letters` or `abc` accordingly, followed by the arguments indicating the format(s).

Examples:

```bash
# Pre-trained models
$ python download_data.py models

# Complete SolidLETTERS dataset in all formats
$ python download_data.py solid_letters --all

# ABC mesh and point clouds only
$ python download_data.py abc --mesh --pc
```

## Other Experiments

### Linear Probes

Ensure you have the Gram matrices for SolidLETTERS subset in 
`PROJECT_ROOT/data/SolidLETTERS/grams/subset`.

```text
$ python experiments/linear_probes.py
```

|    | layer   |   linear_probe |   linear_probe_err |
|---:|:--------|---------------:|-------------------:|
|  0 | 0_feats |       0.992593 |          0.0148148 |
|  1 | 1_conv1 |       1        |          0         |
|  2 | 2_conv2 |       1        |          0         |
|  3 | 3_conv3 |       1        |          0         |
|  4 | 4_fc    |       0.977778 |          0.0181444 |
|  5 | 5_GIN_1 |       0.940741 |          0.0181444 |
|  6 | 6_GIN_2 |       0.874074 |          0.0296296 |
|  7 | content |       0.755556 |          0.0377705 |

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

<p align="center">
    <img src="demo_imgs/hits_at_10_scores.png?raw=true" alt="Hits@10 Scores"
    width="600px">
</p>

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

|    | cats                 | UVStyle-Net            | PSNet*             |       diff |
|---:|:---------------------|:-----------------------|:-------------------|-----------:|
|  2 | flat v electric      | **0.789 &#177; 0.034** | 0.746 &#177; 0.038 | 0.0428086  |
|  0 | free_form v tubular  | **0.839 &#177; 0.011** | 0.808 &#177; 0.023 | 0.0308303  |
|  1 | angular v rounded    | **0.805 &#177; 0.010** | 0.777 &#177; 0.020 | 0.0279178  |

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

<p align="center">
    <img src="demo_imgs/ablation_normalization.png?raw=true" alt="Normalization Comparison"
    width="600px">
</p>

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

<p align="center">
    <img src="demo_imgs/ablation_dimension_reduction.png?raw=true" alt="Dimension Reduction"
    width="600px">
</p>

## Using Your Own Data

### Feature Pipeline for B-Rep to DGL

Coming soon.

### Training the Model

#### SolidLETTERS

```text
$ python train_classifier.py -h

usage: train_classifier.py [-h] [--dataset_path DATASET_PATH]
                           [--device DEVICE] [--batch_size BATCH_SIZE]
                           [--seed SEED] [--epochs EPOCHS] [--lr LR]
                           [--optimizer {SGD,Adam}] [--use-timestamp]
                           [--times TIMES] [--suffix SUFFIX]
                           [--nurbs_model_type {cnn,wcnn}]
                           [--nurbs_emb_dim NURBS_EMB_DIM]
                           [--mask_mode {channel,multiply}]
                           [--area_as_channel]
                           [--input_channels {xyz_only,xyz_normals}]
                           [--brep_model_type BREP_MODEL_TYPE]
                           [--graph_emb_dim GRAPH_EMB_DIM]
                           [--classifier_type {linear,non_linear}]
                           [--final_dropout FINAL_DROPOUT]
                           [--size_percentage SIZE_PERCENTAGE]
                           [--apply_square_symmetry APPLY_SQUARE_SYMMETRY]
                           [--split_suffix SPLIT_SUFFIX]

UV-Net Classifier Training Script for Solids

optional arguments:
  -h, --help            show this help message and exit
  --dataset_path DATASET_PATH
                        Path to the dataset root directory
  --device DEVICE       which gpu device to use (default: 0)
  --batch_size BATCH_SIZE
                        batch size for training and validation (default: 128)
  --seed SEED           random seed (default: 0)
  --epochs EPOCHS       number of epochs to train (default: 350)
  --lr LR               learning rate (default: 0.01)
  --optimizer {SGD,Adam}
  --use-timestamp       Whether to use timestamp in dump files
  --times TIMES         Number of times to run the experiment
  --suffix SUFFIX       Suffix for the experiment name
  --nurbs_model_type {cnn,wcnn}
                        Feature extractor for NURBS surfaces
  --nurbs_emb_dim NURBS_EMB_DIM
                        Embedding dimension for NURBS feature extractor
                        (default: 64)
  --mask_mode {channel,multiply}
                        Whether to consider trimming mask as channel or
                        multiply it with computed features
  --area_as_channel     Whether to use area as a channel in the input
  --input_channels {xyz_only,xyz_normals}
  --brep_model_type BREP_MODEL_TYPE
                        Feature extractor for B-rep face-adj graph
  --graph_emb_dim GRAPH_EMB_DIM
                        Embeddings before graph pooling
  --classifier_type {linear,non_linear}
                        Classifier model
  --final_dropout FINAL_DROPOUT
                        final layer dropout (default: 0.3)
  --size_percentage SIZE_PERCENTAGE
                        Percentage of data to use
  --apply_square_symmetry APPLY_SQUARE_SYMMETRY
                        Probability of applying square symmetry transformation
                        to uv domain
  --split_suffix SPLIT_SUFFIX
                        Suffix for dataset split folders
```
#### ABC

```text
$ python train_test_recon_pc.py train                                                                                               for_public ⬆ ✱ ◼

usage: Pointcloud reconstruction experiments [-h]
                                             [--dataset_path DATASET_PATH]
                                             [--device DEVICE]
                                             [--batch_size BATCH_SIZE]
                                             [--epochs EPOCHS]
                                             [--use-timestamp]
                                             [--suffix SUFFIX]
                                             [--encoder {pointnet,uvnetsolid}]
                                             [--decoder {pointmlp}]
                                             [--dataset {solidmnist,abc}]
                                             [--num_points NUM_POINTS]
                                             [--latent_dim LATENT_DIM]
                                             [--use_tanh]
                                             [--npy_dataset_path NPY_DATASET_PATH]
                                             [--split_suffix SPLIT_SUFFIX]
                                             [--uvnet_sqsym UVNET_SQSYM]
                                             [--state STATE] [--no-cuda]
                                             [--seed SEED]
                                             [--grams_path GRAMS_PATH]
                                             {train,test}

positional arguments:
  {train,test}

optional arguments:
  -h, --help            show this help message and exit

train:
  --dataset_path DATASET_PATH
                        Path to the dataset root directory
  --device DEVICE       Which gpu device to use (default: 0)
  --batch_size BATCH_SIZE
                        batch size for training and validation (default: 128)
  --epochs EPOCHS       number of epochs to train (default: 350)
  --use-timestamp       Whether to use timestamp in dump files
  --suffix SUFFIX       Suffix for the experiment name
  --encoder {pointnet,uvnetsolid}
                        Encoder to use
  --decoder {pointmlp}  Pointcloud decoder to use
  --dataset {solidmnist,abc}
                        Dataset to train on
  --num_points NUM_POINTS
                        Number of points to decode
  --latent_dim LATENT_DIM
                        Dimension of latent space
  --use_tanh            Whether to use tanh in final layer of decoder
  --npy_dataset_path NPY_DATASET_PATH
                        Path to pointcloud dataset when encoder takes in
                        solids
  --split_suffix SPLIT_SUFFIX
                        Suffix for dataset split folders
  --uvnet_sqsym UVNET_SQSYM
                        Probability of applying square symmetry transformation
                        to uv domain
```

### Compute the Gram Matrices

#### SolidLETTERS

Gram matrices will be saved into directory given as `--grams_path`:

```text
$ python test_classifier.py

usage: test_classifier.py [-h] [--state STATE] [--no-cuda] [--subset]
                          [--apply_square_symmetry APPLY_SQUARE_SYMMETRY]

UV-Net Classifier Testing Script for Solids

optional arguments:
  -h, --help            show this help message and exit
  --state STATE         PyTorch checkpoint file of trained network.
  --no-cuda             Run on CPU
  --subset              Compute subset only (default: false)
  --grams_path GRAMS_PATH
                        path to save Gram matrices to
  --apply_square_symmetry APPLY_SQUARE_SYMMETRY
                        Probability of applying square symmetry transformation
                        to uv-domain
```

#### ABC

Gram matrices will be saved to directory given as `--grams_path`:
```text
$ python train_test_recon_pc.py test

  usage: Pointcloud reconstruction experiments [-h]
                                               [--dataset_path DATASET_PATH]
                                               [--device DEVICE]
                                               [--batch_size BATCH_SIZE]
                                               [--epochs EPOCHS]
                                               [--use-timestamp]
                                               [--suffix SUFFIX]
                                               [--encoder {pointnet,uvnetsolid}]
                                               [--decoder {pointmlp}]
                                               [--dataset {solidmnist,abc}]
                                               [--num_points NUM_POINTS]
                                               [--latent_dim LATENT_DIM]
                                               [--use_tanh]
                                               [--npy_dataset_path NPY_DATASET_PATH]
                                               [--split_suffix SPLIT_SUFFIX]
                                               [--uvnet_sqsym UVNET_SQSYM]
                                               [--state STATE] [--no-cuda]
                                               [--seed SEED]
                                               [--grams_path GRAMS_PATH]
                                               {train,test}
  
  positional arguments:
    {train,test}
  
  optional arguments:
    -h, --help            show this help message and exit
  
  train:
    --dataset_path DATASET_PATH
                          Path to the dataset root directory
    --device DEVICE       Which gpu device to use (default: 0)
    --batch_size BATCH_SIZE
                          batch size for training and validation (default: 128)
    --epochs EPOCHS       number of epochs to train (default: 350)
    --use-timestamp       Whether to use timestamp in dump files
    --suffix SUFFIX       Suffix for the experiment name
    --encoder {pointnet,uvnetsolid}
                          Encoder to use
    --decoder {pointmlp}  Pointcloud decoder to use
    --dataset {solidmnist,abc}
                          Dataset to train on
    --num_points NUM_POINTS
                          Number of points to decode
    --latent_dim LATENT_DIM
                          Dimension of latent space
    --use_tanh            Whether to use tanh in final layer of decoder
    --npy_dataset_path NPY_DATASET_PATH
                          Path to pointcloud dataset when encoder takes in
                          solids
    --split_suffix SPLIT_SUFFIX
                          Suffix for dataset split folders
    --uvnet_sqsym UVNET_SQSYM
                          Probability of applying square symmetry transformation
                          to uv domain
  
  test:
    --state STATE         PyTorch checkpoint file of trained network.
    --no-cuda             Run on CPU
    --seed SEED           Seed
    --grams_path GRAMS_PATH
                          directory to save Gram matrices to (default:
                          data/ABC/uvnet_grams)
```

## License
Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
