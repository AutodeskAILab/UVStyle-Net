# UVNet

This is the code base for the UV-Net project and includes scripts for reproducing experiments from the paper and generating the WireMNIST and SolidMNIST datasets.
More details about the project can be found in:

- Autodesk Wiki Page with draft of the initiative: https://wiki.autodesk.com/display/ARES/NURBS-Net
- Pre-print paper: https://arxiv.org/abs/2006.10211

## Scripts

- Classification: `train_test_clf.py`
- Pointcloud reconstruction from solids: `train_test_recon_pc.py`
- Image reconstruction from curve networks: `train_test_recon_img.py`


Details about arguments can be found by running `python script_name.py -h`

## Grams

Run test with:

```shell script
python train_test_recon_pc.py test \
                              --dataset_path /home/ubuntu/abc/bin \
                              --dataset abc \
                              --npy_dataset_path /home/ubuntu/abc/pointclouds \
                              --state PATH_TO_SAVED_MODEL
```

## Data generation

The program for data generation is written in C++ and depends on ASM.
Instructions for setting up ASM can be found here: 

TODO