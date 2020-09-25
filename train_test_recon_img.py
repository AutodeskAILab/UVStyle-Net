import argparse
import torch
from torch import distributions as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import helper
import os.path as osp
import logging
import sklearn.metrics as metrics
import numpy as np
import os
import parse_util
import random
import math
import reconstruction
from PIL import Image


def img2img_step(model, batch, batch_idx, device):
    gt_images, labels = batch
    gt_images = gt_images.to(device)
    labels = labels.to(device).squeeze(-1)
    gt_images = gt_images.permute(0, 3, 1, 2)
    pred_images, embedding = model(gt_images)
    loss = F.binary_cross_entropy_with_logits(pred_images, gt_images)
    return pred_images, gt_images, embedding, loss


def curve2img_step(model, batch, batch_idx, device):
    bg, images, labels = batch
    feat = bg.ndata["x"].permute(0, 2, 1).to(device)
    images = images.to(device)
    labels = labels.to(device).squeeze(-1)
    images = images.permute(0, 3, 1, 2)
    pred_images, embedding = model(bg, feat)
    loss = F.binary_cross_entropy_with_logits(pred_images, images)
    return pred_images, images, embedding, loss


def _save_img(torch_image, img_dir, filename):
    p_r = dist.Bernoulli(logits=torch_image)
    torch_image = p_r.probs  # > 0.2
    im = Image.fromarray(
        torch_image.permute(1, 2, 0).detach().cpu().squeeze(-1).numpy() * 255.0
    )
    im = im.convert("RGB")
    im.save(osp.join(img_dir, filename))


def experiment_name(args):
    from datetime import datetime

    tokens = [
        "AE",
        args.encoder,
        args.decoder,
        args.dataset,
    ]
    if args.encoder == "uvnetsolid":
        tokens.append(f"sqsym_{args.uvnet_sqsym}",)
    if args.use_timestamp:
        timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        tokens.append(timestamp)
    if len(args.suffix) > 0:
        tokens.append(args.suffix)
    return ".".join(map(str, tokens))


def parse():
    parser = argparse.ArgumentParser("Image reconstruction experiments")
    parser.add_argument("traintest", choices=("train", "test"))
    train_args = parser.add_argument_group("train")
    train_args = parse_util.add_train_args(train_args)

    train_args.add_argument(
        "--encoder",
        choices=("uvnetcurve", "uvnetbeziercurve", "svgvae"),
        default="uvnetcurve",
        help="Encoder to use",
    )
    train_args.add_argument(
        "--decoder", choices=("svgvae",), default="svgvae", help="Image decoder to use",
    )
    train_args.add_argument(
        "--dataset", choices=("wiremnist",), default=None, help="Dataset to train on",
    )
    train_args.add_argument(
        "--shape_type",
        type=str,
        default="upper",
        help="Upper or lowercase alphabets (only for WireMNIST)",
    )
    train_args.add_argument(
        "--img_dataset_path",
        type=str,
        default=None,
        help="Path to image dataset when encoder takes in curve-networks",
    )
    train_args.add_argument(
        "--split_suffix", type=str, default="", help="Suffix for dataset split folders"
    )
    train_args.add_argument(
        "--uvnet_linesym",
        type=float,
        default=0.3,
        help="Probability of applying line symmetry transformation to uv domain (applicable only to WireMNIST)",
    )

    test_args = parser.add_argument_group("test")
    test_args = parse_util.add_test_args(test_args)
    test_args.add_argument(
        "--cluster", action="store_true", help="Perform clustering in latent space"
    )
    test_args.add_argument(
        "--retrieval", action="store_true", help="Perform retrieval in latent space"
    )
    test_args.add_argument(
        "--retrieval_split",
        default="test",
        help="Dataset split for retrieval (train/val/test/all)",
    )
    args, _ = parser.parse_known_args()
    return args


def get_model(args):
    from networks import models

    if args.encoder == "uvnetcurve" and args.decoder == "svgvae":
        return (
            models.UVNetCurve2ImageAutoEnc(input_channels="xyz_only"),
            curve2img_step,
        )
    if args.encoder == "uvnetbeziercurve" and args.decoder == "svgvae":
        return (
            models.UVNetBezierCurve2ImageAutoEnc(input_channels="xyz_only"),
            curve2img_step,
        )
    if args.encoder == "svgvae" and args.decoder == "svgvae":
        return models.SVGVAEImage2ImageAutoEnc(), img2img_step
    print(f"No model found with encoder: {args.encoder} and decoder: {args.decoder}")
    raise NotImplementedError


def get_dataset(split, args):
    from datasets.wiremnist import WireMNISTWithImages

    if args.encoder in ("uvnetcurve", "uvnetbeziercurve") and args.decoder in (
        "svgvae",
    ):
        if args.dataset == "wiremnist":
            return WireMNISTWithImages(
                root_dir=args.dataset_path,
                img_root_dir=args.img_dataset_path,
                split=split,
                shape_type=args.shape_type,
                apply_line_symmetry=args.uvnet_linesym,
                size_percentage=0.2,
            )
    if args.encoder in ("svgvae",) and args.decoder in ("svgvae",):
        if args.dataset == "wiremnist":
            # TODO(pradeep): create an images only version of WireMNIST dataset
            return WireMNISTWithImages(
                root_dir=args.dataset_path,
                img_root_dir=args.img_dataset_path,
                split=split,
                shape_type=args.shape_type,
                size_percentage=0.2,
            )
    print(
        f"No dataset named {args.dataset}, found for given encoder+decoder: {args.encoder}+{args.decoder}"
    )
    raise NotImplementedError


def main():
    args = parse()
    print(args)

    device = "cuda:" + str(args.device)

    exp_name = experiment_name(args)
    print(f"Experiment name: {exp_name}")

    if args.traintest == "train":
        train_dset = get_dataset("train", args)
        train_loader = train_dset.get_dataloader(
            batch_size=args.batch_size, shuffle=True
        )
        val_dset = get_dataset("val", args)
        val_loader = val_dset.get_dataloader(batch_size=args.batch_size, shuffle=False)

        model, step = get_model(args)
        model = model.to(device)
        reconstruction.train_val(
            step=step,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            experiment_name=exp_name,
            args=args,
            epochs=args.epochs,
            val_every=5,
        )
    else:
        import helper
        import experiments

        state = helper.load_checkpoint(args.state)

        test_dset = get_dataset("test", state["args"])
        test_loader = test_dset.get_dataloader(32, shuffle=False)
        model, step = get_model(state["args"])
        model = model.to(device)
        model.load_state_dict(state["model"])

        exp_name = experiment_name(state["args"])

        # Test pointcloud reconstruction
        test_loss = reconstruction.test_img(
            step, model, test_loader, device, experiment_name=exp_name, save_images=True
        )
        print("BCE loss: {:2.3f}".format(test_loss))


if __name__ == "__main__":
    main()
