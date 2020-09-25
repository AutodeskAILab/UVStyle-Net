import argparse
import parse_util
from datasets.solidmnist import (
    SolidMNISTWithPointclouds,
    SolidMNIST,
    SolidMNISTPointclouds,
)
from datasets.wiremnist import WireMNIST, WireMNISTWithImages
from datasets.machiningfeature import MachiningFeature, MachiningFeaturePointclouds
from datasets.font_util import save_feature_to_csv
from networks import models
import os.path as osp
import matplotlib.pyplot as plt
import string
import classification


def experiment_name(args) -> str:
    """Generate a name for the experiment based on chosen args"""
    from datetime import datetime

    tokens = ["Clf", args.model, args.dataset]
    if args.model == "pointnet":
        tokens.append(f"points_{args.num_points}")
    elif args.model == "uvnetsolid" or args.model == "uvnetcurve":
        tokens.append(f"channels_{args.uvnet_channels}")
    if args.use_timestamp:
        timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        tokens.append(timestamp)
    if len(args.suffix) > 0:
        tokens.append(args.suffix)
    return ".".join(map(str, tokens))


def point_clf_step(model, batch, batch_idx, device):
    """Train/val step for pointcloud classification"""
    points, labels = batch
    points = points.to(device)
    labels = labels.to(device).squeeze(-1)
    points = points.transpose(-1, 1)
    logits = model(points)
    return logits, labels


def solid_clf_step(model, batch, batch_idx, device):
    """Train/val step for solid classification"""
    bg, labels = batch
    feat = bg.ndata["x"].permute(0, 3, 1, 2).to(device)
    labels = labels.to(device).squeeze(-1)
    logits = model(bg, feat)
    return logits, labels


def wire_clf_step(model, batch, batch_idx, device):
    """Train/val step for wire classification"""
    bg, labels = batch
    feat = bg.ndata["x"].permute(0, 2, 1).to(args.device)
    labels = labels.to(args.device).squeeze(-1)
    logits = model(bg, feat)
    return logits, labels


def img_clf_step(model, batch, batch_idx, device):
    """Train/val step for image classification"""
    _, images, labels = batch
    labels = labels.to(args.device).squeeze(-1)
    images = images.to(args.device)
    images = images.permute(0, 3, 1, 2)
    logits = model(images)
    return logits, labels


def parse():
    parser = argparse.ArgumentParser(
        description="Classification experiments", add_help=True,
    )
    parser.add_argument(
        "traintest",
        choices=("train", "test"),
        default=None,
        help="Whether to train or test",
    )
    train_args = parser.add_argument_group("train")
    train_args = parse_util.add_train_args(train_args)
    train_args.add_argument(
        "--model", type=str, choices=("uvnetsolid", "uvnetcurve", "pointnet", "svgvae"),
    )
    train_args.add_argument(
        "--dataset", type=str, choices=("solidmnist", "wiremnist", "machiningfeature")
    )
    train_args.add_argument(
        "--uvnet_channels",
        type=str,
        choices=("xyz_only", "xyz_normals"),
        default="xyz_only",
    )
    train_args.add_argument(
        "--prob_sqsym",
        type=float,
        default=0.3,
        help="Probability of applying square symmetry transformation to uv domain of solid surfaces (applicable to solid classification only)",
    )
    train_args.add_argument(
        "--prob_linesym",
        type=float,
        default=0.3,
        help="Probability of applying line symmetry transformation to uv domain of wire curves (applicable to wire classification only)",
    )
    train_args.add_argument(
        "--num_points",
        type=int,
        default=1024,
        help="Number of points per pointcloud (applicable for pointcloud classification only)",
    )
    test_args = parser.add_argument_group("test")
    test_args.add_argument(
        "--state",
        type=str,
        default="",
        help="PyTorch checkpoint file of trained network",
    )
    args = parser.parse_args()
    return args


def get_model(num_classes, args):
    if args.model == "uvnetsolid":
        return (
            models.UVNetSolidClassifier(
                num_classes, input_channels=args.uvnet_channels
            ),
            solid_clf_step,
        )
    if args.model == "uvnetcurve":
        return (
            models.UVNetCurveClassifier(
                num_classes, input_channels=args.uvnet_channels
            ),
            wire_clf_step,
        )
    if args.model == "pointnet":
        return models.PointNetClassifier(num_classes=num_classes), point_clf_step
    if args.model == "svgvae":
        return models.SVGVAEImageClassifier(num_classes), img_clf_step
    print(f"No model named {args.model} found")
    raise NotImplementedError


def get_dataset(split, args):
    if args.model in ("uvnetsolid",):
        if args.dataset == "solidmnist":
            return SolidMNIST(
                root_dir=args.dataset_path,
                split=split,
                apply_square_symmetry=args.prob_sqsym,
            )
        if args.dataset == "machiningfeature":
            return MachiningFeature(
                root_dir=args.dataset_path,
                split=split,
                apply_square_symmetry=args.prob_sqsym,
            )
    if args.model in ("uvnetcurve",):
        if args.dataset == "wiremnist":
            return WireMNIST(
                root_dir=args.dataset_path,
                split=split,
                apply_line_symmetry=args.prob_linesym,
            )
    if args.model in ("pointnet",):
        if args.dataset == "solidmnist":
            return SolidMNISTPointclouds(
                root_dir=args.dataset_path, split=split, num_points=args.num_points,
            )
        if args.dataset == "machiningfeature":
            return MachiningFeaturePointclouds(
                root_dir=args.dataset_path, split=split, num_points=args.num_points,
            )
    if args.model in ("svgvae",):
        if args.dataset == "wiremnist":
            return WireMNISTWithImages(root_dir=args.dataset_path, split=split,)
    print(f"No dataset {args.dataset}, found for given model {args.model}")
    raise NotImplementedError


if __name__ == "__main__":
    args = parse()
    print(args)

    device = "cuda:" + str(args.device)

    if args.traintest == "train":
        exp_name = experiment_name(args)

        train_dset = get_dataset("train", args)
        val_dset = get_dataset("val", args)
        train_loader = train_dset.get_dataloader(args.batch_size, shuffle=True)
        val_loader = val_dset.get_dataloader(args.batch_size, shuffle=False)

        model, trainval_step = get_model(train_dset.num_classes, args)
        model = model.to(device)

        classification.train_val(
            step=trainval_step,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=train_loader.dataset.num_classes,
            experiment_name=exp_name,
            epochs=args.epochs,
            args=args,
            device=device,
        )
    else:
        import helper

        state = helper.load_checkpoint(args.state)
        test_dset = get_dataset("test", state["args"])
        test_loader = test_dset.get_dataloader(32, shuffle=False)
        model, test_step = get_model(test_dset.num_classes, state["args"])

        model.load_state_dict(state["model"])
        class_labels = list(string.ascii_lowercase)[: test_dset.num_classes]
        exp_name = experiment_name(state["args"])
        classification.test(
            step=test_step,
            model=model,
            loader=test_loader,
            device=device,
            experiment_name=exp_name,
        )
