import argparse
import os

import parse_util
import reconstruction

chamfer_loss = reconstruction.ChamferLoss()


def pc2pc_step(model, batch, batch_idx, device):
    # points, _ = batch
    (bg, points, graphfiles) = batch
    points = points.to(device)
    pred_points, embeddings = model(points.transpose(-1, 1))
    loss = chamfer_loss(points, pred_points) * 1000
    return pred_points, points, embeddings, loss, bg, graphfiles


def solid2pc_step(model, batch, batch_idx, device):
    (bg, points, graph_files) = batch
    feat = bg.ndata["x"].permute(0, 3, 1, 2).to(device)
    points = points.to(device)
    # labels = labels.to(device).squeeze(-1)
    pred_points, embeddings = model(bg, feat)

    # all_graph_files += graph_files
    loss = chamfer_loss(points, pred_points) * 1000
    return pred_points, points, embeddings, loss, bg, graph_files


def experiment_name(args):
    from datetime import datetime

    tokens = [
        "AE",
        args.encoder,
        args.decoder,
        args.dataset,
        f"points_{args.num_points}",
    ]
    if args.encoder == "uvnetsolid":
        tokens.append(f"sqsym_{args.uvnet_sqsym}", )
    if args.latent_dim is not None:
        tokens.append(f"latent_dim_{args.latent_dim}")
    if args.use_tanh is not None:
        tokens.append("tanh_" + ("True" if args.use_tanh else "False"))
    if args.use_timestamp:
        timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        tokens.append(timestamp)
    if len(args.suffix) > 0:
        tokens.append(args.suffix)
    return ".".join(map(str, tokens))


def parse():
    parser = argparse.ArgumentParser("Pointcloud reconstruction experiments")
    parser.add_argument("traintest", choices=("train", "test"))
    train_args = parser.add_argument_group("train")
    train_args = parse_util.add_train_args(train_args)

    train_args.add_argument(
        "--encoder",
        choices=("pointnet", "uvnetsolid"),
        default="pointnet",
        help="Encoder to use",
    )
    train_args.add_argument(
        "--decoder",
        choices=("pointmlp",),
        default="pointmlp",
        help="Pointcloud decoder to use",
    )
    train_args.add_argument(
        "--dataset",
        choices=("abc"),
        default="abc",
        help="Dataset to train on",
    )
    train_args.add_argument(
        "--num_points", type=int, default=1024, help="Number of points to decode"
    )
    train_args.add_argument(
        "--latent_dim", type=int, default=1024, help="Dimension of latent space"
    )
    train_args.add_argument(
        "--use_tanh",
        action="store_true",
        help="Whether to use tanh in final layer of decoder",
    )
    train_args.add_argument(
        "--npy_dataset_path",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ABC", "pointclouds"),
        help="Path to pointcloud dataset when encoder takes in solids",
    )
    train_args.add_argument(
        "--split_suffix", type=str, default="", help="Suffix for dataset split folders"
    )
    train_args.add_argument(
        "--uvnet_sqsym",
        type=float,
        default=0.3,
        help="Probability of applying square symmetry transformation to uv domain",
    )
    test_args = parser.add_argument_group("test")
    test_args = parse_util.add_test_args(test_args)
    args, _ = parser.parse_known_args()
    return args


def get_model(args):
    from networks import models

    args.latent_dim = 1024
    args.use_tanh = True

    if args.encoder == "uvnetsolid" and args.decoder == "pointmlp":
        return (
            models.UVNetSolid2PointsAutoEnc(
                ae_latent_dim=args.latent_dim if args.latent_dim is not None else 1024,
                # num_out_points=args.num_points,
                use_tanh=args.use_tanh if args.latent_dim is not None else True,
            ),
            solid2pc_step,
        )
    if args.encoder == "pointnet" and args.decoder == "pointmlp":
        return (
            models.Points2PointsAutoEnc(
                ae_latent_dim=args.latent_dim if args.latent_dim is not None else 1024,
                num_out_points=args.num_points,
                use_tanh=args.use_tanh if args.latent_dim is not None else True,
            ),
            pc2pc_step,
        )
    print(f"No model found with encoder: {args.encoder} and decoder: {args.decoder}")
    raise NotImplementedError


def get_dataset(split, args):
    from datasets.abcdataset import ABCDatasetWithPointclouds

    if args.decoder in ("pointmlp",):
        if args.dataset == "abc":
            return ABCDatasetWithPointclouds(
                bin_root_dir=args.dataset_path,
                npy_root_dir=args.npy_dataset_path,
                split=split,
                num_points=args.num_points,
                apply_square_symmetry=args.uvnet_sqsym,
            )
    print(
        f"No dataset named {args.dataset}, found for given encoder+decoder: {args.encoder}+{args.decoder}"
    )
    raise NotImplementedError


def main():
    args = parse()

    device = "cuda:" + str(args.device)

    if args.traintest == "train":
        print(args)
        exp_name = experiment_name(args)
        print(f"Experiment name: {exp_name}")
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
        args.latent_dim = 1024
        args.use_tanh = True
        state = helper.load_checkpoint(args.state)
        print(state["args"])
        test_dset = get_dataset("all", args)
        test_loader = test_dset.get_dataloader(args.batch_size, shuffle=False)
        model, step = get_model(state["args"])
        model = model.to(device)
        model.load_state_dict(state["model"])

        exp_name = experiment_name(state["args"])

        # Test pointcloud reconstruction
        test_loss = reconstruction.test_pc(
            step, model, test_loader, device, args.grams_path, experiment_name=exp_name
        )
        print("Chamfer loss: {:2.3f}".format(test_loss))


if __name__ == "__main__":
    main()
