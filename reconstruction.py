import helper
import logging
import numpy as np
import os
import os.path as osp
import torch
from torch import distributions as dist
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import pandas as pd


def compute_activation_stats(bg, layer, activations):
    grams = []
    for graph_activations in torch.split(activations, bg.batch_num_nodes().tolist()):
        # F = num faces
        # d = num filters/dimensions
        # graph_activations shape: F x d x 10 x 10
        x = graph_activations.flatten(start_dim=2)  # x shape: F x d x 100
        x = torch.cat(list(x), dim=-1)  # x shape: d x 100F
        inorm = torch.nn.InstanceNorm1d(x.shape[0])
        x = inorm(x.unsqueeze(0)).squeeze()
        img_size = x.shape[-1]  # img_size = 100F
        gram = torch.matmul(x, x.transpose(0, 1)) / img_size
        grams.append(gram.flatten())
    return torch.stack(grams).detach().cpu()


def log_activation_stats(bg, all_layers_activations):
    stats = {layer: compute_activation_stats(bg, layer, activations)
             for layer, activations in all_layers_activations.items()}
    return stats


class ChamferLoss(nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()

    def forward(self, preds, gts):
        P = self.batch_pairwise_dist(gts, preds)
        mins, _ = torch.min(P, 1)
        loss_1 = torch.mean(mins)
        mins, _ = torch.min(P, 2)
        loss_2 = torch.mean(mins)

        return 0.5 * (loss_1 + loss_2)

    def batch_pairwise_dist(self, x, y):
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        if self.use_cuda:
            dtype = torch.cuda.LongTensor
        else:
            dtype = torch.LongTensor
        diag_ind_x = torch.arange(0, num_points_x).type(dtype)
        diag_ind_y = torch.arange(0, num_points_y).type(dtype)
        # brk()
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = rx.transpose(2, 1) + ry - 2 * zz
        return P


chamfer_dist = ChamferLoss()


def train_val(
        step,
        model,
        train_loader,
        val_loader,
        experiment_name,
        args,
        epochs=350,
        checkpoint_dir="./tmp",
        device="cuda:0",
        val_every=1,
):
    # Create directories for checkpoints and logging
    log_filename = osp.join("dump", experiment_name, "log.txt")
    checkpoint_dir = osp.join("dump", experiment_name, "checkpoints")
    img_dir = osp.join("dump", experiment_name, "imgs")
    helper.create_dir(checkpoint_dir)
    helper.create_dir(img_dir)
    # Setup logger
    helper.setup_logging(log_filename)
    logging.info("Experiment name: {}".format(experiment_name))

    logging.info(
        "Model has {} trainable parameters".format(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        )
    )
    optimizer = optim.Adam(model.parameters())

    iteration = 0
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        tloss = _train_one_epoch(
            step, model, train_loader, optimizer, epoch, iteration, device
        )
        if epoch % val_every == 0:
            test_loss = _val_one_epoch(step, model, val_loader, epoch, device)

            helper.save_checkpoint(
                osp.join(checkpoint_dir, f"last.pt"), model, optimizer, None, args=args,
            )
            if best_loss > test_loss:
                best_loss = test_loss
                helper.save_checkpoint(
                    osp.join(checkpoint_dir, f"best.pt"),
                    model,
                    optimizer,
                    None,
                    args=args,
                )
    logging.info("Best validation loss {}".format(best_loss))


def _val_one_epoch(step, model, loader, epoch, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            pred_points, gt_points, embeddings, loss = step(
                model, batch, batch_idx, device
            )
            losses.append(loss.item())
    avg_loss = np.mean(losses)
    logging.info("[Val]   Epoch {:03} Loss {:2.3f}".format(epoch, avg_loss.item()))
    return avg_loss


def _train_one_epoch(step, model, loader, optimizer, epoch, iteration, device):
    model.train()
    losses = []
    for batch_idx, batch in enumerate(loader):
        iteration = iteration + 1
        optimizer.zero_grad()
        pred_points, gt_points, embeddings, loss = step(model, batch, batch_idx, device)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if iteration % 200 == True:
            avg_loss = np.mean(losses)
            logging.info(
                "[Train] Epoch {:03}, Iteration {:04}, Loss {:2.3f}".format(
                    epoch, iteration, avg_loss.item()
                )
            )

    avg_loss = np.mean(losses)
    logging.info("[Train] Epoch {:03} Loss {:2.3f}".format(epoch, avg_loss.item()))

    return avg_loss


def test_pc(step, model, loader, device, experiment_name, save_pointclouds=True):
    import experiments

    img_dir = osp.join("dump", experiment_name, "imgs")
    helper.create_dir(img_dir)
    model.eval()
    losses = []
    with torch.no_grad():
        stats = {}
        graph_files = []
        for batch_idx, batch in enumerate(loader):
            print('batch:', batch_idx)
            pred_points, gt_points, embeddings, loss, surface_activations, graph_activations, bg, graph_files_batch = step(
                model, batch, batch_idx, device
            )
            for activations in [model.surf_encoder.activations, model.graph_encoder.activations]:
                batch_stats = log_activation_stats(bg, activations)
                for layer, batch_layer_stats in batch_stats.items():
                    if layer in stats.keys():
                        stats[layer].append(batch_layer_stats)
                    else:
                        stats[layer] = [batch_layer_stats]
            graph_files += graph_files_batch
            if save_pointclouds:
                filename = [
                    osp.split(
                        loader.dataset.pc_files[loader.batch_size * batch_idx + i]
                    )[1]
                    for i in range(loader.batch_size)
                ]
                experiments.visualize_pc(pred_points, gt_points, filename, img_dir)
                csv_file = osp.join(
                    img_dir,
                    osp.split(loader.dataset.pc_files[loader.batch_size * batch_idx])[1]
                    + ".csv",
                )
                np.savetxt(
                    csv_file,
                    pred_points[0].detach().cpu().numpy(),
                    delimiter=",",
                    header="x,y,z",
                )
                # print(f"Saving csv: {csv_file}")

            losses.append(loss.item())
    avg_loss = np.mean(losses)
    print('writing stats...')
    all_stats = {}
    for layer, layer_stats in stats.items():
        # gram = zip(*layer_stats)
        all_stats[layer] = {
            'gram': torch.cat(layer_stats),
        }
    out_dir = 'analysis/uvnet_data/abc_all'
    for i, (layer, layer_stats) in enumerate(all_stats.items()):
        grams = layer_stats['gram'].numpy()
        np.save(out_dir + f'/{i}_{layer}_grams', grams)

    all_graph_files = list(map(lambda file: file.split('/')[-1], graph_files))
    pd.DataFrame(all_graph_files).to_csv(out_dir + '/graph_files.txt', index=False, header=None)
    print('done writing stats')
    return avg_loss


def test_img(step, model, loader, device, experiment_name, save_images=True):
    import experiments

    img_dir = osp.join("dump", experiment_name, "imgs")
    helper.create_dir(img_dir)
    model.eval()
    losses = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            pred_images, gt_images, embeddings, loss = step(
                model, batch, batch_idx, device
            )
            if save_images:
                p_r = dist.Bernoulli(logits=pred_images)
                pred_images = p_r.probs  # > 0.2
                # print(p_r.probs , pred_out)
                # raise "err"
                # print(pred_out[0].permute(1,2,0).shape)
                im = Image.fromarray(
                    pred_images[0].permute(1, 2, 0).detach().cpu().squeeze(-1).numpy()
                    * 255.0
                )
                im = im.convert("RGB")
                im.save(
                    osp.join(
                        img_dir,
                        f"{osp.split(loader.dataset.image_files[loader.batch_size * batch_idx])[1]}.png",
                    )
                )

            losses.append(loss.item())
    avg_loss = np.mean(losses)
    return avg_loss
