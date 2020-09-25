import logging
import numpy as np
import os
import os.path as osp
from torch import optim
import torch.nn.functional as F
import math
import helper
import sklearn.metrics as metrics
import torch
import matplotlib.pyplot as plt
import plot_utils


def train_val(
    step,
    model,
    train_loader,
    val_loader,
    num_classes,
    experiment_name,
    args,
    epochs=350,
    checkpoint_dir="./tmp",
    device="cuda:0",
    val_every=1,
):
    device = torch.device(device)
    model.to(device)

    # Create folders for dumping log and checkpoints
    log_filename = osp.join("dump", experiment_name, "log.txt")
    checkpoint_dir = osp.join("dump", experiment_name, "checkpoints")
    img_dir = osp.join("dump", experiment_name, "imgs")
    helper.create_dir(checkpoint_dir)
    helper.create_dir(img_dir)

    # Setup logger
    helper.setup_logging(log_filename)
    logging.info(args)
    logging.info("Experiment name: {}".format(experiment_name))

    # Train/validate
    logging.info(
        "Initial loss must be close to {}".format(-math.log(1.0 / num_classes))
    )
    logging.info(
        "Model has {} trainable parameters".format(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        )
    )
    optimizer = optim.Adam(model.parameters())
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        tloss, tacc = _clf_train_one_epoch(
            step, model, train_loader, optimizer, epoch, device
        )

        if epoch % val_every == 0:
            vloss, vacc = _clf_val_one_epoch(step, model, val_loader, epoch, device)

            helper.save_checkpoint(
                osp.join(checkpoint_dir, f"last.pt"),
                model,
                optimizer,
                None,
                args=args,
                experiment_name=experiment_name,
            )

            if vacc > best_acc:
                best_acc = vacc
                helper.save_checkpoint(
                    osp.join(checkpoint_dir, f"best.pt"),
                    model,
                    optimizer,
                    None,
                    args=args,
                    experiment_name=experiment_name,
                )
    logging.info("Best validation accuracy: {:2.3f}".format(best_acc))
    logging.info("----------------------------------------------------")


def _clf_train_one_epoch(step, model, loader, optimizer, epoch, device):
    model.train()
    total_loss_array = []
    mean_acc_array = []
    train_true = []
    train_pred = []
    for batch_idx, batch in enumerate(loader):
        optimizer.zero_grad()
        logits, labels = step(model, batch, batch_idx, device)
        loss = F.cross_entropy(logits, labels, reduction="mean")
        loss.backward()
        optimizer.step()

        total_loss_array.append(loss.item())
        train_true.append(labels.cpu().numpy())
        preds = logits.max(dim=1)[1]
        train_pred.append(preds.detach().cpu().numpy())

    train_true = np.concatenate(train_true)
    train_pred = np.concatenate(train_pred)
    acc = metrics.accuracy_score(train_true, train_pred)

    avg_loss = np.mean(total_loss_array)

    logging.info(
        "[Train] Epoch {:03} Loss {:2.3f}, Acc {}".format(epoch, avg_loss.item(), acc)
    )

    return avg_loss, acc


def _clf_val_one_epoch(step, model, loader, epoch, device):
    model.eval()
    true = []
    pred = []
    total_loss_array = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            logits, labels = step(model, batch, batch_idx, device)
            loss = F.cross_entropy(logits, labels, reduction="mean")
            total_loss_array.append(loss.item())
            true.append(labels.cpu().numpy())
            preds = logits.max(dim=1)[1]
            pred.append(preds.detach().cpu().numpy())

    true = np.concatenate(true)
    pred = np.concatenate(pred)
    acc = metrics.accuracy_score(true, pred)
    avg_loss = np.mean(total_loss_array)
    logging.info(
        "[Val]   Epoch {:03} Loss {:2.3f}, Acc {}".format(epoch, avg_loss.item(), acc)
    )
    return avg_loss, acc


def test(step, model, loader, device, class_labels=None, experiment_name=""):
    device = torch.device(device)
    model.to(device)
    model.eval()
    true = []
    pred = []
    total_loss_array = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            logits, labels = step(model, batch, batch_idx, device)
            loss = F.cross_entropy(logits, labels, reduction="mean")
            total_loss_array.append(loss.item())
            true.append(labels.cpu().numpy())
            preds = logits.max(dim=1)[1]
            pred.append(preds.detach().cpu().numpy())

    true = np.concatenate(true)
    pred = np.concatenate(pred)
    acc = metrics.accuracy_score(true, pred)
    avg_loss = np.mean(total_loss_array)
    print("Test accuracy: {:2.3f}".format(acc * 100.0))

    # Plot confusion matrix
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(16, 16)
    plot_utils.confusion_matrix(
        ax, true, pred, title=experiment_name, classes=class_labels
    )
    img_dir = osp.join("dump", experiment_name, "imgs")
    helper.create_dir(img_dir)
    plt.savefig(
        osp.join(img_dir, "confusion_matrix.png"), bbox_inches="tight",
    )
    print(
        "Confusion matrix saved to", osp.join(img_dir, "confusion_matrix.png"),
    )
    # plt.show()
    return avg_loss, acc, pred, true
