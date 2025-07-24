"""
Training functions for HW2P2
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Handle both relative and absolute imports
try:
    from .utils import AverageMeter, accuracy, get_ver_metrics
except ImportError:
    from utils import AverageMeter, accuracy, get_ver_metrics


def train_epoch(model, dataloader, optimizer, lr_scheduler, scaler, device, config):
    """Train for one epoch"""
    model.train()

    # metric meters
    loss_m = AverageMeter()
    acc_m = AverageMeter()

    # Loss function
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    # Progress Bar
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc="Train", ncols=5)

    for i, (images, labels) in enumerate(dataloader):

        optimizer.zero_grad()  # Zero gradients

        # send to cuda
        images = images.to(device, non_blocking=True)
        if isinstance(labels, (tuple, list)):
            targets1, targets2, lam = labels
            labels = (targets1.to(device), targets2.to(device), lam)
        else:
            labels = labels.to(device, non_blocking=True)

        # forward
        with torch.amp.autocast(device_type=device):  # This implements mixed precision. Thats it!
            outputs = model(images)

            # Use the type of output depending on the loss function you want to use
            loss = criterion(outputs["out"], labels)

        scaler.scale(loss).backward()  # This is a replacement for loss.backward()
        scaler.step(optimizer)  # This is a replacement for optimizer.step()
        scaler.update()

        # metrics
        loss_m.update(loss.item())
        if "feats" in outputs:
            acc = accuracy(outputs["out"], labels)[0].item()
        else:
            acc = 0.0
        acc_m.update(acc)

        # tqdm lets you add some details so you can monitor training as you train.
        batch_bar.set_postfix(
            acc="{:.04f}% ({:.04f})".format(acc, acc_m.avg),
            loss="{:.04f} ({:.04f})".format(loss.item(), loss_m.avg),
            lr="{:.04f}".format(float(optimizer.param_groups[0]["lr"])),
        )

        batch_bar.update()  # Update tqdm bar

    # You may want to call some schedulers inside the train function. What are these?
    if lr_scheduler is not None:
        lr_scheduler.step()

    batch_bar.close()

    return acc_m.avg, loss_m.avg


@torch.inference_mode()
def valid_epoch_cls(model, dataloader, device, config):
    """Validation for classification task"""
    model.eval()
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, position=0, leave=False, desc="Val Cls.", ncols=5)

    # metric meters
    loss_m = AverageMeter()
    acc_m = AverageMeter()

    # Loss function
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    for i, (images, labels) in enumerate(dataloader):

        # Move images to device
        images, labels = images.to(device), labels.to(device)

        # Get model outputs
        with torch.inference_mode():
            outputs = model(images)
            loss = criterion(outputs["out"], labels)

        # metrics
        acc = accuracy(outputs["out"], labels)[0].item()
        loss_m.update(loss.item())
        acc_m.update(acc)

        batch_bar.set_postfix(
            acc="{:.04f}% ({:.04f})".format(acc, acc_m.avg), loss="{:.04f} ({:.04f})".format(loss.item(), loss_m.avg)
        )

        batch_bar.update()

    batch_bar.close()
    return acc_m.avg, loss_m.avg


def valid_epoch_ver(model, pair_data_loader, device, config):
    """Validation for verification task"""
    model.eval()
    scores = []
    match_labels = []
    batch_bar = tqdm(total=len(pair_data_loader), dynamic_ncols=True, position=0, leave=False, desc="Val Veri.")

    for i, (images1, images2, labels) in enumerate(pair_data_loader):

        images = torch.cat([images1, images2], dim=0).to(device)
        # Get model outputs
        with torch.inference_mode():
            outputs = model(images)

        feats = F.normalize(outputs["feats"], dim=1)
        feats1, feats2 = feats.chunk(2)
        similarity = F.cosine_similarity(feats1, feats2)
        scores.append(similarity.cpu().numpy())
        match_labels.append(labels.cpu().numpy())
        batch_bar.update()

    scores = np.concatenate(scores)
    match_labels = np.concatenate(match_labels)

    FPRs = ["1e-4", "5e-4", "1e-3", "5e-3", "5e-2"]
    metric_dict = get_ver_metrics(match_labels.tolist(), scores.tolist(), FPRs)
    print(metric_dict)

    return metric_dict["ACC"]


def test_epoch_ver(model, pair_data_loader, device, config):
    """Test for verification task (no labels)"""
    model.eval()
    scores = []
    batch_bar = tqdm(total=len(pair_data_loader), dynamic_ncols=True, position=0, leave=False, desc="Test Veri.")

    for i, (images1, images2) in enumerate(pair_data_loader):

        images = torch.cat([images1, images2], dim=0).to(device)
        # Get model outputs
        with torch.inference_mode():
            outputs = model(images)

        feats = F.normalize(outputs["feats"], dim=1)
        feats1, feats2 = feats.chunk(2)
        similarity = F.cosine_similarity(feats1, feats2)
        scores.extend(similarity.cpu().numpy().tolist())
        batch_bar.update()

    return scores
