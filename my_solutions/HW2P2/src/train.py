"""
Main training script for HW2P2: Image Recognition and Verification
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import gc
import wandb
from config import config
from models import Network
from data import get_classification_dataloaders, get_verification_dataloaders
from training_functions import train_epoch, valid_epoch_cls, valid_epoch_ver
from utils import save_model


def main():
    # Device setup
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: ", DEVICE)

    # Create dataloaders
    train_loader, val_loader, train_dataset, val_dataset = get_classification_dataloaders(config)
    pair_dataloader, test_pair_dataloader = get_verification_dataloaders(config)

    # Print dataset info
    print("Number of classes    : ", len(train_dataset.classes))
    print("No. of train images  : ", train_dataset.__len__())
    print("Shape of image       : ", train_dataset[0][0].shape)
    print("Batch size           : ", config["batch_size"])
    print("Train batches        : ", train_loader.__len__())
    print("Val batches          : ", val_loader.__len__())

    # Initialize model
    model = Network().to(DEVICE)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")

    # Define optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    scaler = torch.amp.GradScaler()

    # Create checkpoint directory
    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    # Initialize wandb (optional)
    try:
        run = wandb.init(
            name="early-submission",
            project="hw2p2-ablations",
            config=config,
            mode="disabled",  # Change to "online" if you want to log to wandb
        )
    except:
        run = None
        print("WandB not available, continuing without logging")

    # Training loop
    e = 0
    best_valid_cls_acc = 0.0
    eval_cls = True
    best_valid_ret_acc = 0.0

    for epoch in range(e, config["epochs"]):
        # epoch
        print("\nEpoch {}/{}".format(epoch + 1, config["epochs"]))

        # train
        train_cls_acc, train_loss = train_epoch(model, train_loader, optimizer, scheduler, scaler, DEVICE, config)
        curr_lr = float(optimizer.param_groups[0]["lr"])
        print(
            "\nEpoch {}/{}: \nTrain Cls. Acc {:.04f}%\t Train Cls. Loss {:.04f}\t Learning Rate {:.04f}".format(
                epoch + 1, config["epochs"], train_cls_acc, train_loss, curr_lr
            )
        )

        metrics = {
            "train_cls_acc": train_cls_acc,
            "train_loss": train_loss,
        }

        # classification validation
        if eval_cls:
            valid_cls_acc, valid_loss = valid_epoch_cls(model, val_loader, DEVICE, config)
            print("Val Cls. Acc {:.04f}%\t Val Cls. Loss {:.04f}".format(valid_cls_acc, valid_loss))
            metrics.update(
                {
                    "valid_cls_acc": valid_cls_acc,
                    "valid_loss": valid_loss,
                }
            )

        # retrieval validation
        valid_ret_acc = valid_epoch_ver(model, pair_dataloader, DEVICE, config)
        print("Val Ret. Acc {:.04f}%".format(valid_ret_acc))
        metrics.update({"valid_ret_acc": valid_ret_acc})

        # save model
        save_model(model, optimizer, scheduler, metrics, epoch, os.path.join(config["checkpoint_dir"], "last.pth"))
        print("Saved epoch model")

        # save best model
        if eval_cls:
            if valid_cls_acc >= best_valid_cls_acc:
                best_valid_cls_acc = valid_cls_acc
                save_model(
                    model, optimizer, scheduler, metrics, epoch, os.path.join(config["checkpoint_dir"], "best_cls.pth")
                )
                if run is not None:
                    wandb.save(os.path.join(config["checkpoint_dir"], "best_cls.pth"))
                print("Saved best classification model")

        if valid_ret_acc >= best_valid_ret_acc:
            best_valid_ret_acc = valid_ret_acc
            save_model(
                model, optimizer, scheduler, metrics, epoch, os.path.join(config["checkpoint_dir"], "best_ret.pth")
            )
            if run is not None:
                wandb.save(os.path.join(config["checkpoint_dir"], "best_ret.pth"))
            print("Saved best retrieval model")

        # log to tracker
        if run is not None:
            run.log(metrics)

        # Clean up GPU memory
        gc.collect()
        torch.cuda.empty_cache()

    print("\nTraining completed!")
    print(f"Best classification accuracy: {best_valid_cls_acc:.4f}%")
    print(f"Best retrieval accuracy: {best_valid_ret_acc:.4f}%")
    print("Use evaluate.py to generate test predictions.")

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
