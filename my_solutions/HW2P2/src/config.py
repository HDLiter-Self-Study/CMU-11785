"""
Configuration file for HW2P2: Image Recognition and Verification
"""

config = {
    # Training hyperparameters
    "batch_size": 256,  # Increase this if your GPU can handle it
    "lr": 0.005,  # Learning rate
    "epochs": 50,
    # Data paths
    "data_dir": "./data/cls_data",
    "data_ver_dir": "./data/ver_data",
    "checkpoint_dir": "./checkpoints",
    # Training options
    "eval_cls": True,  # Whether to evaluate classification task
    "eval_ver": True,  # Whether to evaluate verification task
    "resume_from": None,  # Path to checkpoint to resume from (e.g., "./checkpoints/last.pth")
    # WandB logging
    "use_wandb": False,  # Set to True to enable WandB logging
    "wandb_online": False,  # Set to True for online logging, False for offline
    "wandb_project": "hw2p2-ablations",
    "wandb_run_name": "hw2p2-training",
}
