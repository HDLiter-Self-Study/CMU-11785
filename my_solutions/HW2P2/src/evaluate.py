"""
Evaluation script for HW2P2: Image Recognition and Verification
"""

import torch
import os
import argparse
from config import config
from models import ArchitectureFactory
from data import get_verification_dataloaders
from training_functions import test_epoch_ver
from utils import load_model


def evaluate_model(model_path, output_file="verification_submission.csv"):
    """Evaluate model on test dataset and generate submission file"""
    # Device setup
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: ", DEVICE)

    # Create verification dataloaders
    _, test_pair_dataloader = get_verification_dataloaders(config)
    print(f"Test batches: {len(test_pair_dataloader)}")

    # Initialize model using new architecture system
    model_config = {
        "architecture": "resnet",
        "depth": 18,
        "block_type": "basic",
        "width_multiplier": 1.0,
        "num_classes": 8631,  # Default number of classes
        "use_se": False,
    }

    factory = ArchitectureFactory()
    model = factory.create_model(model_config).to(DEVICE)

    # Load model checkpoint
    if os.path.exists(model_path):
        model, _, _, epoch, metrics = load_model(model, path=model_path)
        print(f"Loaded model from epoch {epoch}")
        print(f"Model metrics: {metrics}")
    else:
        print(f"Warning: Model checkpoint not found at {model_path}")
        print("Using randomly initialized model")

    # Generate test predictions
    print("\nGenerating test predictions...")
    scores = test_epoch_ver(model, test_pair_dataloader, DEVICE, config)

    # Save predictions to CSV
    with open(output_file, "w+") as f:
        f.write("ID,Label\n")
        for i in range(len(scores)):
            f.write("{},{}\n".format(i, scores[i]))

    print(f"Test predictions saved to {output_file}")
    return scores


def main():
    parser = argparse.ArgumentParser(description="Evaluate HW2P2 model on test dataset")
    parser.add_argument("--model_path", type=str, default="./checkpoints/best_ret.pth", help="Path to model checkpoint")
    parser.add_argument(
        "--output_file", type=str, default="verification_submission.csv", help="Output CSV file for predictions"
    )

    args = parser.parse_args()

    # Run evaluation
    scores = evaluate_model(args.model_path, args.output_file)

    print(f"\nEvaluation complete!")
    print(f"Generated {len(scores)} predictions")
    print(f"Score range: [{min(scores):.4f}, {max(scores):.4f}]")


if __name__ == "__main__":
    main()
