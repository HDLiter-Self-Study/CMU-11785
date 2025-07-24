import sys
import torch
import csv
import os
import json

from models.sweep_model import SweepModel
from data.datasets import AudioTestDataset
from torch.utils.data import DataLoader


def main():
    """
    Main function to evaluate a model and save predictions to a CSV file.
    Usage: python eval.py <model_path>
    The model should be a PyTorch model saved with its state_dict.
    model_data = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch,
                    "model_config": {...}  # Include any additional model configuration if needed
                }
    """
    if len(sys.argv) != 2:
        print("Usage: python eval.py <model_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the model
    model_data = torch.load(model_path)
    model_config = model_data["model_config"]
    # Initialize the model with the saved configuration
    model = SweepModel(**model_config).to(device)
    # Load the model state_dict
    model.load_state_dict(model_data["model"])
    model.eval()
    # Load phonemes from the config file
    config_dir = os.path.join(os.path.dirname(__file__), "configs")
    with open(os.path.join(config_dir, "phonemes.json"), "r") as f:
        phonemes_config = json.load(f)
    phonemes = phonemes_config["PHONEMES"]

    # Load the test dataset
    test_dataset = AudioTestDataset(
        root="data",
        phonemes=phonemes,
        context=model_data["context"],
        partition="test-clean",
        cepstral_normalization=model_data["cepstral_normalization"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=model_data["batch_size"],
        shuffle=False,
    )
    # Evaluate the model
    outputs = None
    with torch.inference_mode():
        for batch in test_loader:
            frames = batch
            frames = frames.to(device)
            batch_out = model(frames)
            if outputs is None:
                outputs = batch_out.cpu()
            else:
                outputs = torch.cat((outputs, batch_out.cpu()), dim=0)
    outputs = outputs.argmax(dim=1)  # Get the predicted labels

    # Load phonemes from the config file
    config_dir = os.path.join(os.path.dirname(__file__), "configs")
    with open(os.path.join(config_dir, "phonemes.json"), "r") as f:
        phonemes_config = json.load(f)
    phonemes = phonemes_config["PHONEMES"]

    # Save predictions to CSV
    predictions = outputs.numpy() if isinstance(outputs, torch.Tensor) else outputs
    output_path = "./output/submission.csv"
    if not os.path.exists("./output"):
        os.makedirs("./output")
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Id", "Label"])
        for idx, label in enumerate(predictions):
            writer.writerow([idx, phonemes[label]])


if __name__ == "__main__":
    main()
