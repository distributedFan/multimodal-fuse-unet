import argparse
import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torchvision import transforms

from data_prepare import RegDBDualModalDataset
from model import DualModalUNet

DEFAULT_CONFIG = {
    "data_root": "data/RegDB",
    "train_index_pattern": "data/RegDB/idx/train_thermal_{i}.txt",
    "test_index_pattern": "data/RegDB/idx/test_thermal_{i}.txt",
    "index_start": 1,
    "index_end": 10,
    "image_size": 128,
    "train_batch_size": 64,
    "test_batch_size": 256,
    "epochs": 100,
    "learning_rate": 0.001,
    "thermal_noise_mean": 0.05,
    "thermal_noise_std": 0.11,
    "visible_noise_mean": 0.0,
    "visible_noise_std": 1.0,
}


def load_config(path):
    config = DEFAULT_CONFIG.copy()
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            file_config = json.load(f)
        config.update(file_config)
    return config


def parse_args():
    parser = argparse.ArgumentParser(description="Dual-modal UNet training (RegDB).")
    parser.add_argument("--config", default="configs/unet.json", help="Path to JSON config.")
    parser.add_argument("--data-root", dest="data_root")
    parser.add_argument("--train-index-pattern", dest="train_index_pattern")
    parser.add_argument("--test-index-pattern", dest="test_index_pattern")
    parser.add_argument("--index-start", dest="index_start", type=int)
    parser.add_argument("--index-end", dest="index_end", type=int)
    parser.add_argument("--image-size", dest="image_size", type=int)
    parser.add_argument("--train-batch-size", dest="train_batch_size", type=int)
    parser.add_argument("--test-batch-size", dest="test_batch_size", type=int)
    parser.add_argument("--epochs", dest="epochs", type=int)
    parser.add_argument("--learning-rate", dest="learning_rate", type=float)
    parser.add_argument("--thermal-noise-mean", dest="thermal_noise_mean", type=float)
    parser.add_argument("--thermal-noise-std", dest="thermal_noise_std", type=float)
    parser.add_argument("--visible-noise-mean", dest="visible_noise_mean", type=float)
    parser.add_argument("--visible-noise-std", dest="visible_noise_std", type=float)

    args = parser.parse_args()
    config = load_config(args.config)

    for key, value in vars(args).items():
        if key == "config":
            continue
        if value is not None:
            config[key] = value

    return config


def add_gaussian_noise(tensor, mean=0.01, std=0.1):
    noise = torch.randn(tensor.size()) * std + mean
    return tensor + noise


def evaluate(model, test_loader, device):
    model.eval()
    model.to(device)

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for thermal_images, visible_images, labels in test_loader:
            thermal_images = thermal_images.to(device)
            visible_images = visible_images.to(device)
            labels = labels.to(device)

            outputs = model(thermal_images, visible_images)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"Accuracy: {accuracy:.4f}")


def train_model(
    model,
    train_loader,
    test_loader,
    optimizer,
    criterion,
    device,
    num_epochs,
    thermal_noise_mean,
    thermal_noise_std,
    visible_noise_mean,
    visible_noise_std,
):
    for epoch in range(num_epochs):
        model.train()
        for thermal_images, visible_images, labels in train_loader:
            thermal_images = add_gaussian_noise(
                thermal_images, mean=thermal_noise_mean, std=thermal_noise_std
            ).clamp(0, 1)
            thermal_images = thermal_images.to(device)
            visible_images = add_gaussian_noise(
                visible_images, mean=visible_noise_mean, std=visible_noise_std
            ).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(thermal_images, visible_images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")
        evaluate(model, test_loader, device)


def main():
    config = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [
            transforms.Resize((config["image_size"], config["image_size"])),
            transforms.ToTensor(),
        ]
    )

    train_index_files = [
        config["train_index_pattern"].format(i=i)
        for i in range(config["index_start"], config["index_end"] + 1)
    ]
    train_dataset = RegDBDualModalDataset(
        data_root=config["data_root"],
        index_files=train_index_files,
        transform=transform,
    )
    train_loader = DataLoader(train_dataset, batch_size=config["train_batch_size"], shuffle=True)

    test_index_files = [
        config["test_index_pattern"].format(i=i)
        for i in range(config["index_start"], config["index_end"] + 1)
    ]
    test_dataset = RegDBDualModalDataset(
        data_root=config["data_root"],
        index_files=test_index_files,
        transform=transform,
    )
    test_loader = DataLoader(test_dataset, batch_size=config["test_batch_size"], shuffle=False)

    model = DualModalUNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    train_model(
        model,
        train_loader,
        test_loader,
        optimizer,
        criterion,
        device,
        config["epochs"],
        config["thermal_noise_mean"],
        config["thermal_noise_std"],
        config["visible_noise_mean"],
        config["visible_noise_std"],
    )


if __name__ == "__main__":
    main()
