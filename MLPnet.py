import argparse
import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from data_prepare import WeatherClassificationDataset
from model import DualModalMLPNet

DEFAULT_CONFIG = {
    "data_path": "data/dc_weather.csv",
    "train_split": 0.8,
    "batch_size": 32,
    "input1_dim": 11,
    "input2_dim": 12,
    "hidden1_dim": 4,
    "hidden2_dim": 64,
    "decoder_hidden_dim": 32,
    "output_dim": 17,
    "epochs": 10,
    "learning_rate": 0.001,
}


def load_config(path):
    config = DEFAULT_CONFIG.copy()
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            file_config = json.load(f)
        config.update(file_config)
    return config


def parse_args():
    parser = argparse.ArgumentParser(description="Dual-modal MLP training (weather classification).")
    parser.add_argument("--config", default="configs/mlp.json", help="Path to JSON config.")
    parser.add_argument("--data-path", dest="data_path")
    parser.add_argument("--train-split", dest="train_split", type=float)
    parser.add_argument("--batch-size", dest="batch_size", type=int)
    parser.add_argument("--input1-dim", dest="input1_dim", type=int)
    parser.add_argument("--input2-dim", dest="input2_dim", type=int)
    parser.add_argument("--hidden1-dim", dest="hidden1_dim", type=int)
    parser.add_argument("--hidden2-dim", dest="hidden2_dim", type=int)
    parser.add_argument("--decoder-hidden-dim", dest="decoder_hidden_dim", type=int)
    parser.add_argument("--output-dim", dest="output_dim", type=int)
    parser.add_argument("--epochs", dest="epochs", type=int)
    parser.add_argument("--learning-rate", dest="learning_rate", type=float)

    args = parser.parse_args()
    config = load_config(args.config)

    for key, value in vars(args).items():
        if key == "config":
            continue
        if value is not None:
            config[key] = value

    return config


def main():
    config = parse_args()

    dataset = WeatherClassificationDataset(config["data_path"])

    train_size = int(config["train_split"] * len(dataset))
    train_dataset, test_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    model = DualModalMLPNet(
        config["input1_dim"],
        config["input2_dim"],
        hidden1_dim=config["hidden1_dim"],
        hidden2_dim=config["hidden2_dim"],
        decoder_hidden_dim=config["decoder_hidden_dim"],
        output_dim=config["output_dim"],
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)

    for epoch in range(config["epochs"]):
        model.train()
        for (x1, x2), labels in tqdm(train_loader):
            # x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(x1, x2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch + 1}/{config['epochs']}, Loss: {loss.item():.4f}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for (x1, x2), labels in test_loader:
            # x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)
            outputs = model(x1, x2)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")


if __name__ == "__main__":
    main()
