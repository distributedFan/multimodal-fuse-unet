import os

import torch
from PIL import Image
from torch.utils.data import Dataset

class RegDBDualModalDataset(Dataset):
    def __init__(self, data_root, index_files, transform=None):
        self.data_root = data_root
        self.transform = transform
        self.indices = []
        for index_file in index_files:
            self.indices.extend(self.load_indices(index_file))
        self.thermal_images = []
        self.visible_images = []
        self.load_images()

    def load_indices(self, index_path):
        index_data = []
        with open(index_path, 'r') as file:
            for line in file:
                thermal_path = line.strip().split()[0]
                visible_path = thermal_path.replace("Thermal", "Visible").replace("_t_", "_v_")
                tmp = thermal_path.split('/')[2].split('_')
                label = self.get_label(tmp[0], tmp[1])
                index_data.append((thermal_path, visible_path, int(label)))
        return index_data

    def get_label(self, gender, position):
        if gender == 'female' and position == 'back':
            return 3
        if gender == 'female' and position == 'front':
            return 2
        if gender == 'male' and position == 'back':
            return 1
        return 0

    def load_images(self):
        for thermal_path, visible_path, label in self.indices:
            thermal_image_path = os.path.join(self.data_root, thermal_path)
            visible_image_path = os.path.join(self.data_root, visible_path)
            self.thermal_images.append((thermal_image_path, label))
            self.visible_images.append((visible_image_path, label))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        thermal_img_path, label = self.thermal_images[idx]
        visible_img_path, _ = self.visible_images[idx]

        thermal_image = Image.open(thermal_img_path)
        visible_image = Image.open(visible_img_path)

        if self.transform:
            thermal_image = self.transform(thermal_image)
            visible_image = self.transform(visible_image)

        return thermal_image, visible_image, label

class WeatherRegressionDataset(Dataset):
    def __init__(self, csv_file):
        import pandas as pd

        data = pd.read_csv(csv_file)

        # Define columns to drop by index (A,B,N,AA,AB,AE,AF,AG)
        columns_to_drop_indices = [0, 1, 13, 25, 26, 27, 30, 31, 32]
        target_column_index = 29  # AD

        # Identify target column name
        target_column = data.columns[target_column_index]

        # Drop unwanted columns and the first row
        data_processed = data.drop(data.columns[columns_to_drop_indices], axis=1).iloc[1:]

        # Process features and target
        X = data_processed.drop(columns=[target_column]).apply(pd.to_numeric, errors='coerce').dropna(axis=1)
        y = pd.to_numeric(data_processed[target_column], errors='coerce')

        # Remove rows containing NaN
        valid_indices = y.notna()
        self.X = torch.tensor(X[valid_indices].values, dtype=torch.float32)
        self.y = torch.tensor(y[valid_indices].values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class WeatherClassificationDataset(Dataset):
    def __init__(self, csv_file):
        import pandas as pd

        data = pd.read_csv(csv_file)

        # Define columns to drop by index (A,B,N,AA,AB,AE,AF,AG)
        columns_to_drop_indices = [0, 1, 13, 25, 26, 27, 30, 31, 32]
        target_column_index = 29  # AD

        target_column = data.columns[target_column_index]
        data_processed = data.drop(data.columns[columns_to_drop_indices], axis=1).iloc[1:]

        X = data_processed.drop(columns=[target_column]).apply(pd.to_numeric, errors="coerce").dropna(axis=1)
        y = data_processed[target_column]

        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y_encoded, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # Split features into two separate modalities (even split for simplicity).
        mid_point = self.X.shape[1] // 2
        x1 = self.X[idx, :mid_point]
        x2 = self.X[idx, mid_point:]
        return (x1, x2), self.y[idx]


# Backward-compatible alias for existing imports.
WeatherDataset = WeatherRegressionDataset
