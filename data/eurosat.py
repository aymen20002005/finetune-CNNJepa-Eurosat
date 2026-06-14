# Copyright (c) András Kalapos.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class EuroSATDataset(Dataset):
    """EuroSAT dataset loaded from CSV split files.

    The EuroSAT folder is expected to contain:
        - train.csv, validation.csv, test.csv
        - Class subdirectories (AnnualCrop/, Forest/, ...)

    Each CSV has columns: (index), Filename, Label, ClassName
    where Filename is a relative path like "AnnualCrop/AnnualCrop_142.jpg".

    Args:
        root (str): Path to the EuroSAT directory.
        split (str): Dataset split – one of ``'train'``, ``'validation'``, or ``'test'``.
        transform (callable, optional): Transform applied to PIL images.
        target_transform (callable, optional): Transform applied to integer labels.
    """

    SPLITS = ('train', 'validation', 'test')

    def __init__(self, root, split='train', transform=None, target_transform=None):
        if split not in self.SPLITS:
            raise ValueError(f"split must be one of {self.SPLITS}, got '{split}'")

        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        csv_path = os.path.join(root, f"{split}.csv")
        df = pd.read_csv(csv_path)

        self.filenames = df['Filename'].tolist()
        self.labels = df['Label'].tolist()

        # Build a stable class list sorted alphabetically
        self.classes = sorted(df['ClassName'].unique().tolist())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.filenames[idx])
        label = int(self.labels[idx])

        image = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label
