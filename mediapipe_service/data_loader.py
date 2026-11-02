# data_loader.py
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import torch
try:
    from .sign_config import GESTURES, SEQUENCE_LENGTH
except ImportError:
    from sign_config import GESTURES, SEQUENCE_LENGTH

class LandmarkDataset(Dataset):
    def __init__(self, data_dir="data/live_landmarks", seq_len=SEQUENCE_LENGTH):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.files = sorted(self.data_dir.glob("*.npy"))

        if not self.files:
            raise ValueError(f"No .npy files found in {data_dir}")

        labels = sorted({file.stem.rsplit("_", 1)[0] for file in self.files})
        expected_labels = set(GESTURES)
        if set(labels) != expected_labels:
            raise ValueError(
                f"Expected webcam samples for {sorted(expected_labels)}, found {labels}. "
                "Collect samples for every sign before training."
            )

        self.label_to_idx = {label: index for index, label in enumerate(labels)}
        self.idx_to_label = {index: label for label, index in self.label_to_idx.items()}
        print(f"Labels found: {self.label_to_idx}")
        self.num_classes = len(self.label_to_idx)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        arr = np.load(file_path)  # shape (T, D) where T=40, D=258
        if arr.ndim != 2 or arr.shape[1] != 258 or arr.shape[0] == 0:
            raise ValueError(
                f"Invalid landmark sequence in {file_path}: expected (frames, 258), got {arr.shape}"
            )

        # Pad or truncate to seq_len
        if arr.shape[0] >= self.seq_len:
            seq = arr[:self.seq_len]
        else:
            pad = np.zeros((self.seq_len - arr.shape[0], arr.shape[1]), dtype=np.float32)
            seq = np.vstack([arr, pad])

        label_name = file_path.stem.rsplit('_', 1)[0]
        label_idx = self.label_to_idx[label_name]

        return torch.tensor(seq, dtype=torch.float32), torch.tensor(label_idx, dtype=torch.long)