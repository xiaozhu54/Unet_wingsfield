"""
PyTorch Dataset & DataLoader for airfoil .npz flow field data.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from .preprocess import parse_filename, Normalizer, split_input_target
from typing import Optional, Tuple
from config.train_config import TrainConfig


class AirfoilDataset(Dataset):
    """
    Loads .npz files from a given directory.

    Each .npz stores a (16, 128, 128) array under key "a":
        ch 0-2:   Ma, AoA, Re (scalar fields constant across grid)
        ch 3-11:  grid metric information
        ch 12-15: flow field (rho, p, u, v)

    The Dataset returns (input_tensor, target_tensor) where:
        input:  (12, H, W)  — 3 IC + 9 grid metrics
        target: (4, H, W)   — flow field
    """

    def __init__(self, data_dir: str, normalizer: Optional[Normalizer] = None,
                 fit_normalizer: bool = False, config: TrainConfig = None):
        self.config = config or TrainConfig()
        self.data_dir = data_dir
        self.files = sorted([
            f for f in os.listdir(data_dir)
            if f.endswith(".npz")
        ])
        if len(self.files) == 0:
            raise FileNotFoundError(f"No .npz files found in {data_dir}")

        self.metadata = []
        for f in self.files:
            try:
                self.metadata.append(parse_filename(f))
            except ValueError:
                self.metadata.append(("unknown", -1, -1, -1))

        self.normalizer = normalizer

        if fit_normalizer and normalizer is not None:
            all_data = self._load_all()
            self.normalizer.fit(all_data)

        self._cache: dict[int, tuple] = {}

    def _load_one(self, idx: int) -> np.ndarray:
        path = os.path.join(self.data_dir, self.files[idx])
        data = np.load(path)["a"]
        return data.astype(np.float32)

    def _load_all(self) -> np.ndarray:
        samples = [self._load_one(i) for i in range(len(self.files))]
        return np.stack(samples, axis=0)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        if idx in self._cache:
            return self._cache[idx]

        raw = self._load_one(idx)

        if self.normalizer is not None:
            raw_norm = self.normalizer.normalize(raw[np.newaxis])[0]
        else:
            raw_norm = raw

        inputs, targets = split_input_target(raw_norm[np.newaxis], self.config)
        x = torch.from_numpy(inputs[0])
        y = torch.from_numpy(targets[0])

        return x, y


def create_dataloaders(data_dir: str, config: TrainConfig = None,
                       val_ratio: float = None, test_ratio: float = None
                       ) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], Normalizer]:
    cfg = config or TrainConfig()
    vr = val_ratio if val_ratio is not None else cfg.val_ratio
    tr = test_ratio if test_ratio is not None else cfg.test_ratio

    # Fit normalizer
    temp_dataset = AirfoilDataset(data_dir, config=cfg)
    normalizer = Normalizer()
    all_data = temp_dataset._load_all()
    normalizer.fit(all_data)
    del temp_dataset

    full_dataset = AirfoilDataset(data_dir, normalizer=normalizer, config=cfg)

    n_total = len(full_dataset)
    n_test = int(n_total * tr)
    n_val = int(n_total * vr)
    n_train = n_total - n_val - n_test

    train_ds, val_ds, test_ds = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(cfg.random_seed),
    )

    nw = cfg.effective_num_workers()
    pm = cfg.pin_memory
    pf = cfg.prefetch_factor

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=nw, pin_memory=pm, prefetch_factor=pf if nw > 0 else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=nw, pin_memory=pm, prefetch_factor=pf if nw > 0 else None,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=nw, pin_memory=pm, prefetch_factor=pf if nw > 0 else None,
    ) if n_test > 0 else None

    return train_loader, val_loader, test_loader, normalizer
