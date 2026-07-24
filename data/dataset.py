"""
PyTorch Dataset & DataLoader for airfoil .npz flow field data.
"""
import json
import os
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from config.train_config import TrainConfig
from .preprocess import Normalizer, parse_filename, split_input_target


class AirfoilDataset(Dataset):
    """
    Loads .npz files from a given directory.

    Each .npz stores a (16, 128, 128) array under key "a":
        ch 0-2:   Ma, AoA, Re (scalar fields constant across grid)
        ch 3-11:  grid metric information
        ch 12-15: flow field (rho, p, u, v)

    The Dataset returns (input_tensor, target_tensor) where:
        input:  (12, H, W)  = 3 IC + 9 grid metrics
        target: (4, H, W)   = flow field
    """

    def __init__(self, data_dir: str, normalizer: Optional[Normalizer] = None,
                 fit_normalizer: bool = False, config: TrainConfig = None,
                 cache_data: bool = False):
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

        self._cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        if cache_data:
            self._preload_cache()

    def _load_one(self, idx: int) -> np.ndarray:
        path = os.path.join(self.data_dir, self.files[idx])
        data = np.load(path)["a"]
        return data.astype(np.float32)

    def _load_all(self) -> np.ndarray:
        samples = [self._load_one(i) for i in range(len(self.files))]
        return np.stack(samples, axis=0)

    def _load_indices(self, indices: list[int]) -> np.ndarray:
        samples = [self._load_one(i) for i in indices]
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

    def _preload_cache(self) -> None:
        for idx in range(len(self.files)):
            raw = self._load_one(idx)
            if self.normalizer is not None:
                raw = self.normalizer.normalize(raw[np.newaxis])[0]
            inputs, targets = split_input_target(raw[np.newaxis], self.config)
            self._cache[idx] = (
                torch.from_numpy(inputs[0]),
                torch.from_numpy(targets[0]),
            )


def _loader_kwargs(cfg: TrainConfig) -> dict:
    nw = cfg.effective_num_workers()
    kwargs = {
        "num_workers": nw,
        "pin_memory": cfg.pin_memory,
    }
    if nw > 0:
        kwargs["prefetch_factor"] = cfg.prefetch_factor
        kwargs["persistent_workers"] = cfg.persistent_workers
    return kwargs


def build_split_manifest(data_dir: str, config: TrainConfig = None,
                         val_ratio: float = None, test_ratio: float = None) -> dict:
    cfg = config or TrainConfig()
    vr = val_ratio if val_ratio is not None else cfg.val_ratio
    tr = test_ratio if test_ratio is not None else cfg.test_ratio
    files = sorted([f for f in os.listdir(data_dir) if f.endswith(".npz")])
    if not files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")

    n_total = len(files)
    n_test = int(n_total * tr)
    n_val = int(n_total * vr)
    n_train = n_total - n_val - n_test
    permutation = torch.randperm(
        n_total, generator=torch.Generator().manual_seed(cfg.random_seed)
    ).tolist()

    split_indices_by_name = {
        "train": permutation[:n_train],
        "val": permutation[n_train:n_train + n_val],
        "test": permutation[n_train + n_val:],
    }

    return {
        "version": 1,
        "data_dir": os.path.abspath(data_dir),
        "random_seed": cfg.random_seed,
        "val_ratio": vr,
        "test_ratio": tr,
        "total_samples": n_total,
        "splits": {
            name: [
                {
                    "index": idx,
                    "filename": files[idx],
                }
                for idx in indices
            ]
            for name, indices in split_indices_by_name.items()
        },
    }


def save_split_manifest(manifest: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def load_split_manifest(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def split_indices(manifest: dict, split_name: str) -> list[int]:
    try:
        return [int(item["index"]) for item in manifest["splits"][split_name]]
    except KeyError as exc:
        raise ValueError(f"Split manifest has no '{split_name}' split") from exc


def validate_split_manifest(dataset: AirfoilDataset, manifest: dict) -> None:
    expected_total = int(manifest.get("total_samples", -1))
    if expected_total != len(dataset):
        raise ValueError(
            f"Split manifest sample count {expected_total} does not match "
            f"dataset sample count {len(dataset)}"
        )

    for split_name, items in manifest.get("splits", {}).items():
        for item in items:
            idx = int(item["index"])
            filename = item["filename"]
            if idx < 0 or idx >= len(dataset):
                raise ValueError(f"Split '{split_name}' contains invalid index {idx}")
            if dataset.files[idx] != filename:
                raise ValueError(
                    f"Split manifest mismatch at index {idx}: "
                    f"manifest has {filename}, dataset has {dataset.files[idx]}"
                )


def create_dataloaders(data_dir: str, config: TrainConfig = None,
                       val_ratio: float = None, test_ratio: float = None,
                       manifest: dict = None
                       ) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], Normalizer]:
    cfg = config or TrainConfig()
    split_manifest = manifest or build_split_manifest(data_dir, cfg, val_ratio, test_ratio)

    temp_dataset = AirfoilDataset(data_dir, config=cfg)
    normalizer = Normalizer()
    validate_split_manifest(temp_dataset, split_manifest)
    train_indices = split_indices(split_manifest, "train")
    normalizer.fit(temp_dataset._load_indices(train_indices))
    del temp_dataset

    full_dataset = AirfoilDataset(
        data_dir,
        normalizer=normalizer,
        config=cfg,
        cache_data=getattr(cfg, "cache_data", False),
    )
    validate_split_manifest(full_dataset, split_manifest)
    train_ds = Subset(full_dataset, train_indices)
    val_ds = Subset(full_dataset, split_indices(split_manifest, "val"))
    test_indices = split_indices(split_manifest, "test")
    test_ds = Subset(full_dataset, test_indices) if test_indices else None

    common_kwargs = _loader_kwargs(cfg)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, **common_kwargs
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, **common_kwargs
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False, **common_kwargs
    ) if test_ds is not None else None

    return train_loader, val_loader, test_loader, normalizer
