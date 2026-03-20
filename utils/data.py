"""Dataset I/O, split creation, and label preparation."""

import os
import torch
import pandas as pd
import requests
import zipfile
from io import BytesIO
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.data import Data


SUPPORTED_DATASETS = (
    "cora", "citeseer", "pubmed",
    "amazon-ratings", "tolokers", "roman-empire", "minesweeper", "questions"
)


def make_random_splits(num_nodes: int,
                       train_ratio: float = 0.8,
                       val_ratio: float = 0.1,
                       seed: int = 42,
                       device=None):
    """
    Create boolean train/val/test masks for node-level tasks.
    Deterministic w.r.t. seed.
    """
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")

    g = torch.Generator()
    g.manual_seed(int(seed))

    perm = torch.randperm(num_nodes, generator=g)
    n_train = int(num_nodes * train_ratio)
    n_val = int(num_nodes * val_ratio)

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    dev = device if device is not None else torch.device("cpu")

    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=dev)
    val_mask   = torch.zeros(num_nodes, dtype=torch.bool, device=dev)
    test_mask  = torch.zeros(num_nodes, dtype=torch.bool, device=dev)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    return train_mask, val_mask, test_mask


def ensure_splits(data,
                  seed: int,
                  train_ratio: float = 0.8,
                  val_ratio: float = 0.1):
    """
    Ensure data has 1D train/val/test masks.
    If they do not exist, create them.
    """
    num_nodes = data.num_nodes
    device = data.x.device if hasattr(data, "x") and data.x is not None else torch.device("cpu")

    train_mask, val_mask, test_mask = make_random_splits(
        num_nodes=num_nodes,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        device=device
    )

    return train_mask, val_mask, test_mask


def load_dataset(name, root="./data"):
    """Load a graph dataset by name.

    Supports Planetoid, HeterophilousGraph, CitationFull, Amazon,
    AttributedGraph, GraphLAND (Zenodo), GNNBenchmark, and LRGB families.
    """
    name_lower = name.lower()

    if name_lower in ["cora", "citeseer", "pubmed"]:
        from torch_geometric.datasets import Planetoid
        dataset = Planetoid(root=f"{root}/{name}", name=name.capitalize(), transform=NormalizeFeatures(), split='public')
        data = dataset[0]
        return data, dataset.num_classes

    elif name_lower in ["amazon-ratings", "tolokers", "roman-empire", "minesweeper", "questions"]:
        from torch_geometric.datasets import HeterophilousGraphDataset
        dataset = HeterophilousGraphDataset(root=f"{root}/{name}", name=name, transform=NormalizeFeatures())
        def _fix_split_masks(data):
            for key in ["train_mask", "val_mask", "test_mask"]:
                if hasattr(data, key):
                    m = getattr(data, key)
                    if m is None:
                        continue
                    if m.dim() == 2:
                        m = m.any(dim=1)
                    setattr(data, key, m.bool())
            return data

        data = dataset[0]
        data = _fix_split_masks(data)
        return data, dataset.num_classes

    else:
        raise ValueError(
            f"Dataset {name} not supported. "
            f"Supported datasets: {', '.join(SUPPORTED_DATASETS)}"
        )


def prepare_data_for_method(data, train_mask, val_mask, test_mask, noisy_train_labels, method_name):
    """Prepare a data clone where val/test keep original labels and train gets noisy labels."""
    data_for_method = data.clone()

    data_for_method.y = data.y_original.clone()
    data_for_method.y[train_mask] = noisy_train_labels

    return data_for_method


def verify_label_distribution(data, train_mask, val_mask, test_mask, run_id, method_name):

    print(f"[DEBUG Run {run_id}] {method_name} - Label distribution:")

    if hasattr(data, 'y_original'):

        train_corrupted = (data.y[train_mask] != data.y_original[train_mask]).sum()
        print(f"Training labels corrupted: {train_corrupted}/{train_mask.sum()} nodes")

        val_clean = (data.y[val_mask] == data.y_original[val_mask]).all()
        test_clean = (data.y[test_mask] == data.y_original[test_mask]).all()
        print(f"Val labels clean: {val_clean}")
        print(f"Test labels clean: {test_clean}")
