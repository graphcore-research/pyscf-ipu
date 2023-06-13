from dataclasses import dataclass
from functools import partial
from pprint import pprint
from typing import Optional, Tuple

import torch
from qm1b_dataset import create_qm1b_loader
from model import ModelConfig
from poptorch import Options
from poptorch_geometric.dataloader import CustomFixedSizeDataLoader
from torch_geometric.data import Data, Dataset
from torch_geometric.datasets import QM9


@dataclass
class DataConfig:
    dataset: str = "qm9"
    num_workers: int = 1
    num_train: int = 100000
    num_test: int = 20000
    max_num_examples: int = int(1e9)
    test_file: Optional[str] = None
    train_folder: Optional[str] = None
    shuffle: bool = True


def prep_qm9(data, target=4, use_half: bool = False):
    """
    Prepares QM9 molecules for training SchNet for HOMO-LUMO gap prediction
    task.  Outputs a data object with attributes:
        z: the atomic number as a vector of integers with length [num_atoms]
        pos: the atomic position as a [num_atoms, 3] tensor of float32 values.
        y: the training target value. By default this will be the HOMO-LUMO gap
        energy in electronvolts (eV).
    """
    dtype = torch.float16 if use_half else torch.float32
    return Data(
        z=data.z, pos=data.pos.to(dtype), y=data.y[0, target].view(-1).to(dtype)
    )


def split(dataset: Dataset, data_config: DataConfig):
    # Select test set first to ensure we are sampling the same distribution
    split_ids = [
        (0, data_config.num_test),
        (data_config.num_test, data_config.num_test + data_config.num_train),
    ]
    return [dataset[u:v] for u, v in split_ids]


def create_pyg_dataset(config: DataConfig, use_half: bool = False):
    transform = partial(prep_qm9, use_half=use_half)
    if config.dataset == "qm9":
        dataset = QM9(root="../datasets/qm9", transform=transform)
    else:
        raise ValueError(f"Invalid dataset: {config.dataset}")

    if config.shuffle:
        dataset = dataset.shuffle()
    return split(dataset, config)


def create_loader(
    data_config: DataConfig,
    model_config: ModelConfig,
    options: Tuple[Options],
):
    if data_config.dataset == "qm1b":
        return create_qm1b_loader(data_config, model_config, options)

    data_splits = create_pyg_dataset(data_config, model_config.use_half)

    loader_args = {
        "batch_size": model_config.batch_size,
        "num_workers": data_config.num_workers,
        "persistent_workers": data_config.num_workers > 0,
        "num_nodes": 32 * (model_config.batch_size - 1),
        "shuffle": data_config.shuffle,
        "collater_args": {"add_masks_to_batch": True},
    }

    return [
        CustomFixedSizeDataLoader(d, options=o, **loader_args)
        for d, o in zip(data_splits, options)
    ]


def loader_info(loader, prefix=None):
    info = {
        "len": len(loader),
        "len_dataset": len(loader.dataset),
        "num_workers": loader.num_workers,
        "drop_last": loader.drop_last,
        "batch_size": loader.batch_size,
        "sampler": type(loader.sampler).__name__,
    }

    if prefix is not None:
        info = {f"{prefix}{k}": v for k, v in info.items()}

    pprint(info)
    return info


def fake_batch(
    model_config: ModelConfig,
    options: Options,
    training: bool = True,
):
    num_graphs_per_batch = model_config.batch_size - 1
    combined_batch_size = (
        options.replication_factor
        * options.device_iterations
        * options.Training.gradient_accumulation
    )

    num_nodes = 32 * num_graphs_per_batch * combined_batch_size
    float_dtype = torch.float16 if model_config.use_half else torch.float32
    z = torch.zeros(num_nodes, dtype=torch.long)
    pos = torch.zeros(num_nodes, 3, dtype=float_dtype)
    batch = torch.zeros(num_nodes, dtype=torch.long)
    y = torch.zeros(model_config.batch_size * combined_batch_size, dtype=float_dtype)
    out = (z, pos, batch, y)
    return out if training else out[:-1]
