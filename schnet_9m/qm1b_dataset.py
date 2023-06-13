import torch
import os.path as osp
from glob import glob
from functools import cached_property
from typing import Iterator, Optional

import pyarrow as pa
import pyarrow.dataset as pds
import pyarrow.parquet as pq
import numpy as np
from natsort import natsorted
from torch.utils.data import IterableDataset
from tqdm import tqdm

INT_MAX = np.iinfo(np.int32).max


class QM1B(IterableDataset):
    def __init__(
        self,
        files,
        num_subset: Optional[int] = None,
        shuffle: bool = True,
        max_open_files: int = 128,
    ) -> None:
        super().__init__()
        self.files = files
        self.shuffle = shuffle
        self.max_open_files = max_open_files

        num_samples = np.array([pq.read_metadata(f).num_rows for f in self.files])
        offsets = np.pad(num_samples.cumsum(), (1, 0))
        total_len = offsets[-1]

        if num_subset is None:
            self.eager = False
            self.sample_indices = None
            self.len = total_len
            return

        # Assuming roughly equal number of molecules / file:
        # calculate how many molecules we should sample from each file
        num_files = len(self.files)
        subsets, rem = divmod(num_subset, num_files)
        subsets = np.full(num_files, subsets)
        subsets[:rem] += 1
        self.sample_indices = [
            np.random.choice(n, s, replace=False) for n, s in zip(num_samples, subsets)
        ]

        sample_fraction = num_subset / total_len
        self.eager = sample_fraction < 0.5 or len(files) == 1
        self.len = num_subset

    def __len__(self) -> int:
        return self.len

    def __iter__(self) -> Iterator:
        if self.eager:
            yield from self.iter_examples(self.subset_table)
            return

        for shard in self.iter_shards():
            yield from self.iter_examples(shard)

    @cached_property
    def subset_table(self):
        return pa.concat_tables([s for s in self.iter_shards()])

    def iter_shards(self):
        splits = np.arange(self.max_open_files, len(self.files), self.max_open_files)

        if self.shuffle:
            order = np.random.permutation(len(self.files))
        else:
            order = range(len(self.files))

        for shard in np.array_split(order, splits):
            yield self.read_shard(shard)

    def read_shard(self, files_index):
        files = [self.files[i] for i in files_index]
        ds = pds.dataset(files)
        iter_batches = ds.to_batches(
            columns=["z", "pos", "y"],
            batch_size=INT_MAX,
            fragment_readahead=self.max_open_files,
        )
        iter_batches = zip(files_index, iter_batches)
        batches = []

        for idx, batch in tqdm(iter_batches, total=len(ds.files)):
            if self.sample_indices is not None:
                batch = batch.take(self.sample_indices[idx])

            batches.append(batch)

        return pa.Table.from_batches(batches)

    def iter_examples(self, table):
        if self.shuffle:
            table = safe_permute(table)

        for batch in table.to_batches():
            z = batch["z"].to_numpy(zero_copy_only=False)
            pos = batch["pos"].to_numpy(zero_copy_only=False)
            y = batch["y"].to_numpy(zero_copy_only=False)

            for idx in range(batch.num_rows):
                yield z[idx], pos[idx].reshape(-1, 3), y[idx], z[idx].shape[0]


class QM1BBatch:
    def __init__(self, num_fixed_nodes: int) -> None:
        self.num_fixed_nodes = num_fixed_nodes
        self.z = []
        self.pos = []
        self.y = []
        self.num_nodes = []
        self.graphs_mask = []
        self.batch = []

    def __call__(
        self,
        z,
        pos,
        y,
        num_nodes,
    ):
        # Add the real graphs to our running tally
        self.z += z
        self.pos += pos
        self.y += y
        self.num_nodes += num_nodes

        # Insert padding values
        num_padding_nodes = self.num_fixed_nodes - sum(num_nodes)
        zpad = np.zeros(num_padding_nodes, dtype=z[0].dtype)
        self.z.append(zpad)
        pospad = np.zeros((num_padding_nodes, 3), dtype=pos[0].dtype)
        self.pos.append(pospad)
        self.y.append(0.0)
        self.num_nodes.append(num_padding_nodes)

        # Append batch assignment vector for this mini-batch
        self.batch += [
            np.full(n, i) for i, n in enumerate([*num_nodes, num_padding_nodes])
        ]

        self.graphs_mask += [np.pad(np.ones(len(num_nodes), dtype=bool), (0, 1))]

    def cat(self):
        self.z = torch.from_numpy(np.concatenate(self.z, dtype=np.int64))
        self.pos = torch.from_numpy(np.concatenate(self.pos))
        self.y = torch.from_numpy(np.array(self.y))
        self.num_nodes = torch.from_numpy(np.array(self.num_nodes))
        self.graphs_mask = torch.from_numpy(np.concatenate(self.graphs_mask))
        self.batch = torch.from_numpy(np.concatenate(self.batch))
        return self


class QM1BCollator:
    def __init__(self, num_graphs_per_batch):
        self.num_graphs_per_batch = num_graphs_per_batch
        self.num_fixed_nodes = 32 * self.num_graphs_per_batch

    def __call__(self, data_list) -> QM1BBatch:
        batch = QM1BBatch(self.num_fixed_nodes)

        for idx in range(0, len(data_list), self.num_graphs_per_batch):
            mini_batch = data_list[idx : idx + self.num_graphs_per_batch]
            batch(*zip(*mini_batch))
        return batch.cat()


def safe_permute(T):
    # We could just do subset_table.take(perm) but that fails with arrow 11.0.0
    # ArrowInvalid: offset overflow while concatenating arrays
    perm = np.random.permutation(T.num_rows)
    offset = 0
    batches = []

    for batch in T.to_batches():
        mask = (perm >= offset) & (perm < offset + batch.num_rows)
        batches.append(batch.take(perm[mask] - offset))
        offset += batch.num_rows

    return pa.Table.from_batches(batches)


def combined_batch(options, num_graphs_per_batch):
    combined_batch_size = (
        num_graphs_per_batch
        * options.replication_factor
        * options.device_iterations
        * options.Training.gradient_accumulation
    )

    return combined_batch_size


def create_qm1b_iter_dataset(
    folder: str,
    shuffle: bool = False,
    split: bool = True,
    num_test: Optional[int] = None,
    num_train: Optional[int] = None,
):
    files = glob(osp.join(folder, "*.parquet"))
    files = natsorted(files)

    if shuffle:
        np.random.shuffle(files)

    if not split:
        return QM1B(files)

    # apply test-train split by file (assumes no smiles overlap multiple files)
    n = np.array([pq.read_metadata(f).num_rows for f in files])
    index = (np.cumsum(n) > 1e9).argmax() + 1
    test = QM1B(files[index:], num_subset=num_test, shuffle=shuffle)
    train = QM1B(files[:index], num_subset=num_train, shuffle=shuffle)
    return test, train


def create_qm1b_split_dataset(
    train_folder: str,
    val_file: str,
    shuffle: bool = False,
    num_test: Optional[int] = None,
    num_train: Optional[int] = None,
):
    files = glob(osp.join(train_folder, "*.parquet"))
    files = natsorted(files)

    if shuffle:
        np.random.shuffle(files)

    test = QM1B([val_file], num_subset=num_test, shuffle=shuffle)
    train = QM1B(files, num_subset=num_train, shuffle=shuffle)
    return test, train


def create_qm1b_loader(data_config, model_config, options):
    from torch.utils.data import DataLoader

    if model_config.use_half:
        raise NotImplementedError()

    num_graphs_per_batch = model_config.batch_size - 1
    val_batch_size, train_batch_size = [
        combined_batch(opt, num_graphs_per_batch) for opt in options
    ]

    loader_args = {
        "collate_fn": QM1BCollator(num_graphs_per_batch),
        "drop_last": True,
        "num_workers": data_config.num_workers,
        "shuffle": False,
    }

    test, train = create_qm1b_split_dataset(
        data_config.train_folder,
        data_config.test_file,
        shuffle=data_config.shuffle,
        num_train=data_config.num_train,
        num_test=data_config.num_test,
    )

    test_loader = DataLoader(test, batch_size=val_batch_size, **loader_args)
    train_loader = DataLoader(train, batch_size=train_batch_size, **loader_args)
    return test_loader, train_loader
