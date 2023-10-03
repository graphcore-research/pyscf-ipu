# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import math
from typing import List

import numpy as np
from numpy.typing import DTypeLike, NDArray


def make_num_elements_per_worker(N: int, num_workers: int) -> NDArray[np.int32]:
    """Build an array dividing (evenly) elements between workers.

    Args:
        N: Total number of elements.
        num_workers: Number of worker threads.
    Returns:
        (num_workers,) NumPy array with the division of work.
    """
    assert num_workers > 0
    num_elements = np.zeros((num_workers,), dtype=np.int32)
    # Fill with the base number of elements.
    num_elements.fill(N // num_workers)
    # Spread remainers.
    rem_num_elements = N - num_workers * num_elements[0]
    num_elements[:rem_num_elements] += 1
    return num_elements


def make_ipu_vector1d_worker_offsets(
    size: int, vector_size: int = 2, num_workers: int = 6, wdtype: DTypeLike = np.uint16
) -> NDArray[np.int_]:
    """Make the QR householder row update worker sizes, i.e. how many
    data vectors per worker thread?

    Args:
        size: Size of the vector to divide.
        vector_size: Vector size (2: float, 4: half).
        num_workers: Number of workers.
        wdtype: Worklists dtype.
    Returns:
        (6,) number of data vectors per thread.
    """

    def make_offsets_fn(sizes):
        sizes = [0] + sizes
        offsets = np.cumsum(np.array(sizes, wdtype), dtype=wdtype)
        return offsets

    assert size % vector_size == 0
    # Base worksize on the first few workers.
    base_worksize: int = math.ceil(size / (vector_size * num_workers))
    num_base_workers = size // (vector_size * base_worksize)
    worker_sizes: List[int] = [base_worksize] * num_base_workers
    if num_base_workers == num_workers:
        return make_offsets_fn(worker_sizes)

    # Remainer term, for the next thread.
    rem_worksize = size - base_worksize * vector_size * num_base_workers
    rem_worksize = rem_worksize // vector_size
    worker_sizes += [rem_worksize]
    # Fill the rest with zeros.
    unused_workers = num_workers - num_base_workers - 1
    worker_sizes += [0] * unused_workers
    return make_offsets_fn(worker_sizes)
