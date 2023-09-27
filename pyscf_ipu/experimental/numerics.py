# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from functools import wraps
from typing import Callable

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array


def apply_fpcast(v: Array, dtype: np.dtype):
    if isinstance(v, jnp.ndarray) and np.issubdtype(v, np.floating):
        return v.astype(dtype)

    return v


def fpcast(func: Callable, dtype=jnp.float32):
    @wraps(func)
    def wrapper(*args, **kwargs):
        inputs = [apply_fpcast(v, dtype) for v in args]
        outputs = func(*inputs, **kwargs)
        return outputs

    return wrapper


def compare_fp32_to_fp64(func: Callable):
    @wraps(func)
    def wrapper(*args, **kwargs):
        outputs_fp32 = fpcast(func, dtype=jnp.float32)(*args, **kwargs)
        outputs_fp64 = fpcast(func, dtype=jnp.float64)(*args, **kwargs)
        print_compare(func.__name__, outputs_fp32, outputs_fp64)
        return outputs_fp32

    return wrapper


def print_compare(name: str, fp32, fp64):
    fp32 = [fp32] if isinstance(fp32, jnp.ndarray) else fp32
    fp64 = [fp64] if isinstance(fp64, jnp.ndarray) else fp64

    for idx, (low, high) in enumerate(zip(fp32, fp64)):
        low = np.asarray(low).astype(np.float64)
        high = np.asarray(high)
        print(f"{name} output {idx} has max |fp64 - fp32| = {np.abs(high - low).max()}")
