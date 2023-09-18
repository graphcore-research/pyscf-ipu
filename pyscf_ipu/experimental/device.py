# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from functools import partial, wraps
import numpy as np
from jax import devices, jit


def has_ipu() -> bool:
    try:
        return len(devices("ipu")) > 0
    except RuntimeError:
        pass

    return False


ipu_jit = partial(jit, backend="ipu")


def ipu_func(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        outputs = ipu_jit(func)(*args, **kwargs)

        if not isinstance(outputs, tuple):
            return np.asarray(outputs)

        return [np.asarray(o) for o in outputs]

    return wrapper
