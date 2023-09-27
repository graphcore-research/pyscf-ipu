# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import jax.numpy as jnp
from jax.experimental import enable_x64
from numpy.testing import assert_allclose

from pyscf_ipu.experimental.integrals import factorial_fori, factorial_gamma
from pyscf_ipu.experimental.numerics import compare_fp32_to_fp64


def test_factorial():
    with enable_x64():
        n = 16
        x = jnp.arange(n, dtype=jnp.float32)
        y_fori = compare_fp32_to_fp64(factorial_fori)(x, n)
        y_gamma = compare_fp32_to_fp64(factorial_gamma)(x)
        assert_allclose(y_fori, y_gamma, 1e-2)
