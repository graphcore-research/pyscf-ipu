# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import jax.numpy as jnp
from numpy.testing import assert_allclose

from pyscf_ipu.experimental.integrals import factorial_fori, factorial_gamma
from pyscf_ipu.experimental.numerics import compare_fp32_to_fp64


def test_factorial():
    x = jnp.arange(8, dtype=jnp.float32)
    y_fori = compare_fp32_to_fp64(factorial_fori)(x, 8)
    y_gamma = compare_fp32_to_fp64(factorial_gamma)(x)
    assert_allclose(y_fori, y_gamma, atol=1e-2)
