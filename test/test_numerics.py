# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from pyscf_ipu.experimental.special import (
    binom_beta,
    binom_fori,
    binom_lookup,
    factorial2_fori,
    factorial2_lookup,
    factorial_fori,
    factorial_gamma,
    factorial_lookup,
)


def test_factorial():
    x = jnp.array([1, 2, 3, 4, 5, 6, 7, 8])
    expect = jnp.array([1, 2, 6, 24, 120, 720, 5040, 40320])
    assert_allclose(factorial_fori(x, x[-1]), expect)
    assert_allclose(factorial_lookup(x, x[-1]), expect)
    assert_allclose(factorial_gamma(x), expect)


def test_factorial2():
    x = jnp.array([1, 2, 3, 4, 5, 6, 7, 8])
    expect = jnp.array([1, 2, 3, 8, 15, 48, 105, 384])
    assert_allclose(factorial2_fori(x), expect)
    assert_allclose(factorial2_fori(0), 1)

    assert_allclose(factorial2_lookup(x), expect)
    assert_allclose(factorial2_lookup(0), 1)


@pytest.mark.parametrize("binom_func", [binom_beta, binom_fori, binom_lookup])
def test_binom(binom_func):
    x = jnp.array([4, 4, 4, 4])
    y = jnp.array([1, 2, 3, 4])
    expect = jnp.array([4, 6, 4, 1])
    assert_allclose(binom_func(x, y), expect)

    zero = jnp.array([0])
    assert_allclose(binom_func(zero, y), jnp.zeros_like(x))
    assert_allclose(binom_func(x, zero), jnp.ones_like(y))
    assert_allclose(binom_func(y, y), jnp.ones_like(y))

    one = jnp.array([1])
    assert_allclose(binom_func(one, one), one)
    assert_allclose(binom_func(zero, -one), one)
    assert_allclose(binom_func(zero, zero), one)
