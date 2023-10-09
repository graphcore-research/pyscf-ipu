# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.ops import segment_sum
from jax.scipy.special import betaln, gammaln

from .types import FloatN, IntN
from .units import LMAX


def factorial_fori(n: IntN, nmax: int = LMAX) -> IntN:
    def body_fun(i, val):
        return val * jnp.where(i <= n, i, 1)

    return lax.fori_loop(1, nmax + 1, body_fun, jnp.ones_like(n))


def factorial_gamma(n: IntN) -> IntN:
    """Appoximate factorial by evaluating the gamma function in log-space.

    This approximation is exact for small integers (n < 10).
    """
    approx = jnp.exp(gammaln(n + 1))
    return jnp.rint(approx)


def factorial_lookup(n: IntN, nmax: int = LMAX) -> IntN:
    N = np.cumprod(np.arange(1, nmax + 1))
    N = np.insert(N, 0, 1)
    N = jnp.array(N, dtype=jnp.uint32)
    return N.at[n.astype(jnp.uint32)].get()


factorial = factorial_gamma


def factorial2_fori(n: IntN, nmax: int = 2 * LMAX) -> IntN:
    def body_fun(i, val):
        return val * jnp.where((i <= n) & (n % 2 == i % 2), i, 1)

    return lax.fori_loop(1, nmax + 1, body_fun, jnp.ones_like(n))


def factorial2_lookup(n: IntN, nmax: int = 2 * LMAX) -> IntN:
    stop = nmax + 1 if nmax % 2 == 0 else nmax + 2
    N = np.arange(1, stop).reshape(-1, 2)
    N = np.cumprod(N, axis=0).reshape(-1)
    N = np.insert(N, 0, 1)
    N = jnp.array(N)
    n = jnp.maximum(n, 0)
    return N.at[n].get()


factorial2 = factorial2_lookup


def binom_beta(x: IntN, y: IntN) -> IntN:
    approx = 1.0 / ((x + 1) * jnp.exp(betaln(x - y + 1, y + 1)))
    return jnp.rint(approx)


def binom_fori(x: IntN, y: IntN, nmax: int = LMAX) -> IntN:
    bang = partial(factorial_fori, nmax=nmax)
    c = x * bang(x - 1) / (bang(y) * bang(x - y))
    return jnp.where(x == y, 1, c)


def binom_lookup(x: IntN, y: IntN, nmax: int = LMAX) -> IntN:
    bang = partial(factorial_lookup, nmax=nmax)
    c = x * bang(x - 1) / (bang(y) * bang(x - y))
    return jnp.where(x == y, 1, c)


binom = binom_lookup


def gammanu(nu: IntN, t: FloatN, num_terms: int = 128) -> FloatN:
    """
    eq 2.11 from THO but simplified as derived in equation 19 of gammanu.ipynb
    """
    an = nu + 0.5
    tn = 1 / an
    total = jnp.full_like(nu, tn, dtype=jnp.float32)

    for _ in range(num_terms):
        an = an + 1
        tn = tn * t / an
        total = total + tn

    return jnp.exp(-t) / 2 * total


def binom_factor(i: int, j: int, a: float, b: float, lmax: int = LMAX) -> FloatN:
    """
    Eq. 15 from Augspurger JD, Dykstra CE. General quantum mechanical operators. An
    open-ended approach for one-electron integrals with Gaussian bases. Journal of
    computational chemistry. 1990 Jan;11(1):105-11.
    <https://doi.org/10.1002/jcc.540110113>
    """
    s, t = jnp.tril_indices(lmax + 1)
    out = binom(i, s - t) * binom(j, t) * a ** (i - (s - t)) * b ** (j - t)
    mask = ((s - i) <= t) & (t <= j)
    out = jnp.where(mask, out, 0.0)
    return segment_sum(out, s, num_segments=lmax + 1)
