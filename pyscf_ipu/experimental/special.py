# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.ops import segment_sum
from jax.scipy.special import betaln, gammainc, gammaln

from pyscf_ipu.experimental import binom_factor_table

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

# Various binom implementations


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

# Various gammanu implementations


def gammanu_gamma(nu: IntN, t: FloatN, epsilon: float = 1e-10) -> FloatN:
    """
    eq 2.11 from THO but simplified using SymPy and converted to jax

        t, u = symbols("t u", real=True, positive=True)
        nu = Symbol("nu", integer=True, nonnegative=True)

        expr = simplify(integrate(u ** (2 * nu) * exp(-t * u**2), (u, 0, 1)))
        f = lambdify((nu, t), expr, modules="scipy")
        ?f

    We evaulate this in log-space to avoid overflow/nan
    """
    t = jnp.maximum(t, epsilon)
    x = nu + 0.5
    gn = jnp.log(0.5) - x * jnp.log(t) + jnp.log(gammainc(x, t)) + gammaln(x)
    return jnp.exp(gn)


def gammanu_series(nu: IntN, t: FloatN, num_terms: int = 128) -> FloatN:
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


gammanu = gammanu_series


# Several binom_factor implementations


def binom_factor_direct(i: int, j: int, a: float, b: float, s: int):
    """
    Eq. 15 from Augspurger JD, Dykstra CE. General quantum mechanical operators. An
    open-ended approach for one-electron integrals with Gaussian bases. Journal of
    computational chemistry. 1990 Jan;11(1):105-11.
    <https://doi.org/10.1002/jcc.540110113>
    """
    return sum(
        binom_beta(i, s - t) * binom_beta(j, t) * a ** (i - (s - t)) * b ** (j - t)
        for t in range(max(s - i, 0), j + 1)
    )


def binom_factor_segment_sum(
    i: int, j: int, a: float, b: float, lmax: int = LMAX
) -> FloatN:
    # Vectorized version of above
    s, t = jnp.tril_indices(lmax + 1)
    out = binom(i, s - t) * binom(j, t) * a ** (i - (s - t)) * b ** (j - t)
    mask = ((s - i) <= t) & (t <= j)
    out = jnp.where(mask, out, 0.0)
    return segment_sum(out, s, num_segments=lmax + 1)


def binom_factor__via_segment_sum(
    i: int, j: int, a: float, b: float, s: int, lmax=LMAX
):
    return jnp.take(binom_factor_segment_sum(i, j, a, b, lmax), s)


binom_factor_table_W = jnp.array(binom_factor_table.build_binom_factor_table())


def binom_factor__via_lookup(
    i: int, j: int, a: float, b: float, s: int, lmax=None
) -> FloatN:
    # Lookup-table version of above -- see binom_factor_table.ipynb for the derivation
    # lmax is ignored, but used to allow easy swapping with above implementation
    monomials = jnp.array(binom_factor_table.get_monomials(a, b))
    coeffs = binom_factor_table_W[i, j, s]
    return coeffs @ monomials


binom_factor_default = binom_factor__via_segment_sum
