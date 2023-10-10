# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from dataclasses import asdict
from functools import partial
from itertools import product as cartesian_product
from typing import Callable

import jax.numpy as jnp
import numpy as np
from jax import jit, tree_map, vmap
from jax.ops import segment_sum

from .basis import Basis
from .orbital import batch_orbitals
from .primitive import Primitive, product
from .special import binom, binom_factor, factorial, factorial2, gammanu
from .types import Float3, FloatN, FloatNx3, FloatNxN
from .units import LMAX

"""
JAX implementation for integrals over Gaussian basis functions. Based upon the
closed-form expressions derived in

    Taketa, H., Huzinaga, S., & O-ohata, K. (1966). Gaussian-expansion methods for
    molecular integrals. Journal of the physical society of Japan, 21(11), 2313-2324.
    <https://doi.org/10.1143/JPSJ.21.2313>

Hereafter referred to as the "THO paper"

Related work:

[1] Augspurger JD, Dykstra CE. General quantum mechanical operators. An
    open-ended approach for one-electron integrals with Gaussian bases. Journal of
    computational chemistry. 1990 Jan;11(1):105-11.
    <https://doi.org/10.1002/jcc.540110113>

[2] PyQuante: <https://github.com/rpmuller/pyquante2/>
"""


@partial(vmap, in_axes=(0, 0, 0, 0, None))
def overlap_axis(i: int, j: int, a: float, b: float, alpha: float) -> float:
    idx = [(s, t) for s in range(LMAX + 1) for t in range(2 * s + 1)]
    s, t = jnp.array(idx, dtype=jnp.uint32).T
    out = binom(i, 2 * s - t) * binom(j, t)
    out *= a ** (i - (2 * s - t)) * b ** (j - t)
    out *= factorial2(2 * s - 1) / (2 * alpha) ** s

    mask = (2 * s - i <= t) & (t <= j)
    out = jnp.where(mask, out, 0)
    return jnp.sum(out)


def overlap_basis(b: Basis) -> FloatNxN:
    return integrate(b, vmap_overlap_primitives)


def integrate(b: Basis, primitive_op: Callable) -> FloatNxN:
    def take_primitives(indices):
        p = tree_map(lambda x: jnp.take(x, indices, axis=0), primitives)
        c = jnp.take(coefficients, indices)
        return p, c

    primitives, coefficients, orbital_index = batch_orbitals(b.orbitals)
    ii, jj = jnp.triu_indices(b.num_primitives)
    lhs, cl = take_primitives(ii.reshape(-1))
    rhs, cr = take_primitives(jj.reshape(-1))
    aij = cl * cr * primitive_op(lhs, rhs)
    A = jnp.zeros((b.num_primitives, b.num_primitives))
    A = A.at[ii, jj].set(aij)
    A = A + A.T - jnp.diag(jnp.diag(A))
    index = orbital_index.reshape(1, -1)
    return segment_sum(segment_sum(A, index).T, index)


def _overlap_primitives(a: Primitive, b: Primitive) -> float:
    p = product(a, b)
    pa = p.center - a.center
    pb = p.center - b.center
    out = jnp.power(jnp.pi / p.alpha, 1.5) * p.norm
    out *= jnp.prod(overlap_axis(a.lmn, b.lmn, pa, pb, p.alpha))
    return out


def _kinetic_primitives(a: Primitive, b: Primitive) -> float:
    t0 = b.alpha * (2 * jnp.sum(b.lmn) + 3) * _overlap_primitives(a, b)

    def offset_qn(ax: int, offset: int):
        lmn = b.lmn.at[ax].add(offset)
        return Primitive(**{**asdict(b), "lmn": lmn})

    axes = jnp.arange(3)
    b1 = vmap(offset_qn, (0, None))(axes, 2)
    t1 = jnp.sum(vmap(_overlap_primitives, (None, 0))(a, b1))

    b2 = vmap(offset_qn, (0, None))(axes, -2)
    t2 = jnp.sum(b.lmn * (b.lmn - 1) * vmap(_overlap_primitives, (None, 0))(a, b2))
    return t0 - 2.0 * b.alpha**2 * t1 - 0.5 * t2


def kinetic_basis(b: Basis) -> FloatNxN:
    return integrate(b, vmap_kinetic_primitives)


def build_gindex():
    vals = [
        (i, r, u)
        for i in range(LMAX + 1)
        for r in range(i // 2 + 1)
        for u in range((i - 2 * r) // 2 + 1)
    ]
    i, r, u = jnp.array(vals).T
    return i, r, u


def _nuclear_primitives(a: Primitive, b: Primitive, c: Float3):
    p = product(a, b)
    pa = p.center - a.center
    pb = p.center - b.center
    pc = p.center - c
    epsilon = 1.0 / (4.0 * p.alpha)

    @vmap
    def g_term(l1, l2, pa, pb, cp):
        i, r, u = build_gindex()
        index = i - 2 * r - u
        g = (
            jnp.power(-1, i + u)
            * binom_factor(l1, l2, pa, pb, i)
            * factorial(i)
            * jnp.power(cp, index - u)
            * jnp.power(epsilon, r + u)
        ) / (factorial(r) * factorial(u) * factorial(index - u))

        g = jnp.where(index <= l1 + l2, g, 0.0)
        return jnp.zeros(LMAX + 1).at[index].add(g)

    Gi, Gj, Gk = g_term(a.lmn, b.lmn, pa, pb, pc)

    ijk = jnp.arange(LMAX + 1)
    nu = (
        ijk[:, jnp.newaxis, jnp.newaxis]
        + ijk[jnp.newaxis, :, jnp.newaxis]
        + ijk[jnp.newaxis, jnp.newaxis, :]
    )

    W = (
        Gi[:, jnp.newaxis, jnp.newaxis]
        * Gj[jnp.newaxis, :, jnp.newaxis]
        * Gk[jnp.newaxis, jnp.newaxis, :]
        * gammanu(nu, p.alpha * jnp.inner(pc, pc))
    )

    return -2.0 * jnp.pi / p.alpha * p.norm * jnp.sum(W)


overlap_primitives = jit(_overlap_primitives)
kinetic_primitives = jit(_kinetic_primitives)
nuclear_primitives = jit(_nuclear_primitives)

vmap_overlap_primitives = jit(vmap(_overlap_primitives))
vmap_kinetic_primitives = jit(vmap(_kinetic_primitives))
vmap_nuclear_primitives = jit(vmap(_nuclear_primitives))


@partial(vmap, in_axes=(None, 0, 0))
def nuclear_basis(b: Basis, c: FloatNx3, z: FloatN) -> FloatNxN:
    op = partial(_nuclear_primitives, c=c)
    op = vmap(op)
    op = jit(op)
    return z * integrate(b, op)


def build_cindex():
    vals = [
        (i1, i2, r1, r2, u)
        for i1 in range(2 * LMAX + 1)
        for i2 in range(2 * LMAX + 1)
        for r1 in range(i1 // 2 + 1)
        for r2 in range(i2 // 2 + 1)
        for u in range((i1 + i2) // 2 - r1 - r2 + 1)
    ]
    i1, i2, r1, r2, u = jnp.array(vals).T
    return i1, i2, r1, r2, u


def _eri_primitives(a: Primitive, b: Primitive, c: Primitive, d: Primitive) -> float:
    p = product(a, b)
    q = product(c, d)
    pa = p.center - a.center
    pb = p.center - b.center
    qc = q.center - c.center
    qd = q.center - d.center
    qp = q.center - p.center
    delta = 1 / (4.0 * p.alpha) + 1 / (4.0 * q.alpha)

    def H(l1, l2, a, b, i, r, gamma):
        # Note this should match THO Eq 3.5 but that seems to incorrectly show a
        # 1/(4 gamma) ^(i- 2r) term which is inconsistent with Eq 2.22.
        # Using (4 gamma)^(r - i) matches the reported expressions for H_L
        u = factorial(i) * binom_factor(l1, l2, a, b, i)
        v = factorial(r) * factorial(i - 2 * r) * (4 * gamma) ** (i - r)
        return u / v

    def c_term(la, lb, lc, ld, pa, pb, qc, qd, qp):
        # THO Eq 2.22 and 3.4
        i1, i2, r1, r2, u = build_cindex()
        h = H(la, lb, pa, pb, i1, r1, p.alpha) * H(lc, ld, qc, qd, i2, r2, q.alpha)
        index = i1 + i2 - 2 * (r1 + r2) - u
        x = (-1) ** (i2 + u) * factorial(index + u) * qp ** (index - u)
        y = factorial(u) * factorial(index - u) * delta**index
        c = h * x / y

        mask = (i1 <= (la + lb)) & (i2 <= (lc + ld))
        c = jnp.where(mask, c, 0.0)
        return segment_sum(c, index, num_segments=4 * LMAX + 1)

    # Manual vmap over cartesian axes (x, y, z) as ran into possible bug.
    # See https://github.com/graphcore-research/pyscf-ipu/issues/105
    args = [a.lmn, b.lmn, c.lmn, d.lmn, pa, pb, qc, qd, qp]
    Ci, Cj, Ck = [c_term(*[v.at[i].get() for v in args]) for i in range(3)]

    ijk = jnp.arange(4 * LMAX + 1)
    nu = (
        ijk[:, jnp.newaxis, jnp.newaxis]
        + ijk[jnp.newaxis, :, jnp.newaxis]
        + ijk[jnp.newaxis, jnp.newaxis, :]
    )

    W = (
        Ci[:, jnp.newaxis, jnp.newaxis]
        * Cj[jnp.newaxis, :, jnp.newaxis]
        * Ck[jnp.newaxis, jnp.newaxis, :]
        * gammanu(nu, jnp.inner(qp, qp) / (4.0 * delta))
    )

    return (
        2.0
        * jnp.pi**2
        / (p.alpha * q.alpha)
        * jnp.sqrt(jnp.pi / (p.alpha + q.alpha))
        * p.norm
        * q.norm
        * jnp.sum(W)
    )


eri_primitives = jit(_eri_primitives)
vmap_eri_primitives = jit(vmap(_eri_primitives))


def gen_ijkl(n: int):
    """
    adapted from four-index transformations by S Wilson pg 257
    """
    for idx in range(n):
        for jdx in range(idx + 1):
            for kdx in range(idx + 1):
                lmax = jdx if idx == kdx else kdx
                for ldx in range(lmax + 1):
                    yield idx, jdx, kdx, ldx


def eri_basis_sparse(b: Basis):
    indices = []
    batch = []
    offset = np.cumsum([o.num_primitives for o in b.orbitals])
    offset = np.insert(offset, 0, 0)

    for count, idx in enumerate(gen_ijkl(b.num_orbitals)):
        mesh = [range(offset[i], offset[i + 1]) for i in idx]
        indices += list(cartesian_product(*mesh))
        batch += [count] * (len(indices) - len(batch))

    indices = jnp.array(indices, dtype=jnp.int32).T
    batch = jnp.array(batch, dtype=jnp.int32)
    primitives, coefficients, _ = batch_orbitals(b.orbitals)
    cijkl = jnp.stack([jnp.take(coefficients, idx) for idx in indices]).prod(axis=0)
    pijkl = [
        tree_map(lambda x: jnp.take(x, idx, axis=0), primitives) for idx in indices
    ]
    eris = cijkl * vmap_eri_primitives(*pijkl)
    return segment_sum(eris, batch, num_segments=count + 1)


def eri_basis(b: Basis):
    unique_eris = eri_basis_sparse(b)
    ii, jj, kk, ll = jnp.array(list(gen_ijkl(b.num_orbitals)), dtype=jnp.int32).T

    # Apply 8x permutation symmetry to build dense ERI from sparse ERI.
    eri_dense = jnp.empty((b.num_orbitals,) * 4, dtype=jnp.float32)
    eri_dense = eri_dense.at[ii, jj, kk, ll].set(unique_eris)
    eri_dense = eri_dense.at[ii, jj, ll, kk].set(unique_eris)
    eri_dense = eri_dense.at[jj, ii, kk, ll].set(unique_eris)
    eri_dense = eri_dense.at[jj, ii, ll, kk].set(unique_eris)
    eri_dense = eri_dense.at[kk, ll, ii, jj].set(unique_eris)
    eri_dense = eri_dense.at[kk, ll, jj, ii].set(unique_eris)
    eri_dense = eri_dense.at[ll, kk, ii, jj].set(unique_eris)
    eri_dense = eri_dense.at[ll, kk, jj, ii].set(unique_eris)
    return eri_dense
