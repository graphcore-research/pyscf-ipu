# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import jax.numpy as jnp
from jax import jit, tree_map, vmap
from jax.ops import segment_sum

from .basis import Basis
from .integrals import _overlap_primitives
from .orbital import batch_orbitals
from .primitive import Primitive
from .types import Float3


def grad_overlap_primitives(i: int, a: Primitive, b: Primitive) -> Float3:
    """Analytic gradient of overlap integral with respect to atom i center"""
    axes = jnp.arange(3)
    lhs_p1 = vmap(a.offset_lmn, (0, None))(axes, 1)
    t1 = 2 * a.alpha * vmap(_overlap_primitives, (0, None))(lhs_p1, b)

    lhs_m1 = vmap(a.offset_lmn, (0, None))(axes, -1)
    t2 = jnp.where(a.lmn > 0, a.lmn, jnp.zeros_like(a.lmn))
    t2 *= vmap(_overlap_primitives, (0, None))(lhs_m1, b)
    grad_out = t1 - t2
    return jnp.where(a.atom_index == i, grad_out, jnp.zeros_like(grad_out))


# output is [3, N, N]
def grad_overlap_basis(b: Basis):
    def take_primitives(indices):
        p = tree_map(lambda x: jnp.take(x, indices, axis=0), primitives)
        c = jnp.take(coefficients, indices)
        return p, c

    primitives, coefficients, orbital_index = batch_orbitals(b.orbitals)
    ii, jj = jnp.meshgrid(*[jnp.arange(b.num_primitives)] * 2, indexing="ij")
    lhs, cl = take_primitives(ii.reshape(-1))
    rhs, cr = take_primitives(jj.reshape(-1))

    op = vmap(grad_overlap_primitives, (None, 0, 0))
    op = jit(vmap(op, (0, None, None)))
    atom_indices = jnp.arange(b.structure.num_atoms)
    out = op(atom_indices, lhs, rhs)
    out = jnp.sum(out, axis=0)

    out = cl * cr * out.T
    out = out.reshape(3, b.num_primitives, b.num_primitives)
    out = segment_sum(jnp.rollaxis(out, 1), orbital_index)
    out = segment_sum(jnp.rollaxis(out, -1), orbital_index)

    return jnp.rollaxis(out, -1)
