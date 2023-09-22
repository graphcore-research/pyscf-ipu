# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from functools import partial
from typing import Callable

import jax.numpy as jnp
from jax import jit, tree_map, vmap
from jax.ops import segment_sum

from .basis import Basis
from .integrals import _kinetic_primitives, _overlap_primitives
from .orbital import batch_orbitals
from .primitive import Primitive
from .types import Float3, Float3xNxN


def grad_primitive_integral(
    primitive_op: Callable, atom_index: int, a: Primitive, b: Primitive
) -> Float3:
    """Generic gradient of a one-electron integral with respect the atom_index center"""
    axes = jnp.arange(3)
    lhs_p1 = vmap(a.offset_lmn, (0, None))(axes, 1)
    t1 = 2 * a.alpha * vmap(primitive_op, (0, None))(lhs_p1, b)

    lhs_m1 = vmap(a.offset_lmn, (0, None))(axes, -1)
    t2 = jnp.where(a.lmn > 0, a.lmn, jnp.zeros_like(a.lmn))
    t2 *= vmap(primitive_op, (0, None))(lhs_m1, b)
    grad_out = t1 - t2
    return jnp.where(a.atom_index == atom_index, grad_out, jnp.zeros_like(grad_out))


def grad_integrate(b: Basis, primitive_op: Callable) -> Float3xNxN:
    """Generic gradient of one-electron integrals over the basis set"""

    def take_primitives(indices):
        p = tree_map(lambda x: jnp.take(x, indices, axis=0), primitives)
        c = jnp.take(coefficients, indices)
        return p, c

    primitives, coefficients, orbital_index = batch_orbitals(b.orbitals)
    ii, jj = jnp.meshgrid(*[jnp.arange(b.num_primitives)] * 2, indexing="ij")
    lhs, cl = take_primitives(ii.reshape(-1))
    rhs, cr = take_primitives(jj.reshape(-1))

    op = vmap(primitive_op, (None, 0, 0))
    op = jit(vmap(op, (0, None, None)))
    atom_indices = jnp.arange(b.structure.num_atoms)
    out = op(atom_indices, lhs, rhs)
    out = jnp.sum(out, axis=0)

    out = cl * cr * out.T
    out = out.reshape(3, b.num_primitives, b.num_primitives)
    out = segment_sum(jnp.rollaxis(out, 1), orbital_index)
    out = segment_sum(jnp.rollaxis(out, -1), orbital_index)

    return jnp.rollaxis(out, -1)


grad_overlap_primitives = partial(grad_primitive_integral, _overlap_primitives)
grad_kinetic_primitives = partial(grad_primitive_integral, _kinetic_primitives)


grad_overlap_basis = partial(grad_integrate, primitive_op=grad_overlap_primitives)
grad_kinetic_basis = partial(grad_integrate, primitive_op=grad_kinetic_primitives)
