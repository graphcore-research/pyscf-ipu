# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from functools import partial
from typing import Callable

import jax.numpy as jnp
from jax import tree_map, vmap
from jax.ops import segment_sum

from .basis import Basis
from .integrals import _kinetic_primitives, _nuclear_primitives, _overlap_primitives
from .orbital import batch_orbitals
from .primitive import Primitive
from .types import Float3, Float3xNxN


def grad_primitive_integral(
    primitive_op: Callable, a: Primitive, b: Primitive
) -> Float3:
    """gradient of a one-electron integral over primitive functions defined as:

        < grad_a a | p | b >

    where the cartesian gradient is evaluated with respect to a.center.  For Gaussian
    primitives this gradient simplifies to:

        2 * alpha < a(l+1) | p | b > - l * < a(l-1) | p | b >

    where a(l+/-1) -> offset the lmn component for the corresponding gradient component.

    Args:
        primitive_op (Callable): integral operation over two primitives (a, b) -> float
        a (Primitive): left hand side of the integral
        b (Primitive): right hand side of the integral

    Returns:
        Float3: Gradient of the integral with respect to cartesian axes.
    """

    axes = jnp.arange(3)
    lhs_p1 = vmap(a.offset_lmn, (0, None))(axes, 1)
    t1 = 2 * a.alpha * vmap(primitive_op, (0, None))(lhs_p1, b)

    lhs_m1 = vmap(a.offset_lmn, (0, None))(axes, -1)
    t2 = a.lmn * vmap(primitive_op, (0, None))(lhs_m1, b)
    grad_out = t1 - t2
    return grad_out


def grad_overlap_primitives(a: Primitive, b: Primitive) -> Float3:
    """Evaluate the gradient of the overlap integral between primitives a and b. The
    gradient is with respect to a.center.

    Args:
        a (Primitive): left hand side of the overlap integral.
        b (Primitive): right hand side of the overlap integral.

    Returns:
        Float3: Gradient of the overlap integral with respect to cartesian axes.
    """
    return grad_primitive_integral(_overlap_primitives, a, b)


def grad_kinetic_primitives(a: Primitive, b: Primitive) -> Float3:
    """Evaluate the gradient of the kinetic energy integral between primitives a and b.
    The gradient is with respect to a.center.

    Args:
        a (Primitive): left hand side of the kinetic energy integral.
        b (Primitive): right hand side of the kinetic energy integral.

    Returns:
        Float3: Gradient of the kinetic energy integral with respect to cartesian axes.
    """
    return grad_primitive_integral(_kinetic_primitives, a, b)


def grad_nuclear_primitives(a: Primitive, b: Primitive, c: Float3) -> Float3:
    """Evaluate the gradient of the nuclear attraction integral between primitives a and
    b, and the nuclear potential centered on c. Gradient is with respect to a.center.

    Args:
        a (Primitive): left hand side of the nuclear attraction integral.
        b (Primitive): right hand side of the nuclear attraction integral.
        c (Float3): center for the nuclear attraction potential 1/(r - c)

    Returns:
        Float3: Gradient of the nuclear attraction integral with respect to cartesian
        axes
    """
    return grad_primitive_integral(partial(_nuclear_primitives, c=c), a, b)


def grad_integrate(basis: Basis, primitive_op: Callable) -> Float3xNxN:
    """gradient of a one-electron integral over the basis set of atomic orbitals.

    Args:
        basis (Basis): basis set of N atomic orbitals
        primitive_op (Callable): integral operation over two primitives (a, b) -> float

    Returns:
        Float3xNxN: Gradient of the integral with respect to cartesian axes evaluated
        over the NxN combinations of atomic orbitals.
    """

    def take_primitives(indices):
        p = tree_map(lambda x: jnp.take(x, indices, axis=0), primitives)
        c = jnp.take(coefficients, indices)
        return p, c

    primitives, coefficients, orbital_index = batch_orbitals(basis.orbitals)
    ii, jj = jnp.meshgrid(*[jnp.arange(basis.num_primitives)] * 2, indexing="ij")
    lhs, cl = take_primitives(ii.reshape(-1))
    rhs, cr = take_primitives(jj.reshape(-1))
    out = vmap(primitive_op)(lhs, rhs)

    out = cl * cr * out.T
    out = out.reshape(3, basis.num_primitives, basis.num_primitives)
    out = segment_sum(jnp.rollaxis(out, 1), orbital_index)
    out = segment_sum(jnp.rollaxis(out, -1), orbital_index)
    return jnp.rollaxis(out, -1)


def grad_overlap_basis(basis: Basis) -> Float3xNxN:
    """gradient of the overlap integral over the basis set of atomic orbitals

    Args:
        basis (Basis): basis set of N atomic orbitals

    Returns:
        Float3xNxN: Gradient of the overlap integral with respect to cartesian axes
        evaluated over the NxN combinations of atomic orbitals.
    """
    return grad_integrate(basis, grad_overlap_primitives)


def grad_kinetic_basis(basis: Basis) -> Float3xNxN:
    """gradient of the kinetic energy integral over the basis set of atomic orbitals

    Args:
        basis (Basis): basis set of N atomic orbitals

    Returns:
        Float3xNxN: Gradient of the kinetic energy integral with respect to cartesian
        axes evaluated over the NxN combinations of atomic orbitals.
    """
    return grad_integrate(basis, grad_kinetic_primitives)


def grad_nuclear_basis(basis: Basis) -> Float3xNxN:
    """gradient of the nuclear attraction integral over the basis set of atomic orbitals

    Args:
        basis (Basis): basis set of N atomic orbitals

    Returns:
        Float3xNxN: Gradient of the nuclear attraction integral with respect to
        cartesian axes evaluated over the NxN combinations of atomic orbitals.
    """

    def nuclear(c, z):
        op = partial(grad_nuclear_primitives, c=c)
        return z * grad_integrate(basis, op)

    out = vmap(nuclear)(basis.structure.position, basis.structure.atomic_number)
    return jnp.sum(out, axis=0)
