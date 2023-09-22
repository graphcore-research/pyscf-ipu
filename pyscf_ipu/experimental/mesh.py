# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from typing import Optional, Tuple, Union

import jax.numpy as jnp

from .basis import Basis
from .types import FloatN, FloatNx3, FloatNxN


def uniform_mesh(
    n: Union[int, Tuple] = 50, b: Union[float, Tuple] = 10.0, ndim: int = 3
):
    if isinstance(n, int):
        n = (n,) * ndim

    if isinstance(b, float):
        b = (b,) * ndim

    if not isinstance(n, (tuple, list)):
        raise ValueError("Expected an integer ")

    if len(n) != ndim:
        raise ValueError("n must be a tuple with {ndim} elements")

    if len(b) != ndim:
        raise ValueError("b must be a tuple with {ndim} elements")

    axes = [jnp.linspace(-bi, bi, ni) for bi, ni in zip(b, n)]
    mesh = jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1)
    mesh = mesh.reshape(-1, ndim)
    return mesh


def electron_density(
    basis: Basis, mesh: FloatNx3, C: Optional[FloatNxN] = None
) -> FloatN:
    C = jnp.eye(basis.num_orbitals) if C is None else C
    orbitals = basis(mesh) @ C
    density = jnp.sum(basis.occupancy * orbitals * orbitals, axis=-1)
    return density
