# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from typing import Tuple
from functools import partial

import chex
import jax.numpy as jnp
from jax import tree_map, vmap

from .primitive import Primitive, eval_primitive
from .types import FloatN, FloatNx3


@chex.dataclass
class Orbital:
    primitives: Tuple[Primitive]
    coefficients: FloatN

    @property
    def num_primitives(self) -> int:
        return len(self.primitives)

    def __call__(self, pos: FloatNx3) -> FloatN:
        assert pos.shape[-1] == 3, "pos must be have shape [N,3]"

        @partial(vmap, in_axes=(0, 0, None))
        def eval_orbital(p: Primitive, coef: float, pos: FloatNx3):
            return coef * eval_primitive(p, pos)

        batch = tree_map(lambda *xs: jnp.stack(xs), *self.primitives)
        out = jnp.sum(eval_orbital(batch, self.coefficients, pos), axis=0)
        return out

    @staticmethod
    def from_bse(center, alphas, lmn, coefficients):
        coefficients = coefficients.reshape(-1)
        assert len(coefficients) == len(alphas), "Expecting same size vectors!"
        p = [Primitive(center=center, alpha=a, lmn=lmn) for a in alphas]
        return Orbital(primitives=p, coefficients=coefficients)


def batch_orbitals(orbitals: Tuple[Orbital]):
    primitives = [p for o in orbitals for p in o.primitives]
    primitives = tree_map(lambda *xs: jnp.stack(xs), *primitives)
    coefficients = jnp.concatenate([o.coefficients for o in orbitals])
    orbital_index = jnp.concatenate(
        [
            i * jnp.ones(o.num_primitives, dtype=jnp.int32)
            for i, o in enumerate(orbitals)
        ]
    )
    return primitives, coefficients, orbital_index
