# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from typing import Optional

import chex
import jax.numpy as jnp
from jax.scipy.special import gammaln

from .types import Float3, FloatN, FloatNx3, Int3


@chex.dataclass
class Primitive:
    center: Float3 = jnp.zeros(3, dtype=jnp.float32)
    alpha: float = 1.0
    lmn: Int3 = jnp.zeros(3, dtype=jnp.int32)
    norm: Optional[float] = None

    def __post_init__(self):
        if self.norm is None:
            self.norm = normalize(self.lmn, self.alpha)

    @property
    def angular_momentum(self) -> int:
        return jnp.sum(self.lmn)

    def __call__(self, pos: FloatNx3) -> FloatN:
        return eval_primitive(self, pos)


def normalize(lmn: Int3, alpha: float) -> float:
    L = jnp.sum(lmn)
    N = ((1 / 2) / alpha) ** (L + 3 / 2)
    N *= jnp.exp(jnp.sum(gammaln(lmn + 1 / 2)))
    return N**-0.5


def product(a: Primitive, b: Primitive) -> Primitive:
    alpha = a.alpha + b.alpha
    center = (a.alpha * a.center + b.alpha * b.center) / alpha
    lmn = a.lmn + b.lmn
    c = a.norm * b.norm
    Rab = a.center - b.center
    c *= jnp.exp(-a.alpha * b.alpha / alpha * jnp.inner(Rab, Rab))
    return Primitive(center=center, alpha=alpha, lmn=lmn, norm=c)


def eval_primitive(p: Primitive, pos: FloatNx3) -> FloatN:
    assert pos.shape[-1] == 3, "pos must be have shape [N,3]"
    pos_translated = pos[:, jnp.newaxis] - p.center
    v = p.norm * jnp.exp(-p.alpha * jnp.sum(pos_translated**2, axis=-1))
    v *= jnp.prod(pos_translated**p.lmn, axis=-1)
    return v
