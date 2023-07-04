# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import jax.numpy as jnp
import jax

def __lda(rho): return -jnp.exp(1/3*jnp.log(rho) - 0.30305460484554375)
