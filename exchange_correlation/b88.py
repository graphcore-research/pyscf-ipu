import jax.numpy as jnp
import jax
import numpy as np

def __b88(a, gaa):
        # precompute
        c1 = (4.0 / 3.0)
        c2 = (-8.0 / 3.0)
        c3 = (-3.0 / 4.0) * (6.0 / np.pi) ** (1.0 / 3.0) * 2
        d  = 0.0042
        d2 = d * 2.
        d12 = d *12.

        # actual compute
        log_a     = jnp.log(a/2)
        na43      = jnp.exp(log_a * c1)
        chi2      = gaa / 4* jnp.exp(log_a * c2 )
        chi       = jnp.exp(jnp.log( chi2 ) / 2 )
        b88       = -(d * na43 * chi2) / (1.0 + 6*d * chi * jnp.arcsinh(chi)) *2
        slaterx_a = c3 * na43
        return slaterx_a + b88
