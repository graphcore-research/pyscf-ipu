import jax
import jax.numpy as jnp
import numpy

# we'll have to cut off some of the smaller ones
# previously, we got NaN from this guy in float32
# in dft.py we called jnp.nan_to_num( .. , nan=0 )
# this now actually produces numbers; we'll have to fix that.
def __lyp(n, gnn):

        # precompute
        A  = 0.04918
        B  = 0.132
        C  = 0.2533
        Dd = 0.349
        CF = 0.3 * (3.0 * numpy.pi * numpy.pi) ** (2.0 / 3.0)
        c0 = 2.0 ** (11.0 / 3.0) * (1/2)**(8/3)
        c1 = (1/3 + 1/8)*4

        # actual compute
        log_n = jnp.log(n)
        icbrtn = jnp.exp(log_n * (-1.0 / 3.0) )

        P       = 1.0 / (1.0 + Dd * icbrtn)
        omega   = jnp.exp(-C * icbrtn) * P
        delta   = icbrtn * (C + Dd * P)

        n_five_three  = jnp.exp(log_n*(-5/3))

        result = -A * (
        n *  P
        + B
        * omega
        * 1/ 4 *(
                 2 * CF * n * c0+
                   gnn * (60 - 14.0 * delta) /36  * n_five_three
                - gnn *c1  *      n_five_three
        )
        )

        return result
