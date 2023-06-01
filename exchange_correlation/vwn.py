import jax.numpy as jnp
import jax
import numpy as np

def __vwn(n):
        # All of this is okay, we compute it in numpy in float64.
        p     = np.array( [-0.10498, 0.0621813817393097900698817274255, 3.72744, 12.9352])
        f     = p[0] * p[2] / (p[0] * p[0] + p[0] * p[2] + p[3]) - 1.0
        f_inv_p1 = 1/f+1
        f_2  = f * 0.5
        sqrt =  np.sqrt(4.0 * p[3] - p[2] * p[2])
        precompute = p[2] * ( 1.0 / sqrt
                        - p[0]
                        / (
                                (p[0] * p[0] + p[0] * p[2] + p[3])
                                * sqrt
                                / (p[2] + 2.0 * p[0])
                        )
        )
        log_s_c =  np.log( 3.0 /(4*np.pi) ) / 6

        # Below e use in the same dtype as n
        dtype = n.dtype
        p = p.astype(dtype)
        f = f.astype(dtype)
        f_inv_p1 = (f_inv_p1).astype(dtype)
        f_2 = f_2.astype(dtype)
        sqrt = sqrt.astype(dtype)
        precompute = precompute.astype(dtype)
        log_s_c =log_s_c.astype(dtype)

        # compute stuff that depends on n
        log_s = - jnp.log(n) / 6 + log_s_c
        s_2   = jnp.exp( log_s *2)
        s     = jnp.exp( log_s )
        z     = sqrt / (2.0 * s + p[2])



        result = n * p[1] * (
                log_s
                #+ f *  jnp.log( jnp.sqrt( s_2 + p[2] * s + p[3] ) / (s-p[0])**(1/f+1) ) # problem with float, 1/f+1 was done in np which automatically sticks to float64
                + f *  jnp.log( jnp.sqrt( s_2 + p[2] * s + p[3] ) / (s-p[0])**(f_inv_p1) )
                + precompute * jnp.arctan(z)

        )


        return result
