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


if __name__ == "__main__": 
        import argparse
        from b3lyp import plot 

        parser = argparse.ArgumentParser(description='Arguments for Density Functional Theory. ')
        parser.add_argument('-float32',   action="store_true", help='Whether to use float32 (default is float64). ')
        parser.add_argument('-backend',   default="cpu", help='Which backend to compile to {CPU, GPU, IPU}. ')
        args = parser.parse_args()

        if args.float32: 
                EPSILON_B3LYP  = 1e-30
        else:  # float64
                from jax import config 
                config.update('jax_enable_x64', True)
                EPSILON_B3LYP  = 0 #1e-20 

        import pyscf.dft
        import os 
        import numpy as np

        if not os.path.exists("rhos_He2.npz"):
                import gdown 
                print("donwloading test cases")
                gdown.download("https://drive.google.com/u/1/uc?id=1u5y5XfbPNSelzAo-DP4Vg_xF_jEJlxtU&export=download", "rhos_He2.npz")

        rhos = np.load("rhos_He2.npz")["rhos"]
        rho = rhos[1]  

        numpy_rho =  rho
        jax_rho   = jnp.array(numpy_rho)


        _hyb, fn_facs = pyscf.dft.libxc.parse_xc("b3lyp") # .2 * HF + .08 * LDA + .72 * B88, .810 * LYP + .19 * VW
        b, g          = pyscf.dft.libxc._eval_xc(_hyb, fn_facs[3:4], rho, spin=0, relativity=0, deriv=1, verbose=False)[:2]

        import time 

        t0 = time.time()
        f  = jax.vmap(jax.value_and_grad(lambda x: __vwn(x)*0.19))
        f = jax.jit(f, backend=args.backend)
        a, grad = f(rho.T[:, 0])
        print(time.time()-t0)
        t0 = time.time()
        a, grad = np.array(a), np.array(grad)
        print(a.shape,grad.shape)
        a = a / rho[0]
        plot(rho, a, b, g, grad, name="WVN")
        print(time.time()-t0)
        