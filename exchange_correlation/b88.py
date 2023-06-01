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

#def __b88_n_gnn(n, gnn):
#    return 2*_b88_a_gaa(n/2, gnn/4) 

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
        b, g          = pyscf.dft.libxc._eval_xc(_hyb, fn_facs[1:2], rho, spin=0, relativity=0, deriv=1, verbose=False)[:2]

        rho0  = rho.T[:, 0]
        norms = np.linalg.norm(rho[1:], axis=0).T**2
        #a, grad = jax.vmap(jax.value_and_grad(lambda x: b88(x)*0.72))(rho.T)

        f = jax.vmap(jax.value_and_grad(lambda rho0, norm: __b88_n_gnn(rho0, norm)*0.72, (0,1)))
        f = jax.jit(f, backend=args.backend)
        a, (vrho, vnorm)=  f(rho0, norms)
        a = a / rho[0]

        a, vrho, vnorm = [np.array(a) for a in [a, vrho, vnorm ]]

        plot(rho, b, a, g, vrho, vnorm=vnorm, name="B88")

