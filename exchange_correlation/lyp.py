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



if __name__ == "__main__": 
        import argparse
        from b3lyp import plot 

        parser = argparse.ArgumentParser(description='Arguments for Density Functional Theory. ')
        parser.add_argument('-float32',   action="store_true", help='Whether to use float32 (default is float64). ')
        parser.add_argument('-backend',   default="cpu", help='Which backend to compile to {CPU, GPU, IPU}. ')
        parser.add_argument('-water',   action="store_true", help='load rho resulting from DFT on 2 water molecules.')
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

        rho = np.load("rho_water_5.npz")["rho"]
        print(rho.shape)

        numpy_rho =  rho
        jax_rho   = jnp.array(numpy_rho)

        _hyb, fn_facs = pyscf.dft.libxc.parse_xc("b3lyp") # .2 * HF + .08 * LDA + .72 * B88, .810 * LYP + .19 * VW
        b, g          = pyscf.dft.libxc._eval_xc(_hyb, fn_facs[2:3], rho, spin=0, relativity=0, deriv=1, verbose=False)[:2]

        rho0  = rho.T[:, 0]
        norms = np.linalg.norm(rho[1:], axis=0).T**2
        print(rho0.shape, norms.shape)

        import time 
        t0 = time.time()

        f = jax.vmap(jax.value_and_grad(lambda rho0, norm: __lyp(rho0, norm)*0.810, (0,1)))
        f = jax.jit(f, backend=args.backend)

        a, (vrho, vnorm)= f(rho0, norms)
        print(time.time()-t0)
        t0 = time.time()

        a, vrho, vnorm = np.array(a), np.array(vrho), np.array(vnorm)


        a = a / rho[0]
        print(a.shape, vrho.shape, vnorm.shape)

        plot(rho, b, a, g, vrho, vnorm=vnorm, name="LYP")
        print(time.time()-t0)
        