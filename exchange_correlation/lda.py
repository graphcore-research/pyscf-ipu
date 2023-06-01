import jax.numpy as jnp 
import jax 

# explicitly compute gradient aswell?
def __lda(rho): return -jnp.exp(1/3*jnp.log(rho) - 0.30305460484554375)

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
        print("downloading test cases")
        gdown.download("https://drive.google.com/u/1/uc?id=1u5y5XfbPNSelzAo-DP4Vg_xF_jEJlxtU&export=download", "rhos_He2.npz")

    rhos = np.load("rhos_He2.npz")["rhos"]
    rho = rhos[1]  

    numpy_rho =  rho
    jax_rho   = jnp.array(numpy_rho)

    _hyb, fn_facs = pyscf.dft.libxc.parse_xc("b3lyp") # .2 * HF + .08 * LDA + .72 * B88, .810 * LYP + .19 * VW
    b, g          = pyscf.dft.libxc._eval_xc(_hyb, fn_facs[:1], rho, spin=0, relativity=0, deriv=1, verbose=False)[:2]

    import time 
    t0 = time.time()
    f = jax.vmap(jax.value_and_grad(lambda x: __lda(x)*0.08))
    f = jax.jit(f, backend=args.backend)
    a, grad = f(rho.T[:, 0])
    print(time.time()-t0)
    t0 = time.time()
    a, grad = [np.array(a) for a in [a, grad]]
    print(grad.shape)
    grad = np.array(grad)
    grad[:] = grad[:]*4*rho[0]
    plot(rho, a, b, g, grad, name="LDA")
    print(time.time()-t0)