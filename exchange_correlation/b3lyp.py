import jax.numpy as jnp 
import jax 

import sys 
import pathlib 
sys.path.append( str(pathlib.Path().resolve()) + "/" )

try: 
    from exchange_correlation.lda import __lda 
    from exchange_correlation.lyp import __lyp
    from exchange_correlation.b88 import __b88
    from exchange_correlation.vwn import __vwn
except: 
    from lda import __lda 
    from lyp import __lyp
    from b88 import __b88
    from vwn import __vwn


import time 

# ideas
# 1. pyscf has a "transform b3lyp" function thay may give us gradients without having to backprop? 
#    this may be faster to compile and give different memory-layouts ?
# 2. get a big test case like 5 water molecules by default; this will make sure we optimize time for a case we actually care about! 


# compare memory consumption with across different grid sizes. 
# perhaps add gradient checkpointing to grad? 


#@jax.jit # does removing inner jax.jit change stuff? 

def b3lyp(rho, EPSILON_B3LYP=0): 
    print(rho.shape, rho.dtype)

    rho0  = rho.T[:, 0] 
    norms = jnp.linalg.norm(rho[1:], axis=0).T**2+EPSILON_B3LYP 

    # change to jax.checkpoint(__b88)
    def lda(rho0):        return jax.vmap(jax.value_and_grad(lambda x: __lda(x)*0.08)) (rho0)
    def vwn(rho0):        return jax.vmap(jax.value_and_grad(lambda x: __vwn(x)*0.19)) (rho0) 

    # TODO; check if this really does increase numerical error (Checkpoint increased numerical error from 32 to ~1 for -gdb 11 -float32 ! )
    #def b88(rho0, norms): return jax.vmap(jax.value_and_grad(lambda rho0, norm: jax.checkpoint(__b88)(rho0, norm)*0.72, (0, 1))) (rho0, norms) 
    #def lyp(rho0, norms): return jax.vmap(jax.value_and_grad(lambda rho0, norm: jax.checkpoint(__lyp)(rho0, norm)*0.810, (0, 1))) (rho0, norms)

    def b88(rho0, norms): return jax.vmap(jax.value_and_grad(lambda rho0, norm: __b88(rho0, norm)*0.72, (0,1))) (rho0, norms) 
    def lyp(rho0, norms): return jax.vmap(jax.value_and_grad(lambda rho0, norm: __lyp(rho0, norm)*0.810, (0,1)))  (rho0, norms)

    # adding the jax.jit so it 
    e_xc_lda, v_rho_lda               = jax.jit(lda)(rho0)
    e_xc_vwn, v_rho_vwn               = jax.jit(vwn)(rho0)

    #./profile.sh  -id 3  -backend ipu -float32   
    # [##################################################] 100% Compilation Finished [Elapsed: 00:01:12.6]

    # removing these two reduces from 2min to [1min]?
    e_xc_b88, (v_rho_b88, v_norm_b88) = jax.jit(b88)(rho0, norms)
    e_xc_lyp, (v_rho_lyp, v_norm_lyp) = jax.jit(lyp)(rho0, norms)

    #e_xc_b88, (v_rho_b88, v_norm_b88) = 0, (0, 0) 
    #e_xc_lyp, (v_rho_lyp, v_norm_lyp) = 0, (0, 0) 


    e_xc       = e_xc_lda + (e_xc_vwn + e_xc_b88 + e_xc_lyp) / rho[0]


    v_xc_rho   = v_rho_lda*4*rho[0] + v_rho_vwn + v_rho_b88 + v_rho_lyp 
    v_xc_norms = v_norm_b88 + v_norm_lyp 
    #v_xc_norms = jnp.zeros(rho[0].shape)
    #print(v_xc_norms.shape, rho.shape)

    print(e_xc.dtype, v_xc_rho.dtype, v_xc_norms.dtype)


    return e_xc, v_xc_rho, v_xc_norms



@jax.jit
def do_lda(rho, EPSILON_B3LYP=0): 
  rho0  = rho.T[:, 0] 
  norms = jnp.linalg.norm(rho[1:], axis=0).T**2+EPSILON_B3LYP 

  # simple wrapper to get names in popvision; lambda doesn't give different names.. 
  def lda(rho0):        return jax.vmap(jax.value_and_grad(lambda x: __lda(x)*0.08)) (rho0)

  e_xc_lda, v_rho_lda               = jax.jit(lda)(rho0)

  e_xc       = e_xc_lda 
  v_xc_rho   = v_rho_lda*4*rho[0] 
  v_xc_norms = jnp.zeros(rho[0].shape)# v_norm_b88 + v_norm_lyp 

  return e_xc, v_xc_rho, v_xc_norms

def plot(rho, b, a, g, grad, vnorm=None, name=""): # b is pyscf a is us 

        import matplotlib.pyplot as plt 
        import numpy as np 

        fig, ax = plt.subplots(1, 3, figsize=(14, 4))
        ax[0].plot(rho[0], -b, 'o', label="pyscf.eval_b3lyp",  ms=7)
        ax[0].plot(rho[0], -a, 'x', label="jax_b3lyp", ms=2)

        print(np.max(np.abs(b)-np.abs(a)))

        ax[1].plot(rho[0], np.abs(a-b), 'x', label="absolute error")
        ax[1].set_yscale("log") 
        ax[1].set_xscale("log") 
        ax[1].set_xlabel("input to b3lyp") 
        ax[1].set_ylabel("absolute error") 
        ax[1].legend()

        ax[2].plot(rho[0], np.abs(a-b)/np.abs(b), 'x', label="relative error")
        ax[2].set_yscale("log") 
        ax[2].set_xscale("log") 
        ax[2].set_xlabel("input to b3lyp") 
        ax[2].set_ylabel("relative absolute error")
        ax[2].legend()

        ax[0].set_yscale("log")
        ax[0].set_xscale("log")
        ax[0].set_ylabel("input to b3lyp")
        ax[0].set_xlabel("output of b3lyp")
        ax[0].legend()

        plt.tight_layout()
        ax[1].set_title("E_xc [%s]" % name)
        plt.savefig("%s_1.jpg"%name)

        fig, ax = plt.subplots(1,3, figsize=(14, 4))

        ax[0].plot(rho[0], -g[0], 'o',                label="pyscf grad", ms=7)

        if grad.ndim == 2: grad = grad[:, 0]

        ax[0].plot(rho[0], -grad, 'x', label="jax grad",   ms=2) 

        print(np.max(np.abs(g[0])-np.abs(grad)))

        ax[0].legend()

        ax[1].plot(rho[0], np.abs(g[0]-grad), 'x', label="absolute error")
        ax[1].legend()
        ax[1].set_xlabel("input")
        ax[1].set_ylabel("absolute gradient error")
        ax[1].set_title("d E_xc / d electron_density [%s]" % name)

        ax[2].plot(rho[0], np.abs(g[0]-grad)/np.abs(g[0]), 'x', label="relative error")
        ax[2].legend()
        ax[2].set_xlabel("input")
        ax[2].set_ylabel("relative gradient error")


        ax[0].set_yscale("log")
        ax[0].set_xscale("log")

        ax[1].set_yscale("log")
        ax[1].set_xscale("log")

        ax[2].set_yscale("log")
        ax[2].set_xscale("log")
        plt.tight_layout()
        plt.savefig("%s_2.jpg"%name)

        if vnorm is not None: 
            fig, ax = plt.subplots(1,3, figsize=(14, 4))

            ax[0].plot(rho[0], np.abs(g[1]), 'o',                label="pyscf grad", ms=7)
            ax[0].plot(rho[0], np.abs(vnorm), 'x', label="jax grad",   ms=2) 
            ax[0].legend()

            print(np.max(np.abs(g[1])-np.abs(vnorm)))

            ax[1].plot(rho[0], np.abs(g[1]-vnorm), 'x', label="absolute error")
            ax[1].legend()
            ax[1].set_xlabel("input")
            ax[1].set_ylabel("absolute gradient error")

            ax[1].set_title("d E_xc / d norms [%s]" % name)

            ax[2].plot(rho[0], np.abs(g[1]-vnorm)/np.abs(g[1]), 'x', label="relative error")
            ax[2].legend()
            ax[2].set_xlabel("input")
            ax[2].set_ylabel("relative gradient error")

            ax[0].set_yscale("log")
            ax[0].set_xscale("log")

            ax[1].set_yscale("log")
            ax[1].set_xscale("log")

            ax[2].set_yscale("log")
            ax[2].set_xscale("log")
            plt.tight_layout()
            plt.savefig("%s_3.jpg"%name)


if __name__ == "__main__": 

    # perhaps make 5 water molecule test case? 
    import argparse
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
    import os.path as osp
    import numpy as np

    if not osp.exists("rhos_He2.npz"):
        import gdown  # pip install gdown 
        print("downloading test cases") 
        gdown.download("https://drive.google.com/u/1/uc?id=1u5y5XfbPNSelzAo-DP4Vg_xF_jEJlxtU&export=download", "rhos_He2.npz")

    rhos = np.load("rhos_He2.npz")["rhos"]
    
    rho = rhos[1]  
    pwd = osp.dirname(osp.realpath(__file__))
    rho = np.load(osp.join(pwd, "rho_water_5.npz"))["rho"]
    print(rho)
    print(rho.shape)

    numpy_rho =  rho
    jax_rho   = jnp.array(numpy_rho)

    _hyb, fn_facs = pyscf.dft.libxc.parse_xc("b3lyp") # .2 * HF + .08 * LDA + .72 * B88, .810 * LYP + .19 * VW
    b, g          = pyscf.dft.libxc._eval_xc(_hyb, fn_facs, rho, spin=0, relativity=0, deriv=1, verbose=False)[:2]

    # for some reason compiling this doesn't make the compile bar 
    # it took 14s to compile. 

    t0 = time.time()
    b3lyp = jax.jit(b3lyp, backend=args.backend)
    e_xc, v_xc_rho, v_xc_norm = b3lyp(jax_rho)
    print(time.time()-t0)
    print("plotting.. ")

    t0 = time.time()

    e_xc, v_xc_rho, v_xc_norm = [np.array(a) for a in [e_xc, v_xc_rho, v_xc_norm]]
    print(time.time()-t0)

    plot(rho, b, e_xc, g, v_xc_rho , vnorm=v_xc_norm, name="B3LYP")
    print(time.time()-t0)



