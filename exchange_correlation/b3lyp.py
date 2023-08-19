# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import jax.numpy as jnp
import jax

from exchange_correlation.lda import __lda
from exchange_correlation.lyp import __lyp
from exchange_correlation.b88 import __b88
from exchange_correlation.vwn import __vwn

CLIP_RHO_MIN  = 1e-9
CLIP_RHO_MAX  = 1e12

def b3lyp(rho, EPSILON_B3LYP=0):

    rho   = jnp.concatenate([jnp.clip(rho[:1], CLIP_RHO_MIN, CLIP_RHO_MAX), rho[1:4]*2])  

    rho0  = rho.T[:, 0]
    norms = jnp.linalg.norm(rho[1:], axis=0).T**2+EPSILON_B3LYP

    def lda(rho0):        return jax.vmap(jax.value_and_grad(lambda x: __lda(x)*0.08)) (rho0)
    def vwn(rho0):        return jax.vmap(jax.value_and_grad(lambda x: __vwn(x)*0.19)) (rho0)

    # disabled gradient checkpointing
    #def b88(rho0, norms): return jax.vmap(jax.value_and_grad(lambda rho0, norm: jax.checkpoint(__b88)(rho0, norm)*0.72, (0, 1))) (rho0, norms)
    #def lyp(rho0, norms): return jax.vmap(jax.value_and_grad(lambda rho0, norm: jax.checkpoint(__lyp)(rho0, norm)*0.810, (0, 1))) (rho0, norms)

    def b88(rho0, norms): return jax.vmap(jax.value_and_grad(lambda rho0, norm: __b88(rho0, norm)*0.72, (0,1)))(rho0, norms)
    def lyp(rho0, norms): return jax.vmap(jax.value_and_grad(lambda rho0, norm: __lyp(rho0, norm)*0.810, (0,1)))(rho0, norms)

    e_xc_lda, v_rho_lda               = jax.jit(lda)(rho0)
    e_xc_vwn, v_rho_vwn               = jax.jit(vwn)(rho0)
    e_xc_b88, (v_rho_b88, v_norm_b88) = jax.jit(b88)(rho0, norms)
    e_xc_lyp, (v_rho_lyp, v_norm_lyp) = jax.jit(lyp)(rho0, norms)

    e_xc       = e_xc_lda + (e_xc_vwn + e_xc_b88 + e_xc_lyp) / rho0 
    v_xc_rho   = v_rho_lda*4*rho0 + v_rho_vwn + v_rho_b88 + v_rho_lyp
    v_xc_norms = v_norm_b88 + v_norm_lyp

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
