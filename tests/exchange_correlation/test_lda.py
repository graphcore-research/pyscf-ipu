import pyscf.dft
import jax 
from exchange_correlation.lda import __lda 

import numpy as np
import pytest
from numpy.testing import assert_allclose


# Globally enable support for float64 used for CPU testing
from jax import config 
config.update('jax_enable_x64', True)

def lda_harness(rhos, rtol, backend="cpu", dtype=np.float32):
    for i, rho in enumerate(rhos):
        if i == 0:
          # TODO: investigate iteration zero failing for inputs small inputs
          # concretely: inputs <10^-8 -> outputs [10^-35, 10^-17] 
          continue

        # fn_facs[:1] selects the "LDA" component out of b3lyp.
        _hyb, fn_facs = pyscf.dft.libxc.parse_xc("b3lyp") # .2 * HF + .08 * LDA + .72 * B88, .810 * LYP + .19 * VW
        b, g          = pyscf.dft.libxc._eval_xc(_hyb, fn_facs[:1], rho, spin=0, relativity=0, deriv=1, verbose=False)[:2]

        _hyb, fn_facs = pyscf.dft.libxc.parse_xc("lda*0.08") # .2 * HF + .08 * LDA + .72 * B88, .810 * LYP + .19 * VW
        _b, _g          = pyscf.dft.libxc._eval_xc(_hyb, fn_facs[:1], rho, spin=0, relativity=0, deriv=1, verbose=False)[:2]
        assert_allclose(b, _b)
        assert_allclose(g[0], _g[0])


        f = jax.vmap(jax.value_and_grad(lambda x: __lda(x)*0.08))
        f = jax.jit(f, backend=backend)
        a, grad = f(rho.T[:, 0].astype(dtype))
        a, grad = np.array(a), np.array(grad) 
        grad = grad*4*rho[0]

        assert_allclose(a, b, rtol=rtol, err_msg=f"iteration{i}")
        assert_allclose(grad, g[0], rtol=rtol, err_msg=f"iteration{i}")

    # TODO: refactor plotting to experiments folder
    plot(rho, a, b, g, grad, name="LDA")



def test_lda_range_float32_cpu():
  with jax.default_device(jax.devices('cpu')[0]):
      N = 20
      rho = np.logspace(-10, 10, N, dtype=np.float32)
      grid = np.zeros((3, N), dtype=np.float32)
      grid[0, :] = np.linspace(-1, 1, N)
      rho = np.vstack([rho, grid])
      expected, expected_grad = pyscf.dft.libxc.eval_xc('lda', rho.astype(np.float64))[:2]
      f = jax.vmap(jax.value_and_grad(lambda x: __lda(x)))
      f = jax.jit(f)
      actual, actual_grad = f(rho.T[:, 0])
      actual_grad = actual_grad * 4 * rho[0]
      
      assert_allclose(actual, expected, rtol=1e-6)
      assert_allclose(actual_grad, expected_grad[0], rtol=1e-6)


@pytest.fixture
def rho_he2():
    return np.load("./data/rhos_He2.npz")["rhos"]


def test_lda_He2_float32_cpu(rho_he2):
    lda_harness(rho_he2, rtol=1e-6) 


def test_lda_He2_float64_cpu(rho_he2):
    lda_harness(rho_he2, dtype=np.float64, rtol=1e-14)


def test_lda_He2_float32_ipu(rho_he2):
    lda_harness(rho_he2, rtol=1e-5, backend='ipu')


# TODO: refactor plotting to experiments folder
def plot(rho, b, a, g, grad, vnorm=None, name=""): # b is pyscf a is us 
  import matplotlib.pyplot as plt

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
  plt.savefig("tests/exchange_correlation/%s_1.jpg"%name)

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
  plt.savefig("tests/exchange_correlation/%s_2.jpg"%name)

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
      plt.savefig("tests/exchange_correlation/%s_3.jpg"%name)
