# TODO
# Inside 'def iter(..)' theres an eigendecomposition. 
# This causes a bit of numerical errors when we switch from float64 to flaot32. 
# Notably, we only use the eigenvectors (not the eigenvalues). 
# This leaves us free to scale the input matrix arbitrarily! 
# eigvects = jnp.linalg.eigh(A*c)[1] for all c
# If all the values in A are small scaling up can improve numerical stability. 
# However, the condition number cond(A*c)=eig_min / eig_max is independent of c!
# We can improve the condition number by translating cond(A+c*I) = (eig_min+c)/(eig_min+c)
# This notebook loads whatever DFT uses as input to the eigendecompostion and tries to find the parameters
# for both scaling and subtracting identity to get the best possible numerical precision. 

import numpy as np 
import scipy 
import jax.numpy as jnp 
import jax 
import jax.numpy as jnp 
from jax.config import config
config.FLAGS.jax_platform_name = 'cpu'
# optimize first for -skipdiis 
# add all the other things from xc to output! 
# E_xc, vhro, ... 
import matplotlib.pyplot as plt 
import matplotlib
import os 
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

import seaborn as sns
sns.set_theme()
sns.set_style("white")

float_name = {True: "float32", False: "float64"}

# add automatic plot after running! 
vals = []
xticks = []
xticklabels = []
iter_label = []



for outer_num, i in enumerate([20,  2,  4, 8, 16, 24]): # plot energy aswell 

  val = np.load("../tmp/ghamil.npz")["v"]
  print(val.shape)
  print(val.dtype)

  '''from jax_ipu_research.tile import ipu_eigh
  from functools import partial 

  @partial(jax.jit, backend="ipu")
  def test():
    ji_Q32, _ = ipu_eigh(val.astype(jnp.float32), sort_eigenvalues=True, num_iters=12)  
    return ji_Q32

  #ji_Q32 = np.asarray(test())'''
  ji_Q32 = np.zeros(val.shape) 

  config.update('jax_enable_x64', True)

  #initial_vals = np.linalg.eigh(val * np.linalg.eigh(val)[0].max() - np.eye(10))[0]
  initial_vals = np.linalg.eigh(val + np.linalg.eigh(val)[0].max() * np.eye(10))[0]

  val = val #* 1000 
  print(val)

  # GOTCHA! 
  # np.linalg.eigh was 10x better than jnp.linalg.eigh! 
  # how about scipy

  vals1, np_Q32 = np.linalg.eigh(val.astype(jnp.float32))
  vals2, np_Q64 = np.linalg.eigh(val.astype(jnp.float64))

  vals1, jnp_Q32 = jnp.linalg.eigh(val.astype(jnp.float32))
  vals2, jnp_Q64 = jnp.linalg.eigh(val.astype(jnp.float64))

  _, sp_Q32 = scipy.linalg.eigh(val.astype(jnp.float32))
  _, sp_Q64 = scipy.linalg.eigh(val.astype(jnp.float64))

  # our @Paul Balanca implementation 
  _, sp_Q32 = scipy.linalg.eigh(val.astype(jnp.float32))
  _, sp_Q64 = scipy.linalg.eigh(val.astype(jnp.float64))

  fig, ax= plt.subplots(3,4, figsize=(10, 10))
  #for a in ax.reshape(-1): 
  #  #a.set_xticks([], [])
  #  #a.set_yticks([], [])

  ax[2, 3].plot(np.linalg.eigh(val)[0].reshape(-1))

  ax[2, 0].imshow(val)
  ax[2, 1].plot(np.abs(val.reshape(-1)))
  ax[2,1].set_yscale("log")

  ax[2, 2].plot(np.abs(np_Q64.reshape(-1)), label="np_q64")
  ax[2, 2].plot(np.abs(np_Q32.reshape(-1)), label="np_q32")

  ax[2, 2].plot(np.abs(jnp_Q64.reshape(-1)), label="jnp_q64")
  ax[2, 2].plot(np.abs(jnp_Q32.reshape(-1)), label="jnp_q32")

  ax[2, 2].plot(np.abs(sp_Q64.reshape(-1)), label="sp_q64")
  ax[2, 2].plot(np.abs(sp_Q32.reshape(-1)), label="sp_q32")

  ax[2, 2].plot(np.abs(ji_Q32.reshape(-1)), label="ji_q32")
  ax[2,2].legend(fontsize="small")
  ax[2,2].set_yscale("log")



  ax[0,0].set_ylabel("float64")
  ax[1,0].set_ylabel("float32")

  ax[0,0].set_title("numpy")
  ax[0,1].set_title("jax.numpy")
  ax[0,2].set_title("scipy")
  ax[0,3].set_title("jacobi_ipu")

  ax[0, 0].imshow(np_Q64)
  ax[0, 1].imshow(jnp_Q64)
  ax[0, 2].imshow(sp_Q64)
  ax[1, 0].imshow(np_Q32)
  ax[1, 1].imshow(jnp_Q32)
  ax[1, 2].imshow(sp_Q32)
  ax[1, 3].imshow(ji_Q32)
  plt.tight_layout()
  plt.savefig("eigh.jpg")

  print(np.max(np.abs(np_Q64 - jnp_Q64)))
  print(np.max(np.abs(np_Q64 - sp_Q64)))
  print(np.max(np.abs(np_Q64 - jnp_Q32)))
  print(np.max(np.abs(np_Q64 - np_Q32)))
  print(np.max(np.abs(np_Q64 - sp_Q32)))
  print(np.max(np.abs(np_Q64 - ji_Q32)))
  exit()


