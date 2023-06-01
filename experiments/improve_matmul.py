
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


def kahan_dot(xy): 
    sum   = jnp.array(0.0 ,dtype=jnp.float32)
    error = jnp.array(0.0, dtype=jnp.float32)

    for i in range(len(xy)):
        prod = xy[i]- error
        temp = sum + prod
        error = (temp - sum) - prod
        sum = temp
    return sum


  
def kahan_sum(xy): 
  n  = xy.shape[0]
  sum   = jnp.zeros((n, n), dtype=jnp.float32)
  error = jnp.zeros((n, n), dtype=jnp.float32)
  print(xy.shape, sum.shape, error.shape)

  for i in range(n):
    print(xy[i].shape, error.shape)
    prod = xy[i] - error
    temp = sum + prod
    error = (temp - sum) - prod
    sum = temp
  return sum


def stable_mult(A, B): 
  n = A.shape[0]
  A   = A.T

  all = (A.reshape(n, n, 1) * B.reshape(n, 1, n))

  abs_sort_indices = np.argsort(np.abs(all), axis=0)

  all = all[abs_sort_indices, np.arange(all.shape[1])[:, None], np.arange(all.shape[2])]
  print(all[:, 3,3])

  #return np.sum(all, axis=0)
  #return kahan_sum(all)

  sum = np.zeros((n, n))
  for i in range(n):
    sum += all[i]

  return sum 

for outer_num, i in enumerate([20,  2,  4, 8, 16, 24]): # plot energy aswell 

  np.random.seed(42)
  H = np.load("numerror/%i_hamiltonian_False.npz"%i)["v"]
  L_inv = np.load("numerror/%i_L_inv_False.npz"%i)["v"]


  #H.mean()
  scale_H = (1/H.mean()) # normalize to mean = 0 
  scale_L = (1/ L_inv.mean())
  H = H * scale_H
  L_inv = L_inv * scale_L
  #H = H * 100 
  #L_inv = L_inv * 100 


  # plot the numerical range of the matrices we are to multiply. 
  if True: 
    fig, ax = plt.subplots()

    xs = -np.ones(H.size)*outer_num
    ax.plot(H.reshape(-1).astype(np.float64), xs, 'C%io'%(i%10), ms=8, alpha=0.4)
    ax.plot(H.reshape(-1).astype(np.float32), xs, 'kx', ms=2)

    xs = -np.ones(L_inv.size)*outer_num+1
    ax.plot(L_inv.reshape(-1).astype(np.float64), xs, 'C%io'%(i%10), ms=8, alpha=0.4)
    ax.plot(L_inv.reshape(-1).astype(np.float32), xs, 'kx', ms=2)

    plt.plot(
    [10**(-10), 10**(-10)], 
    [0, 1], 
    'C7--', alpha=0.6)
    plt.plot(
      [10**(10), 10**10], 
      [0, 1], 
      'C7--', alpha=0.6)
    plt.plot(
      [10**(0), 10**0], 
      [0, 1], 
      'C7-', alpha=1)


    plt.xscale("log")
    plt.xlim([10**(-15), 10**18])
    plt.tight_layout()
    plt.savefig("mult_.jpg")

    plt.text(1e10, 0-0.25, "Hamiltonian", horizontalalignment='left', size='small', color='black', weight='normal')
    plt.text(1e10, 1-0.25, "L_inv", horizontalalignment='left', size='small', color='black', weight='normal')

  A = (L_inv @ H) / scale_H / scale_L #@ L_inv.T 

  C = stable_mult(L_inv.astype(np.float32).astype(np.float64), H.astype(np.float32).astype(np.float64)) / scale_H / scale_L

  L_inv = L_inv.astype(np.float32)
  H = H.astype(np.float32)

  B = (L_inv @ H) / scale_H / scale_L #@ L_inv.T 


  print(np.max(np.abs(A - A)))
  print(np.max(np.abs(A - B)))
  print(np.max(np.abs(A - C)))

  print(A.shape, B.shape, C.shape)

  fig, ax = plt.subplots(1,3)

  ax[0].imshow(A)
  ax[1].imshow(B)
  ax[2].imshow(C)
  plt.savefig("mult.jpg")



  exit()

