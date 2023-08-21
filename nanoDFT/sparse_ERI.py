import numpy as np 
import jax.numpy as jnp
import os 
import pyscf 
import jax
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_platform_name', "cpu")
os.environ['OMP_NUM_THREADS'] = "16"

# Construct molecule we can use for test case. 
mol = pyscf.gto.Mole(atom=[["C", (0, 0, i)] for i in range(16)], basis="sto3g")
mol.build()
N              = mol.nao_nr()                          # N: number of atomic orbitals (AO) 
density_matrix = pyscf.scf.hf.init_guess_by_minao(mol) # (N, N)
ERI            = mol.intor("int2e_sph")                # (N, N, N, N)
print(ERI.shape, density_matrix.shape)

# nanoDFT uses ERI in an "einsum" which is equal to matrix vector multiplication. 
truth          = jnp.einsum('ijkl,ji->kl', ERI, density_matrix) # <--- einsum 
ERI            = ERI.reshape(N**2, N**2)
density_matrix = density_matrix.reshape(N**2)                 
print(ERI.shape, density_matrix.shape)
alternative    = ERI @ density_matrix                           # <--- matrix vector mult 
alternative    = alternative.reshape(N, N)

assert np.allclose(truth, alternative)                          # they're equal! 

# First trick. The matrix is sparse! 
print(f"The matrix is {np.around(np.sum(ERI == 0)/ERI.size*100, 2)}% zeros!")

def sparse_representation(ERI):
  rows, cols = np.nonzero(ERI)
  values     = ERI[rows, cols]
  return rows, cols, values 

def sparse_mult(sparse, vector):
  rows, cols, values = sparse
  in_         = vector.take(cols, axis=0)
  prod        = in_*values
  segment_sum = jax.ops.segment_sum(prod, rows, N**2)
  return segment_sum 

sparse_ERI = sparse_representation(ERI) 
res = jax.jit(sparse_mult, backend="cpu")(sparse_ERI, density_matrix).reshape(N, N)

assert np.allclose(truth, res)

# Problem:
# If I increase molecule size to N=256 it all fits IPU but I get a memory spike.
# I currently believe memory spike is caused by vector.take (and/or) jax.ops.segment_sum. 
# 
# Instead of doing all of ERI@density_matrix in sparse_mult at one go, we can just 
# have a for loop and to k rows at a time. This will require changing the sparse representation to 
# take a parameter k. 
#