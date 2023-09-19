# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import numpy as np 
import jax.numpy as jnp
import os 
import pyscf 
import jax
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_platform_name', "cpu")
os.environ['OMP_NUM_THREADS'] = "16"

# Construct molecule we can use for test case. 
mol = pyscf.gto.Mole(atom=[["C", (0, 0, i)] for i in range(8)], basis="sto3g")
mol.build()
N              = mol.nao_nr()                          # N: number of atomic orbitals (AO) 
density_matrix = pyscf.scf.hf.init_guess_by_minao(mol) # (N, N)
ERI            = mol.intor("int2e_sph")                # (N, N, N, N)
print(ERI.shape, density_matrix.shape)

# ERI satisfies the following symmetry where we can interchange (k,l) with (l,k)
#   ERI[i,j,k,l]=ERI[i,j,l,k]
i,j,k,l = 5,10,20,25
print(ERI[i,j,k,l], ERI[i,j,l,k])
assert np.allclose(ERI[i,j,k,l], ERI[i,j,l,k])

# In turns out all of the following indices can be interchagned! 
#   ERI[ijkl]=ERI[ijlk]=ERI[jikl]=ERI[jilk]=ERI[lkij]=ERI[lkji]=ERI[lkij]=ERI[lkji]
print(ERI[i,j,k,l], ERI[i,j,l,k], ERI[j,i,k,l], ERI[j,i,l,k], ERI[l,k,i,j], ERI[l,k,j,i], ERI[l,k,i,j], ERI[l,k,j,i])

# Recall sparse_ERI.py uses the following matrix vector multiplication.
ERI            = ERI.reshape(N**2, N**2)
density_matrix = density_matrix.reshape(N**2)
truth          = ERI @ density_matrix

# But most of ERI are zeros! We therefore use a sparse matrix multiplicaiton (See below)
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
res = jax.jit(sparse_mult, backend="cpu")(sparse_ERI, density_matrix)

assert np.allclose(truth, res)

# Here's the problem. 
# Most entries in ERI are repeated 8 times due to ERI[i,j,k,l]=ERI[i,j,l,k]=...
# We can therefore save 8x memory by only storing each element once! 
# When we do matrix vector multiplication and need E[i,j,l,k] we then just look at ERI[i,j,k,l] instead. 
# This is what the ipu_einsum does in nanoDFT. 
# After looking at sparse_ERI.py, I think we may be able to do this using the same take/segment_sum tricks! 
# Because of this I think we may get an sufficiently efficient implementation in Jax (we may later choose to move it to poplar). 

# Potentially tricky part: 
# My first idea was to make a dictionary that maps ijkl -> ijlk (and so on). 
# We should do this, it's an useful exercise. Unfortunately, this will take as much memory as storing ERI, 
# so we wont win anything. We thus have to instead compute the ijkl translation on the go. 
# A further caveat is that we'll need to sequentialize; if we make the segment sum over the entire thing this will also take too much memory. 