import pyscf 
import numpy as np 
import jax 
import jax.numpy as jnp 
jax.config.update('jax_platform_name', "cpu")
jax.config.update('jax_enable_x64', False) 
import argparse 
parser = argparse.ArgumentParser(prog='', description='', epilog='')
parser.add_argument('-backend', default="cpu"),
args = parser.parse_args()
backend = args.backend 

mol = pyscf.gto.Mole(atom="".join(f"C 0 {1.54*j} {1.54*i};" for i in range(3) for j in range(4))) 
mol.build()
dense_ERI = mol.intor("int2e_sph", aosym="s1")
distinct_ERI = mol.intor("int2e_sph", aosym="s8")
N = mol.nao_nr()
dm = pyscf.scf.hf.init_guess_by_minao(mol)         
HYB_B3LYP = 0.2
scale = HYB_B3LYP/2
J = np.einsum("ijkl,ji->kl", dense_ERI, dm)
K = np.einsum("ijkl,jk->il", dense_ERI, dm)
truth = J - K / 2 * HYB_B3LYP

nonzero_indices      = np.nonzero(distinct_ERI)[0]
nonzero_distinct_ERI = distinct_ERI[nonzero_indices]
print("[%i]"%mol.nao_nr())
print("dense: ", dense_ERI.shape, dense_ERI.nbytes/10**6/2)
print("sparse: ", distinct_ERI.shape, distinct_ERI.nbytes/10**6/2)
print("sparse+symmetric: ", nonzero_distinct_ERI.shape, nonzero_distinct_ERI.nbytes/10**6/2)

def max_val(N):
    x_candidate = (-1 + jnp.sqrt(1 + 8*N)) / 2
    x_candidate = jnp.floor(x_candidate)
    x = jnp.where(x_candidate * (x_candidate + 1) // 2 <= N, x_candidate, x_candidate - 1)
    return x.astype(jnp.int32)
max_vals  = jax.vmap(max_val)
def get_i_j(val):
    i = max_val(val)
    j = val - i*(i+1)//2
    return i, j
def inverse_index_J(value, symmetry): 
    ij = max_val(value)
    kl = value - ij*(ij+1)//2
    i, j = get_i_j(ij)
    k, l = get_i_j(kl)
    matrix = [(i*N+ j, k*N+ l), (j*N+ i, k*N+ l), (i*N+ j, l*N+ k), (j*N+ i, l*N+ k),  
                        (k*N+ l, i*N+ j), (l*N+ k, i*N+ j), (k*N+ l, j*N+ i), (l*N+ k, j*N+ i) ]
    v      = matrix[symmetry] 
    return v[0], v[1]
vmap_inverse_index_J = jax.vmap(inverse_index_J, in_axes=(0, None))

def inverse_index_K(value, symmetry): 
    ij = max_val(value)
    kl = value - ij*(ij+1)//2
    i, j = get_i_j(ij)
    k, l = get_i_j(kl)
    # same as _J but "flip 0'th and 2'nd column"
    matrix = [(k*N+ j, i*N+ l), (k*N+ i, j*N+ l), (l*N+ j, i*N+ k), (l*N+ i, j*N+ k), 
              (i*N+ l, k*N+ j), (i*N+ k, l*N+ j), (j*N+ l, k*N+ i), (j*N+ k, l*N+ i) ]
    v      = matrix[symmetry] 
    return v[0], v[1]
vmap_inverse_index_K = jax.vmap(inverse_index_K, in_axes=(0, None))

def num_repetitions_fast(value):
    ij = max_val(value)
    kl = value - ij*(ij+1)//2
    i, j = get_i_j(ij)
    k, l = get_i_j(kl)

    # compute: repetitions = 2^((i==j) + (k==l) + (k==i and l==j or k==j and l==i))
    repetitions = 2**(
        jnp.equal(i,j).astype(jnp.int32) + 
        jnp.equal(k,l).astype(jnp.int32) + 
        (1 - ((1 - jnp.equal(k,i) * jnp.equal(l,j)) * 
        (1- jnp.equal(k,j) * jnp.equal(l,i))).astype(jnp.int32))
    )

    return repetitions
vmap_num_repetitions_fast = jax.vmap(num_repetitions_fast, in_axes=(0))

def sparse_symmetric_einsum(nonzero_distinct_ERI, nonzero_indices, dm):
    diff_JK = jnp.zeros((N**2))
    dm = dm.reshape(-1)

    # loop over all 8 symmetries 
    for symmetry in range(8):   
        # compute indices for take and segment_sum. 
        dm_indices, ss_indices = vmap_inverse_index_J(nonzero_indices, symmetry)  # may introduce additional overhead? 

        # perform einsum using 8x symmetry and sparsity. 
        dm_values = jnp.take(dm, dm_indices, axis=0)
        prod      = dm_values * nonzero_distinct_ERI 
        diff_JK   = diff_JK + jax.ops.segment_sum(prod, ss_indices, N**2)
                
    # loop over all 8 symmetries 
    for symmetry in range(8):   
        # compute indices for take and segment_sum. 
        dm_indices, ss_indices = vmap_inverse_index_K(nonzero_indices, symmetry) 

        # perform einsum using 8x symmetry and sparsity. 
        dm_values = jnp.take(dm, dm_indices, axis=0)
        prod      = dm_values * nonzero_distinct_ERI 
        diff_JK   = diff_JK - jax.ops.segment_sum(prod, ss_indices, N**2) / 2 * HYB_B3LYP 

    return diff_JK

# Compute number of repetitions per nonzero index and re-scale ERI. 
rep = jax.jit(vmap_num_repetitions_fast, backend="cpu")(nonzero_indices)
nonzero_distinct_ERI = nonzero_distinct_ERI / rep

diff_JK = jax.jit(jax.checkpoint(sparse_symmetric_einsum), backend=backend)(nonzero_distinct_ERI, nonzero_indices, dm)
diff_JK = np.array(diff_JK)

diff_JK = diff_JK.reshape(N, N)
print(diff_JK.reshape(-1)[::51])
print(truth.reshape(-1)[::51])
print(np.max(np.abs(diff_JK.reshape(-1) - truth.reshape(-1))))
assert np.allclose(diff_JK, truth, atol=1e-6)
print("PASSED!")