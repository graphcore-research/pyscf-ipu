from tqdm import tqdm 
import pyscf 
import numpy as np 
import jax 
import jax.numpy as jnp 
jax.config.update('jax_platform_name', "cpu")
jax.config.update('jax_enable_x64', False) 

mol = pyscf.gto.Mole(atom="".join(f"C 0 0 {1.54*i};" for i in range(4))) 

mol.build()
dense_ERI = mol.intor("int2e_sph", aosym="s1")
distinct_ERI = mol.intor("int2e_sph", aosym="s8")
N = mol.nao_nr()
dm = pyscf.scf.hf.init_guess_by_minao(mol)         
truth = np.einsum("ijkl,ji->kl", dense_ERI, dm)

nonzero_indices      = np.nonzero(distinct_ERI)[0]
nonzero_distinct_ERI = distinct_ERI[nonzero_indices]

us = np.zeros((N,N))

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

def inverse_index(value, symmetry): 
    ij = max_val(value)
    kl = value - ij*(ij+1)//2
    i, j = get_i_j(ij)
    k, l = get_i_j(kl)

    N1 = N+1
    matrix = jnp.array([(i*N1+ j, k*N+ l), (j*N1+ i, k*N+ l), (i*N1+ j, l*N+ k), (j*N1+ i, l*N+ k), 
                        (k*N1+ l, i*N+ j), (l*N1+ k, i*N+ j), (k*N1+ l, j*N+ i), (l*N1+ k, j*N+ i) ])
    v      = matrix[symmetry] 
    
    is_in_prev = jnp.any( jnp.all( matrix[:symmetry] == v, axis=-1))

    return (1-is_in_prev) * v[0] - is_in_prev , v[1]
compute_indices = jax.vmap(inverse_index, in_axes=(0, None))

def sparse_symmetric_einsum(nonzero_distinct_ERI, nonzero_indices, dm):
    us = jnp.zeros((N**2))
    dm = jnp.pad(dm, ((0, 0), (0,1)))
    dm = dm.reshape(-1)

    # loop over all 8 symmetries 
    for symmetry in range(8):   
        # compute indices for take and segment_sum. 
        dm_indices, ss_indices = compute_indices(nonzero_indices, symmetry) 

        # perform einsum using 8x symmetry and sparsity. 
        dm_values  = jnp.take(dm, dm_indices, axis=0)
        prod       = dm_values * nonzero_distinct_ERI 
        us        += jax.ops.segment_sum(prod, ss_indices, N**2)  
    return us 

backend = "cpu"

us = jax.jit(sparse_symmetric_einsum, backend=backend)(nonzero_distinct_ERI, nonzero_indices, dm)
us = np.array(us)

us = us.reshape(N, N)
print(us.reshape(-1)[::51])
print(truth.reshape(-1)[::51])
print(np.max(np.abs(us.reshape(-1) - truth.reshape(-1))))
assert np.allclose(us, truth, atol=1e-6)
print("PASSED!")