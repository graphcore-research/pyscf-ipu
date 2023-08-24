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

    matrix = jnp.array([(i*N+ j, k*N+ l), (j*N+ i, k*N+ l), (i*N+ j, l*N+ k), (j*N+ i, l*N+ k), 
                        (k*N+ l, i*N+ j), (l*N+ k, i*N+ j), (k*N+ l, j*N+ i), (l*N+ k, j*N+ i) ])
    v      = matrix[symmetry] 

    return v[0], v[1]
vmap_inverse_index = jax.vmap(inverse_index, in_axes=(0, None))

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
    us = jnp.zeros((N**2))
    dm = dm.reshape(-1)

    # loop over all 8 symmetries 
    for symmetry in range(8):   
        # compute indices for take and segment_sum. 
        dm_indices, ss_indices = vmap_inverse_index(nonzero_indices, symmetry) 

        # perform einsum using 8x symmetry and sparsity. 
        dm_values  = jnp.take(dm, dm_indices, axis=0)
        prod       = dm_values * nonzero_distinct_ERI 
        us        += jax.ops.segment_sum(prod, ss_indices, N**2)  
    return us 

backend = "cpu"

# compute number of repetitions per nonzero index
rep = jax.jit(vmap_num_repetitions_fast, backend=backend)(nonzero_indices)

# pre-scale sparse matrix with relevant number of repetitions
nonzero_distinct_ERI = nonzero_distinct_ERI / rep

us = jax.jit(sparse_symmetric_einsum, backend=backend)(nonzero_distinct_ERI, nonzero_indices, dm)
us = np.array(us)

us = us.reshape(N, N)
print(us.reshape(-1)[::51])
print(truth.reshape(-1)[::51])
print(np.max(np.abs(us.reshape(-1) - truth.reshape(-1))))
assert np.allclose(us, truth, atol=1e-6)
print("PASSED!")