import pyscf 
import numpy as np 
import jax 
import jax.numpy as jnp 
jax.config.update('jax_platform_name', "cpu")
jax.config.update('jax_enable_x64', False) 
import argparse 
parser = argparse.ArgumentParser(prog='', description='', epilog='')
parser.add_argument('-backend', default="cpu"),
parser.add_argument('-natm', default=3),
args = parser.parse_args()
backend = args.backend 

natm = int(args.natm) 

mol = pyscf.gto.Mole(atom="".join(f"C 0 {1.54*j} {1.54*i};" for i in range(natm) for j in range(natm))) 
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

nonzero_indices      = np.nonzero(distinct_ERI)[0].astype(np.int32)
nonzero_distinct_ERI = distinct_ERI[nonzero_indices].astype(np.float32)
print("[%i]"%mol.nao_nr())
print("dense: ", dense_ERI.shape, dense_ERI.nbytes/10**6)
print("sparse: ", distinct_ERI.shape, distinct_ERI.nbytes/10**6)
print("sparse+symmetric: ", nonzero_distinct_ERI.shape, nonzero_distinct_ERI.nbytes/10**6)

# todo: check if squareroot causes problems (as seen before). can mitigate using following trick! 
def sqrt(x): return jnp.exp(1/2*jnp.log(x))

def max_val(N):
    x_candidate = (-1 + jnp.sqrt(1 + 8*N)) // 2 # i think above trick leads to overflow! 
    x_candidate = (x_candidate).astype(jnp.int32) # this cast works as floor 
    x = jnp.where(x_candidate * (x_candidate + 1) // 2 <= N, x_candidate, x_candidate - 1)
    return x
max_vals  = jax.vmap(max_val)
def get_i_j(val):
    i = max_val(val)
    j = val - i*(i+1)//2
    return i, j

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

def inverse_index_J_dm(value, symmetry, mask, f): 
    ij = max_val(value*mask[0])
    kl = value - ij*(ij+1)//2
    i, j = get_i_j(ij)
    k, l = get_i_j(kl)

    v      = f(i,j,k,l,symmetry) 
    return v
def inverse_index_J_ss(value, symmetry, mask): 
    ij = max_val(value*mask[0])
    kl = value - ij*(ij+1)//2
    i, j = get_i_j(ij)
    k, l = get_i_j(kl)
    matrix = [k*N+ l, k*N+ l, l*N+ k, l*N+ k ,  i*N+ j, i*N+ j, j*N+ i, j*N+ i ]
    v      = matrix[symmetry] 
    return  v
get_ijkl = jax.vmap(inverse_index_J_dm, in_axes=(0, None, None, None))
vmap_inverse_index_J_ss = jax.vmap(inverse_index_J_ss, in_axes=(0, None, None))

value =jnp.ones(1, dtype=jnp.int32)*10
mask = jnp.ones(1, dtype=jnp.int32)
symmetry = 2

import os.path as osp
from tessellate_ipu import create_ipu_tile_primitive, ipu_cycle_count, tile_map, tile_put_sharded, tile_put_replicated
from functools import partial 


@partial(jax.jit, backend="ipu")
def _vmap_inverse_index_J_dm(value, symmetry, N, mask):
    vertex_filename  = osp.join(osp.dirname(__file__), "compute_indices.cpp")
    compute_indices= create_ipu_tile_primitive(
            "indices",
            "indices",
            inputs=["value", "symmetry", "input_N", "mask", "start", "stop"],
            outputs={"value": 0},
            gp_filename=vertex_filename,
            perf_estimate=100,
    )

    if value.shape[0] <= (1472-1)*6: 
        tiles = tuple((np.arange(0,value.shape[0]) % (value.shape[0]//6) + 1).astype(np.int32).tolist())
        symmetry = jnp.array(symmetry, dtype=jnp.int32).reshape(1,)
        value = tile_put_sharded(value, tiles)
        symmetry = tile_put_replicated(symmetry,   tiles)
        N  = tile_put_replicated(N,   tiles)
        mask  = tile_put_replicated(mask,   tiles)
        index = tile_map(compute_indices, value, symmetry, N, mask )

    else: 
        # loop around them.
        size = value.shape[0]
        print(value.shape)
        total_threads = (1472-1) * 6 
        remainder = size % total_threads
        if remainder != 0: 
            value = jnp.pad(value, (0, total_threads-remainder))

        tiles = tuple((np.arange(0,total_threads) % (1471) + 1).astype(np.int32).tolist())
        value = value.reshape(total_threads, -1) 
        value = tile_put_sharded(value, tiles)
        symmetry = jnp.array(symmetry, dtype=jnp.int32).reshape(1,)
        symmetry = tile_put_replicated(symmetry,   tiles)
        mask     = tile_put_replicated(mask,   tiles)
        N        = tile_put_replicated(N,   tiles)
        start    = tile_put_replicated(0,   tiles)
        stop     = tile_put_replicated(value.shape[1],   tiles)

        step_size = 1 
        print(size//total_threads+1, step_size)
        value = tile_map(compute_indices, value, symmetry, N, mask, start, stop  ) 

    return value.array.reshape(-1)[:size]

def inverse_index_K_dm(value, symmetry, mask): 
    ij = max_val(value*mask[0])
    kl = value - ij*(ij+1)//2
    i, j = get_i_j(ij) 
    k, l = get_i_j(kl)
    # same as _J but "flip 0'th and 2'nd column"
    matrix = [k*N+ j, k*N+ i, l*N+ j, l*N+ i, i*N+ l, i*N+ k, j*N+ l, j*N+ k ]
    v      = matrix[symmetry] 
    return v
def inverse_index_K_ss(value, symmetry, mask): 
    ij = max_val(value*mask[0])
    kl = value - ij*(ij+1)//2
    i, j = get_i_j(ij)
    k, l = get_i_j(kl)
    # same as _J but "flip 0'th and 2'nd column"
    matrix = [i*N+ l, j*N+ l, i*N+ k, j*N+ k, k*N+ j, l*N+ j, k*N+ i, l*N+ i ]
    v      = matrix[symmetry] 
    return  v
vmap_inverse_index_K_dm = jax.vmap(inverse_index_K_dm, in_axes=(0, None, None))
vmap_inverse_index_K_ss = jax.vmap(inverse_index_K_ss, in_axes=(0, None, None))

def sparse_symmetric_einsum(nonzero_distinct_ERI, nonzero_indices, dm, mask, diff_JK):
    mask      = (mask-1)*jnp.sum(diff_JK)*jnp.sum(dm) + mask 

    dm_indices_func_J = lambda i,j,k,l,symmetry: jnp.array([i*N+ j, j*N+ i, i*N+ j, j*N+ i, k*N+ l, l*N+ k, k*N+ l, l*N+ k])[symmetry]
    ss_indices_func_J = lambda i,j,k,l,symmetry: jnp.array([k*N+ l, k*N+ l, l*N+ k, l*N+ k, i*N+ j, i*N+ j, j*N+ i, j*N+ i])[symmetry]
    dm_indices_func_K = lambda i,j,k,l,symmetry: jnp.array([k*N+ j, k*N+ i, l*N+ j, l*N+ i, i*N+ l, i*N+ k, j*N+ l, j*N+ k])[symmetry]
    ss_indices_func_K = lambda i,j,k,l,symmetry: jnp.array([i*N+ l, j*N+ l, i*N+ k, j*N+ k, k*N+ j, l*N+ j, k*N+ i, l*N+ i])[symmetry]

    for symmetry in range(8):   
        # J matrix 
        dm_indices = get_ijkl(nonzero_indices, symmetry, mask, dm_indices_func_J)  
        dm_values  = jnp.take(dm*mask, dm_indices, axis=0)
        prod       = dm_values * nonzero_distinct_ERI* mask 
        mask       = (mask-1)*jnp.sum(prod) + mask 
        ss_indices = get_ijkl(nonzero_indices, symmetry, mask, ss_indices_func_J) 
        diff_JK    = diff_JK + jax.ops.segment_sum(prod, ss_indices, N**2) 
        mask       = (mask-1)*jnp.sum(diff_JK) + mask 
            
        # K matrix 
        dm_indices = get_ijkl(nonzero_indices, symmetry, mask, dm_indices_func_K) 
        dm_values  = jnp.take(dm*mask, dm_indices, axis=0)
        prod       = dm_values * nonzero_distinct_ERI*mask
        mask       = (mask-1)*jnp.sum(prod) + mask 
        ss_indices = get_ijkl(nonzero_indices, symmetry, mask, ss_indices_func_K) 
        diff_JK    = diff_JK - jax.ops.segment_sum(prod, ss_indices, N**2) / 2 * HYB_B3LYP 
        mask       = (mask-1)*jnp.sum(diff_JK) + mask 

    return diff_JK

# Compute number of repetitions per nonzero index and re-scale ERI. 
rep = jax.jit(vmap_num_repetitions_fast, backend="cpu")(nonzero_indices)
nonzero_distinct_ERI = nonzero_distinct_ERI / rep

mask = np.ones(1)
dm = dm.reshape(-1)
diff_JK = np.zeros(dm.shape)
diff_JK = jax.jit(sparse_symmetric_einsum, backend=backend)(nonzero_distinct_ERI, nonzero_indices, dm, mask, diff_JK)
diff_JK = np.array(diff_JK)

diff_JK = diff_JK.reshape(N, N)
print(diff_JK.reshape(-1)[::51])
print(truth.reshape(-1)[::51])
print(np.max(np.abs(diff_JK.reshape(-1) - truth.reshape(-1))))
assert np.allclose(diff_JK, truth, atol=1e-6)
print("PASSED!")