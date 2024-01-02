# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import pyscf 
import numpy as np 
import jax 
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp 
from functools import partial 
from icecream import ic
HYB_B3LYP = 0.2

def get_i_j(val):
    i = (np.sqrt(1 + 8*val.astype(np.uint64)) - 1)//2 # no need for floor, integer division acts as floor. 
    j = (((val - i) - (i**2 - val))//2)
    return i, j

def _ijkl(value, symmetry, N, f):
    #i, j, k, l = value[0].astype(np.uint32), value[1].astype(np.uint32), value[2].astype(np.uint32), value[3].astype(np.uint32)
    i, j, k, l = value[0], value[1], value[2], value[3]
    return f(i,j,k,l,symmetry,N)
ijkl = jax.vmap(_ijkl, in_axes=(0, None, None, None))

def np_ijkl(value, symmetry, N, f):
    #i, j, k, l = value[0].astype(np.uint32), value[1].astype(np.uint32), value[2].astype(np.uint32), value[3].astype(np.uint32)
    i, j, k, l = value[:, 0], value[:, 1], value[:, 2], value[:, 3]
    return f(i,j,k,l,symmetry,N)


def num_repetitions_fast(ij, kl):
    i, j = get_i_j(ij)
    k, l = get_i_j(kl)

    # compute: repetitions = 2^((i==j) + (k==l) + (k==i and l==j or k==j and l==i))
    repetitions = 2**(
        np.equal(i,j).astype(np.uint64) + 
        np.equal(k,l).astype(np.uint64) + 
        (1 - ((1 - np.equal(k,i) * np.equal(l,j)) * 
        (1- np.equal(k,j) * np.equal(l,i))).astype(np.uint64))
    )
    return repetitions

indices_func = lambda i,j,k,l,symmetry,N: jnp.array([i*N+j, j*N+i, i*N+j, j*N+i, k*N+l, l*N+k, k*N+l, l*N+k,
                                                     k*N+l, k*N+l, l*N+k, l*N+k, i*N+j, i*N+j, j*N+i, j*N+i,
                                                     k*N+j, k*N+i, l*N+j, l*N+i, i*N+l, i*N+k, j*N+l, j*N+k,
                                                     i*N+l, j*N+l, i*N+k, j*N+k, k*N+j, l*N+j, k*N+i, l*N+i])[symmetry]

def _indices_func(i, j, k, l, symmetry, N):
    if symmetry == 0: return i * N + j
    elif symmetry == 1: return j * N + i
    elif symmetry == 2: return i * N + j
    elif symmetry == 3: return j * N + i
    elif symmetry == 4: return k * N + l
    elif symmetry == 5: return l * N + k
    elif symmetry == 6: return k * N + l
    elif symmetry == 7: return l * N + k
    elif symmetry == 8 or symmetry == 9: return k * N + l
    elif symmetry == 10 or symmetry == 11: return l * N + k
    elif symmetry == 12 or symmetry == 13: return i * N + j
    elif symmetry == 14 or symmetry == 15: return j * N + i
    elif symmetry == 16: return k * N + j
    elif symmetry == 17: return k * N + i
    elif symmetry == 18: return l * N + j
    elif symmetry == 19: return l * N + i
    elif symmetry == 20: return i * N + l
    elif symmetry == 21: return i * N + k
    elif symmetry == 22: return j * N + l
    elif symmetry == 23: return j * N + k
    elif symmetry == 24: return i * N + l #j*N+l, i*N+k, j*N+k,
    elif symmetry == 25: return j*N+l 
    elif symmetry == 26: return i*N+k
    elif symmetry == 27: return j*N+k
    elif symmetry == 28: return k * N + j
    elif symmetry == 29: return l * N + j
    elif symmetry == 30: return k * N + i
    elif symmetry == 31: return l * N + i


def sparse_symmetric_einsum(nonzero_distinct_ERI, nonzero_indices, dm, foriloop):
    dm = dm.reshape(-1)
    diff_JK = jnp.zeros(dm.shape)
    N = int(np.sqrt(dm.shape[0]))

    dnums = jax.lax.GatherDimensionNumbers(
        offset_dims=(), 
        collapsed_slice_dims=(0,),
        start_index_map=(0,))
    scatter_dnums = jax.lax.ScatterDimensionNumbers(
    update_window_dims=(), 
    inserted_window_dims=(0,),
    scatter_dims_to_operand_dims=(0,))
    Z = jnp.zeros((N**2,), dtype=dm.dtype)

    # todo: how much faster if we precompute dm/ss indices? 
    def iteration(symmetry, vals): 
        diff_JK = vals 
        is_K_matrix = (symmetry >= 8)

        def sequentialized_iter(i, vals):
            # Generalized J/K computation: does J when symmetry is in range(0,8) and K when symmetry is in range(8,16)
            # Trade-off: Using one function leads to smaller always-live memory.
            diff_JK = vals 
            indices = nonzero_indices[i]#.astype(np.int32) # 
            eris    = nonzero_distinct_ERI[i]

            dm_indices = ijkl(indices, symmetry+is_K_matrix*8, N, indices_func).reshape(-1, 1)
            #dm_values = jnp.take(dm, dm_indices, axis=0)[:, 0] # for our special case the 50 lines of code reduces to the one line below. 
            dm_values = jax.lax.gather(dm, dm_indices, dimension_numbers=dnums, slice_sizes=(1,), mode=jax.lax.GatherScatterMode.FILL_OR_DROP)
            dm_values = dm_values * eris  
            
            ss_indices = ijkl(indices, symmetry+8+is_K_matrix*8, N, indices_func) .reshape(-1,1)
            # diff_JK = diff_JK + jax.lax.segment_sum( ...) # for our special case the 100 lines of code reduces to the one line below. 
            diff_JK = diff_JK + jax.lax.scatter_add(Z,
                                            ss_indices, dm_values, 
                                            scatter_dnums, indices_are_sorted=True, unique_indices=False, mode=jax.lax.GatherScatterMode.FILL_OR_DROP)\
                                *(-HYB_B3LYP/2)**is_K_matrix
            
            return diff_JK

        batches = nonzero_indices.shape[0] 

        # forloop makes training slower but compile time faster. 
        if foriloop: 
            diff_JK = jax.lax.fori_loop(0, batches, sequentialized_iter, diff_JK) 
        else:
            for i in range(batches):
                diff_JK = sequentialized_iter(i, diff_JK)
        return diff_JK

    if foriloop: 
        diff_JK = jax.lax.fori_loop(0, 16, iteration, diff_JK) 
    else:
        for i in range(0, 16): 
            diff_JK = iteration(i, diff_JK)
    #diff_JK = jax.lax.fori_loop(0, 16, iteration, diff_JK) 
    #return jax.lax.psum(diff_JK, axis_name="p")
    return diff_JK.reshape(N, N)

    
def sparse_einsum(nonzero_distinct_ERI, precomputed_indices, dm, foriloop):
    dm = dm.reshape(-1)
    diff_JK = jnp.zeros(dm.shape)
    N = int(np.sqrt(dm.shape[0]))

    dnums = jax.lax.GatherDimensionNumbers( offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,))
    scatter_dnums = jax.lax.ScatterDimensionNumbers( update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,))
    Z = jnp.zeros((N**2,), dtype=dm.dtype)

    def iteration(symmetry, vals): 
        diff_JK = vals 
        is_K_matrix = (symmetry >= 8)

        def sequentialized_iter(i, vals):
            # Generalized J/K computation: does J when symmetry is in range(0,8) and K when symmetry is in range(8,16)
            # Trade-off: Using one function leads to smaller always-live memory.
            diff_JK = vals 
            eris    = nonzero_distinct_ERI[i]

            #dm_values = jnp.take(dm, dm_indices, axis=0)[:, 0] # for our special case the 50 lines of code reduces to the one line below. 
            dm_indices = precomputed_indices[symmetry, i, 0]
            ss_indices = precomputed_indices[symmetry, i, 1]
            dm_values = jax.lax.gather(dm, dm_indices, dimension_numbers=dnums, slice_sizes=(1,), mode=jax.lax.GatherScatterMode.FILL_OR_DROP)
            dm_values = dm_values * eris  

            #ss_indices = ijkl(indices, symmetry+8+is_K_matrix*8, N, indices_func) .reshape(-1,1)
            # diff_JK = diff_JK + jax.lax.segment_sum( ...) # for our special case the 100 lines of code reduces to the one line below. 
            diff_JK = diff_JK + jax.lax.scatter_add(Z, ss_indices, dm_values, 
                                            scatter_dnums, indices_are_sorted=True, unique_indices=False, mode=jax.lax.GatherScatterMode.FILL_OR_DROP)\
                                *(-HYB_B3LYP/2)**is_K_matrix
            
            return diff_JK

        batches = precomputed_indices.shape[1] 

        # forloop makes training slower but compile time faster. 
        if foriloop: 
            diff_JK = jax.lax.fori_loop(0, batches, sequentialized_iter, diff_JK) 
        else:
            for i in range(batches):
                diff_JK = sequentialized_iter(i, diff_JK)
        return diff_JK

    if foriloop: 
        diff_JK = jax.lax.fori_loop(0, 16, iteration, diff_JK) 
    else:
        for i in range(0, 16): 
            diff_JK = iteration(i, diff_JK)
    #diff_JK = jax.lax.fori_loop(0, 16, iteration, diff_JK) 
    #return jax.lax.psum(diff_JK, axis_name="p")
    return diff_JK.reshape(N, N)

    
    
def precompute_indices(nonzero_indices, N):

    def iteration(symmetry): 
        is_K_matrix = (symmetry >= 8)

        def sequentialized_iter(i):
            # Generalized J/K computation: does J when symmetry is in range(0,8) and K when symmetry is in range(8,16)
            # Trade-off: Using one function leads to smaller always-live memory.
            indices = nonzero_indices[i]
            dm_indices = np_ijkl(indices, symmetry+is_K_matrix*8, N, _indices_func).reshape(-1, 1)
            ss_indices = np_ijkl(indices, symmetry+8+is_K_matrix*8, N, _indices_func) .reshape(-1,1)

            return dm_indices, ss_indices 

        batches = nonzero_indices.shape[0] 

        # forloop makes training slower but compile time faster. 
        _indices = [None for _ in range(batches)]
        for i in range(batches):
            _indices[i] = sequentialized_iter(i)
        return _indices

    _indices = [None for _ in range(16)]
    for i in range(0, 16): 
        _indices[i] = iteration(i)
    return np.array(_indices )

if __name__ == "__main__": 
    import time 
    import argparse 
    parser = argparse.ArgumentParser(prog='', description='', epilog='')
    parser.add_argument('-backend', default="cpu"),
    parser.add_argument('-natm', default=3),
    parser.add_argument('-test', action="store_true")
    parser.add_argument('-prof', action="store_true")
    parser.add_argument('-batches', default=5)
    parser.add_argument('-skip', action="store_true") 
    
    args = parser.parse_args()
    backend = args.backend 

    natm = int(args.natm) 
    nipu = 1

    start = time.time()

    mol = pyscf.gto.Mole(atom="".join(f"C 0 {1.54*j} {1.54*i};" for i in range(natm) for j in range(natm))) 
    #mol = pyscf.gto.Mole(atom="".join(f"C 0 {15.4*j} {15.4*i};" for i in range(1) for j in range(75))) 
    mol.build()
    N = mol.nao_nr()
    print("N %i"%mol.nao_nr())
    print("NxN:", (N**2, N**2))
    print("Naive operations: ", N**4*2/10**9, "[Giga]")
    if not args.skip: dense_ERI = mol.intor("int2e_sph", aosym="s1")
    distinct_ERI = mol.intor("int2e_sph", aosym="s8")
    #distinct_ERI[np.abs(distinct_ERI)<1e-9] = 0  # zero out stuff 
    dm = pyscf.scf.hf.init_guess_by_minao(mol)         
    scale = HYB_B3LYP/2
    if not args.skip: 
        J = np.einsum("ijkl,ji->kl", dense_ERI, dm)
        K = np.einsum("ijkl,jk->il", dense_ERI, dm)
        truth = J - K / 2 * HYB_B3LYP

    nonzero_indices      = np.nonzero(distinct_ERI)[0].astype(np.uint64) 
    nonzero_distinct_ERI = distinct_ERI[nonzero_indices]#.astype(np.float32)
    print("Nonzero Operations:", nonzero_indices.size*8*2/10**9, "[Giga]")
    ij, kl               = get_i_j(nonzero_indices)
    rep                  = num_repetitions_fast(ij, kl)
    nonzero_distinct_ERI = nonzero_distinct_ERI / rep
    dm                   = dm.reshape(-1)
    diff_JK              = np.zeros(dm.shape)

    batches  = int(args.batches) 
    remainder = nonzero_indices.shape[0] % (nipu*batches)

    if remainder != 0:
        print(nipu*batches-remainder, ij.shape)
        ij = np.pad(ij, ((0,nipu*batches-remainder)))
        kl = np.pad(kl, ((0,nipu*batches-remainder)))
        nonzero_distinct_ERI = np.pad(nonzero_distinct_ERI, (0,nipu*batches-remainder))

    ij = ij.reshape(nipu, batches, -1)
    kl = kl.reshape(nipu, batches, -1)
    nonzero_distinct_ERI = nonzero_distinct_ERI.reshape(nipu, batches, -1)

    i, j = get_i_j(ij.reshape(-1))
    k, l  = get_i_j(kl.reshape(-1))
    nonzero_indices = np.vstack([i,j,k,l]).T.reshape(nipu, batches, -1, 4).astype(np.int32)
    #nonzero_indices = jax.lax.bitcast_convert_type(nonzero_indices, np.float16)

    #diff_JK = jax.pmap(sparse_symmetric_einsum, in_axes=(0,0,None,None), static_broadcasted_argnums=(3,), backend=backend, axis_name="p")(nonzero_distinct_ERI, nonzero_indices, dm, args.backend) 
    diff_JK = jax.jit(sparse_symmetric_einsum, static_argnums=(3,), backend=backend)(nonzero_distinct_ERI[0], nonzero_indices[0], dm, args.backend) 
    #diff_JK = jax.jit(sparse_symmetric_einsum, backend=backend, static_argnums=(3,))(nonzero_distinct_ERI[0], nonzero_indices[0], dm, False) 

    indices = precompute_indices(nonzero_indices[0], N)
    print(np.max(indices)) # this is just N**2! 
    indices = indices.astype(np.int16)
    print(np.max(indices))
    print(nonzero_distinct_ERI.nbytes/10**9, nonzero_indices.nbytes/10**9, indices.nbytes/10**9) 
    print(nonzero_distinct_ERI.shape, nonzero_indices.shape, indices.shape) 
    print(np.max(indices))

    _diff_JK = jax.jit(sparse_einsum, static_argnums=(3,), backend=backend)(nonzero_distinct_ERI[0], indices, dm, args.backend) 


    if args.skip: 
        exit()

    diff_JK = diff_JK.reshape(N, N)
    print(diff_JK.reshape(-1)[::51])
    print(truth.reshape(-1)[::51])
    print(np.max(np.abs(diff_JK.reshape(-1) - truth.reshape(-1))))
    print(np.max(np.abs(_diff_JK.reshape(-1) - truth.reshape(-1))))
    assert np.allclose(diff_JK, truth, atol=1e-6)
    assert np.allclose(_diff_JK, truth, atol=1e-6)
    print("PASSED!")
    