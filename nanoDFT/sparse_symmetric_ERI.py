# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import pyscf 
import numpy as np 
import jax 
import jax.numpy as jnp 
import os.path as osp
from tessellate_ipu import create_ipu_tile_primitive, ipu_cycle_count, tile_map, tile_put_sharded, tile_put_replicated
from functools import partial 
jax.config.update('jax_platform_name', "cpu")
jax.config.update("jax_debug_nans", True)
HYB_B3LYP = 0.2

import numpy as np 
import jax 
import jax.numpy as jnp 

def matmul_cumsum_jax(arr):
    return jnp.tril(jnp.ones((len(arr), len(arr)))) @ arr 

def cumsum_jax(arr):
    chunk_size = 2**7 
    original_shape = arr.shape 
    padding = chunk_size - (len(arr) % chunk_size) if len(arr) % chunk_size != 0 else 0
    arr = jnp.pad(arr, (0, padding))  
    num_chunks = -(-len(arr) // chunk_size) 
    chunks = arr.T.reshape(num_chunks, chunk_size) 
    chunks = jax.vmap(matmul_cumsum_jax)(chunks)
    offset = 0
    offsets = [offset]
    for i, chunk in enumerate(chunks):
        offset += chunk[-1]
        offsets.append(offset)
    chunks = jax.vmap(jax.lax.add, in_axes=(0,0))(chunks.astype(jnp.int32), jnp.array(offsets[:-1], dtype=np.int32))
    return jnp.concatenate(chunks).reshape(-1)[:original_shape[0]]

def max_val(N):
    x_candidate = (-1 + jnp.sqrt(1 + 8*N.astype(np.int64))) // 2 
    x = jnp.where(x_candidate * (x_candidate + 1) // 2 <= N, x_candidate, x_candidate - 1)
    return x
max_vals  = jax.vmap(max_val)
def get_i_j(val):
    i = max_val(val)
    j = val - i*(i+1)//2
    return i, j

def cpu_ijkl(value, symmetry, f): 
    ij = max_val(value)
    kl = value - ij*(ij+1)//2
    i, j = get_i_j(ij)
    k, l = get_i_j(kl)
    v      = f(i,j,k,l,symmetry) 
    return v.astype(np.int64), (i,j,k,l)
cpu_ijkl = jax.vmap(cpu_ijkl, in_axes=(0, None, None))


@partial(jax.jit, backend="ipu")
def ipu_ijkl(nonzero_indices, symmetry, N):
    vertex_filename  = osp.join(osp.dirname(__file__), "compute_indices.cpp")
    compute_indices= create_ipu_tile_primitive(
            "PairSymmetryIndices",
            "PairSymmetryIndices",
            inputs=["value_low", "value_high", "symmetry", "input_N", "start", "stop"],
            outputs={"out": 0},
            gp_filename=vertex_filename,
            perf_estimate=100,
    )
    nonzero_indices_low = nonzero_indices[0]
    nonzero_indices_high = nonzero_indices[1]

    size = np.prod(nonzero_indices_low.shape)
    total_threads = (1472-1) * 6 
    remainder = size % total_threads
    if remainder != 0: 
        nonzero_indices_low = jnp.pad(nonzero_indices_low, (0, total_threads-remainder))
        nonzero_indices_high = jnp.pad(nonzero_indices_high, (0, total_threads-remainder))
    nonzero_indices_low = nonzero_indices_low.reshape(total_threads, -1) 
    nonzero_indices_high = nonzero_indices_high.reshape(total_threads, -1) 

    tiles = tuple((np.arange(0,total_threads) % (1471) + 1).astype(np.int32).tolist())
    nonzero_indices_high = tile_put_sharded(nonzero_indices_high, tiles)
    nonzero_indices_low = tile_put_sharded(nonzero_indices_low, tiles)
    symmetry = jnp.array(symmetry, dtype=jnp.int32).reshape(1,)
    symmetry = tile_put_replicated(symmetry,   tiles) 
    N        = tile_put_replicated(N,   tiles)
    start    = tile_put_replicated(0,   tiles)
    stop     = tile_put_replicated(nonzero_indices_low.shape[1],   tiles)

    print(nonzero_indices_low.dtype, nonzero_indices_high.dtype, symmetry.dtype, N.dtype, start.dtype,stop.dtype)

    value = tile_map(compute_indices, nonzero_indices_low, nonzero_indices_high, symmetry, N, start, stop)  

    return value.array.reshape(-1)[:size]

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

def sparse_symmetric_einsum(nonzero_distinct_ERI, nonzero_indices, offset, dm, backend, nipu, pad_length):
    dm = dm.reshape(-1)
    diff_JK = jnp.zeros(dm.shape)
    N = int(np.sqrt(dm.shape[0]))

    assert nonzero_indices.dtype == np.int32, nonzero_indices.dtype
    assert offset[0].dtype == np.int32, offset[0].dtype
    assert offset[1].dtype == np.int32, offset[1].dtype
    #nonzero_indices = jnp.cumsum(nonzero_indices, axis=1) 
    
    if backend == "cpu":
        indices_func = lambda i,j,k,l,symmetry: jnp.array([i*N+j, j*N+i, i*N+j, j*N+i, k*N+l, l*N+k, k*N+l, l*N+k,
                                                        k*N+l, k*N+l, l*N+k, l*N+k, i*N+j, i*N+j, j*N+i, j*N+i,
                                                        k*N+j, k*N+i, l*N+j, l*N+i, i*N+l, i*N+k, j*N+l, j*N+k,
                                                        i*N+l, j*N+l, i*N+k, j*N+k, k*N+j, l*N+j, k*N+i, l*N+i])[symmetry]

    def iteration(symmetry, vals): 
        diff_JK = vals 
        is_K_matrix = (symmetry >= 8)

        def sequentialized_iter(i, vals):
            diff_JK = vals 
            eris    = nonzero_distinct_ERI[i]

            def add_int64(a, b):
                # Unpack (low, high) pairs
                low_a, high_a = a
                low_b, high_b = b
                
                # Perform the addition on the low 32 bits
                low_sum = jnp.int32(low_a) + jnp.int32(low_b)
                
                # Calculate the carry: 1 if overflow occurred, otherwise 0
                carry = jnp.int32((low_sum < low_a) | (low_sum < low_b))
                
                # Add the high 32 bits, along with the carry
                high_sum = jnp.int32(high_a) + jnp.int32(high_b) + carry
                
                return (jnp.int32(low_sum), jnp.int32(high_sum))

            print("tracing post cumsum")
            if backend == "gpu": 
                cumsum_low  = jnp.cumsum(nonzero_indices[i]) 
                cumsum_high = jnp.cumsum(cumsum_low<nonzero_indices[i])
            else: 
                cumsum_low  = cumsum_jax(nonzero_indices[i]) 
                cumsum_high = cumsum_jax(cumsum_low<nonzero_indices[i])

            low, high = add_int64( (cumsum_low, cumsum_high), (offset[0][i], offset[0][i]) )

            if backend == "cpu": 
                indices = recover(low, high)
                indices = indices.astype(np.int64)
                dm_indices, ijkl = cpu_ijkl(indices, symmetry+is_K_matrix*8, indices_func)  
            else:                
                dm_indices = ipu_ijkl((low, high), symmetry+is_K_matrix*8, N)  

            dm_values  = jnp.take(dm, dm_indices, axis=0) 
            dm_values = dm_values.at[:].mul( eris ) # this is prod, but re-use variable for inplace update. 
            
            if backend == "cpu": 
                ss_indices, _ = cpu_ijkl(indices, symmetry+8+is_K_matrix*8, indices_func) 
            else:                
                ss_indices = ipu_ijkl((low, high), symmetry+8+is_K_matrix*8, N) 
            diff_JK    = diff_JK + jax.ops.segment_sum(dm_values, ss_indices, N**2) * (-HYB_B3LYP/2)**is_K_matrix # causes peak 2 
            
            return diff_JK

        batches = nonzero_indices.shape[0] # before pmap, tensor had shape (nipus, batches, -1) so [0]=batches after pmap

        diff_JK = jax.lax.fori_loop(0, batches, sequentialized_iter, diff_JK) 
        #for i in range(batches):
        #    diff_JK = sequentialized_iter(i, diff_JK)
        return diff_JK

    diff_JK = jax.lax.fori_loop(0, 16, iteration, diff_JK) 
    #for i in range(16):
    #    diff_JK = iteration(i, diff_JK)

    if nipu > 1: return jax.lax.psum(diff_JK, axis_name="p")
    else: return diff_JK 

if __name__ == "__main__":
    import time 
    import argparse 
    parser = argparse.ArgumentParser(prog='', description='', epilog='')
    parser.add_argument('-backend', default="cpu"),
    parser.add_argument('-natm', default=3),
    parser.add_argument('-test', action="store_true")
    parser.add_argument('-prof', action="store_true")
    parser.add_argument('-batches', default=5)
    parser.add_argument('-nipu', default=16, type=int)
    parser.add_argument('-skip', action="store_true") 
    parser.add_argument('-edge', action="store_true") 
    args = parser.parse_args()
    backend = args.backend 

    natm = int(args.natm) 
    nipu = int(args.nipu)
    if backend == "cpu": 
        nipu = 1
        jax.config.update('jax_enable_x64', True) 
    else: 
        jax.config.update('jax_enable_x64', False) 

    start = time.time()
    #mol = pyscf.gto.Mole(atom="".join(f"C 0 {1.54*j} {1.54*i};" for i in range(natm) for j in range(natm)))
    # smallest case which overflows in int32! 
    if args.edge:  mol = pyscf.gto.Mole(atom="".join(f"C 0 {15.4*j} {15.4*i};" for i in range(1) for j in range(65))) 
    else:          mol = pyscf.gto.Mole(atom="".join(f"C 0 {15.4*j} {15.4*i};" for i in range(1) for j in range(3))) 

    mol.build()
    N = mol.nao_nr()
    print("[%i]"%mol.nao_nr())

    print("distinct_ERI", time.time()-start, end="")
    distinct_ERI = mol.intor("int2e_sph", aosym="s8")
    print(" ", distinct_ERI.size/10**9, 2**31/10**9) # max int32 for overflow. 
    if not args.skip: 
        print("dense_ERI", time.time()-start, end="")
        dense_ERI = mol.intor("int2e_sph", aosym="s1")
        print(dense_ERI.shape)
    print("Minao init", time.time()-start, distinct_ERI.shape)
    dm = pyscf.scf.hf.init_guess_by_minao(mol)         
    scale = HYB_B3LYP/2
    if not args.skip: 
        print("einsum J", time.time()-start)
        print(dense_ERI.shape, dm.shape)
        J = np.einsum("ijkl,ji->kl", dense_ERI, dm)
        print("einsum K", time.time()-start)
        K = np.einsum("ijkl,jk->il", dense_ERI, dm)
        print("J-K/2", time.time()-start)
        truth = J - K / 2 * HYB_B3LYP

    print("\n----------")
    print("nonzero_indices ", time.time()-start)
    nonzero_indices      = np.nonzero(distinct_ERI)[0]#.astype(np.intu32)

    # could even shift the nonzero_indices[0] to minimize int32 [max-min]

    print("grap ERI values", time.time()-start)
    nonzero_distinct_ERI = distinct_ERI[nonzero_indices].astype(np.float32)
    if not args.skip: 
        print("dense: ", dense_ERI.shape, dense_ERI.nbytes/10**6)
    print("sparse: ", distinct_ERI.shape, distinct_ERI.nbytes/10**6)
    print("sparse+symmetric: ", nonzero_distinct_ERI.shape, nonzero_distinct_ERI.nbytes/10**6)
    print("per ipu", nonzero_distinct_ERI.nbytes/10**6/16*2) # obs: most time might be spent compiling loading ERI from H->device! 
    print("----------\n")

    # Compute number of repetitions per nonzero index and re-scale ERI. 
    print("compute eri normalization ", time.time()-start)
    rep = jax.jit(vmap_num_repetitions_fast, backend="cpu")(nonzero_indices)
    print("perform normalization ", time.time()-start)
    nonzero_distinct_ERI = nonzero_distinct_ERI / rep
    dm = dm.reshape(-1)
    diff_JK = np.zeros(dm.shape)

    print("pad remainder ", time.time()-start)
    print(nonzero_distinct_ERI.shape, nonzero_indices.shape)
    batches = int(args.batches) # perhaps make 10 batches? 
    remainder = nonzero_indices.shape[0] % (nipu*batches)

    print("PAD: ", nipu*batches-remainder)

    pad_length = nipu*batches-remainder
    if remainder != 0:
        nonzero_indices      = np.pad(nonzero_indices,      (0, pad_length))
        nonzero_distinct_ERI = np.pad(nonzero_distinct_ERI, (0, pad_length))

    print("reshape to 16 IPUs pmap", time.time()-start)
    nonzero_indices      = nonzero_indices.reshape(nipu, batches, -1)
    nonzero_distinct_ERI = nonzero_distinct_ERI.reshape(nipu, batches, -1)

    print(nonzero_indices.shape)
    print(nonzero_distinct_ERI.shape, nonzero_indices[0,:,0].shape)
    print(nonzero_indices.reshape(-1)[-1], 2**32, nonzero_indices.reshape(-1)[-1]<2**32)
    a       = np.diff(nonzero_indices, axis=2)#, prepend=nonzero_indices[0,:,0])
    offset  = nonzero_indices[0, :, 0].reshape(1, -1, 1).astype(np.int64)
    #nonzero_indices = np.concatenate((offset, a), axis=2)
    nonzero_indices = np.concatenate((np.zeros(offset.shape), a), axis=2).astype(np.int32)
    #nonzero_indices = np.concatenate( ( offset, offset+np.cumsum(a, axis=2) ), axis=2)

    # store offset as two int32. 
    def split(x):
        low = x.astype(np.int32)
        high = np.right_shift(low, 31).astype(np.int32)
        return low, high

    def recover(low, high):
        return low.astype(np.int64) + jnp.left_shift(high,31).astype(np.int64)

    low, high = split(offset)
    rec = recover(low, high)

    assert np.allclose(offset, rec)
    offset = split(offset)
    #exit()
    #nonzero_indices = np.diff(nonzero_indices, axis=-1, prepend=nonzero_indices[0,:,0]).astype(np.int32) # compression 
    #nonzero_indices = np.cumsum(nonzero_indices, axis=-1).astype(np.int64)
    #print(nonzero_indices.shape)
    print(nonzero_indices.shape)
    print(nonzero_distinct_ERI.shape)

    print("call pmap", time.time()-start)
    print("[%i]"%mol.nao_nr())
    if nipu > 1: 
        diff_JK = jax.pmap( sparse_symmetric_einsum, in_axes=(0,0,(0,0),None,None, None), static_broadcasted_argnums=(4,5), backend=backend, axis_name="p") (nonzero_distinct_ERI, nonzero_indices, offset, dm, args.backend, args.nipu) 
    else: 
        offset  = [o[0] for o in offset]
        diff_JK = jax.jit(sparse_symmetric_einsum, static_argnums=(4,5,6), backend=backend)(nonzero_distinct_ERI[0], nonzero_indices[0], offset, dm, args.backend, args.nipu, pad_length) 
        diff_JK = np.asarray(diff_JK)
        #diff_JK =  sparse_symmetric_einsum(nonzero_distinct_ERI[0], nonzero_indices[0], offset, dm, args.backend, args.nipu) 
        #diff_JK =  sparse_symmetric_einsum(nonzero_distinct_ERI[0], nonzero_indices[0], offset, dm, args.backend, args.nipu) 

    if args.skip: 
        exit()

    if nipu > 1: 
        diff_JK = np.array(diff_JK[0])

    diff_JK = diff_JK.reshape(N, N)
    print(diff_JK.reshape(-1)[::51])
    print(truth.reshape(-1)[::51])
    print(np.max(np.abs(diff_JK.reshape(-1) - truth.reshape(-1))))
    assert np.allclose(diff_JK, truth, atol=1e-6)
    print("PASSED!")
