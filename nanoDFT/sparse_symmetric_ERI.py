# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import pyscf 
import numpy as np 
import jax 
import jax.numpy as jnp 
import os.path as osp
from tessellate_ipu import create_ipu_tile_primitive, ipu_cycle_count, tile_map, tile_put_sharded, tile_put_replicated
from functools import partial 
from icecream import ic
from jax.experimental import checkify
jax.config.update('jax_platform_name', "cpu")
jax.config.update('jax_enable_x64', False) 
HYB_B3LYP = 0.2

def get_i_j(val, precision='x32'):
    '''
    This ij->i,j implementation relies on sqrt in fp32 which limits the range of values
    we are able to handle to val < N^2 = 3258^2 (about 2^21 from the total of 2^32 in an uint32)
    '''
    
    # we need it to work for val.dtype == uint64 when called with the main index
    xnp = jnp if precision=='x32' else np # for compatibility
    sqrt_precision = xnp.float32 if precision=='x32' else xnp.float64
    
    i = xnp.floor((xnp.sqrt(1 + 8*val.astype(sqrt_precision)) - 1)/2).astype(xnp.uint32)
    j = (((val - i) - (i**2 - val))//2).astype(np.uint32) # val - i*(i+1)/2

    return i, j

def cpu_ijkl(value, symmetry, f): 
    indexdim = 'x64' if value.shape == () else 'x32'
    if indexdim == 'x32':
        # x32
        i, j = get_i_j(value[0])
        k, l = get_i_j(value[1])
    else:
        # x64
        ij, kl = get_i_j(value)
        i, j = get_i_j(ij)
        k, l = get_i_j(kl)

    return f(i,j,k,l,symmetry)
cpu_ijkl = jax.vmap(cpu_ijkl, in_axes=(0, None, None))

@partial(jax.jit, backend="ipu")
def ipu_ijkl(nonzero_indices, symmetry, N):
    indexdim = 'x32' if nonzero_indices.shape[-1] == 2 else 'x64'
    vertex_filename  = osp.join(osp.dirname(__file__), "compute_indices.cpp")
    compute_indices= create_ipu_tile_primitive(
            "SymmetryIndices2D" if indexdim == 'x32' else "SymmetryIndices1D",
            "SymmetryIndices2D" if indexdim == 'x32' else "SymmetryIndices1D",
            inputs=["value_ij", "value_kl", "symmetry", "input_N", "start", "stop"] if indexdim == 'x32' else ["value", "symmetry", "input_N", "start", "stop"],
            outputs={"out": 0},
            gp_filename=vertex_filename,
            perf_estimate=100,
    )

    size = nonzero_indices.shape[0]
    total_threads = (1472-1) * 6 
    remainder = size % total_threads
    
    if indexdim == 'x32':
        # x32
        if remainder != 0: 
            nonzero_indices = jnp.pad(nonzero_indices, ((0, total_threads-remainder), (0, 0)))
        nonzero_indices = nonzero_indices.reshape(total_threads, -1, 2)
    else:
        # x64
        if remainder != 0: 
            nonzero_indices = jnp.pad(nonzero_indices, (0, total_threads-remainder))
        nonzero_indices = nonzero_indices.reshape(total_threads, -1) 
    
    stop = nonzero_indices.shape[1]

    tiles = tuple((np.arange(0,total_threads) % (1471) + 1).astype(np.uint32).tolist())
    symmetry = tile_put_replicated(jnp.array(symmetry, dtype=jnp.uint32),   tiles) 
    N        = tile_put_replicated(jnp.array(N, dtype=jnp.uint32),   tiles)
    start    = tile_put_replicated(jnp.array(0, dtype=jnp.uint32),   tiles)
    stop     = tile_put_replicated(jnp.array(stop, dtype=jnp.uint32),   tiles)
    if indexdim == 'x32':
        nonzero_indices_ij = tile_put_sharded(nonzero_indices[:, :, 0], tiles)
        nonzero_indices_kl = tile_put_sharded(nonzero_indices[:, :, 1], tiles)
        value = tile_map(compute_indices, nonzero_indices_ij, nonzero_indices_kl, symmetry, N, start, stop)
    else:
        nonzero_indices = tile_put_sharded(nonzero_indices, tiles)
        value = tile_map(compute_indices, nonzero_indices, symmetry, N, start, stop)

    return value.array.reshape(-1)[:size]

def num_repetitions_fast(value):
    indexdim = 'x64' if value.shape == () else 'x32'
    if indexdim == 'x32':
        # x32
        i, j = get_i_j(value[0])
        k, l = get_i_j(value[1])
    else:
        # x64
        ij, kl = get_i_j(value)
        i, j = get_i_j(ij)
        k, l = get_i_j(kl)

    # compute: repetitions = 2^((i==j) + (k==l) + (k==i and l==j or k==j and l==i))
    repetitions = 2**(
        jnp.equal(i,j).astype(jnp.uint32) + 
        jnp.equal(k,l).astype(jnp.uint32) + 
        (1 - ((1 - jnp.equal(k,i) * jnp.equal(l,j)) * 
        (1- jnp.equal(k,j) * jnp.equal(l,i))).astype(jnp.uint32))
    )

    return repetitions
vmap_num_repetitions_fast = jax.vmap(num_repetitions_fast, in_axes=(0))

def sparse_symmetric_einsum(nonzero_distinct_ERI, nonzero_indices, dm, backend):
    dm = dm.reshape(-1)
    diff_JK = jnp.zeros(dm.shape)
    N = int(np.sqrt(dm.shape[0]))
    
    if backend == "cpu":
        indices_func = lambda i,j,k,l,symmetry: jnp.array([i*N+j, j*N+i, i*N+j, j*N+i, k*N+l, l*N+k, k*N+l, l*N+k,
                                                        k*N+l, k*N+l, l*N+k, l*N+k, i*N+j, i*N+j, j*N+i, j*N+i,
                                                        k*N+j, k*N+i, l*N+j, l*N+i, i*N+l, i*N+k, j*N+l, j*N+k,
                                                        i*N+l, j*N+l, i*N+k, j*N+k, k*N+j, l*N+j, k*N+i, l*N+i])[symmetry]

    def iteration(symmetry, vals): 
        diff_JK = vals 
        is_K_matrix = (symmetry >= 8)

        def sequentialized_iter(i, vals):
            # Generalized J/K computation: does J when symmetry is in range(0,8) and K when symmetry is in range(8,16)
            # Trade-off: Using one function leads to smaller always-live memory.
            diff_JK = vals 

            indices = nonzero_indices[i]
            eris    = nonzero_distinct_ERI[i]

            if backend == "cpu": dm_indices = cpu_ijkl(indices, symmetry+is_K_matrix*8, indices_func)  
            else:                dm_indices = ipu_ijkl(indices, symmetry+is_K_matrix*8, N)  
            dm_values  = jnp.take(dm, dm_indices, axis=0) # causes peak 1 
            print(eris.shape, dm_values.shape, indices.shape)
            dm_values = dm_values.at[:].mul( eris ) # this is prod, but re-use variable for inplace update. 
            
            if backend == "cpu": ss_indices = cpu_ijkl(indices, symmetry+8+is_K_matrix*8, indices_func) 
            else:                ss_indices = ipu_ijkl(indices, symmetry+8+is_K_matrix*8, N) 
            diff_JK    = diff_JK + jax.ops.segment_sum(dm_values, ss_indices, N**2) * (-HYB_B3LYP/2)**is_K_matrix # causes peak 2 
            
            return diff_JK

        batches = nonzero_indices.shape[0] # before pmap, tensor had shape (nipus, batches, -1) so [0]=batches after pmap
        diff_JK = jax.lax.fori_loop(0, batches, sequentialized_iter, diff_JK) 
        return diff_JK

    diff_JK = jax.lax.fori_loop(0, 16, iteration, diff_JK) 
    return jax.lax.psum(diff_JK, axis_name="p")

if __name__ == "__main__":
    import time 
    import argparse 
    parser = argparse.ArgumentParser(prog='', description='', epilog='')
    parser.add_argument('-backend', default="cpu"),
    parser.add_argument('-natm', default=3),
    parser.add_argument('-test', action="store_true")
    parser.add_argument('-prof', action="store_true")
    parser.add_argument('-batches', default=5)
    parser.add_argument('-nipu', default=16)
    parser.add_argument('-skip', action="store_true") 
    parser.add_argument('-indexdim', default='x32', choices=['x64', 'x32']) 
    parser.add_argument('-indexcheck', action="store_true") 
    
    args = parser.parse_args()
    backend = args.backend 

    natm = int(args.natm) 
    nipu = int(args.nipu)
    if backend == "cpu": nipu = 1

    start = time.time()

    mol = pyscf.gto.Mole(atom="".join(f"C 0 {1.54*j} {1.54*i};" for i in range(natm) for j in range(natm))) 
    mol.build()
    N = mol.nao_nr()
    print("N %i"%mol.nao_nr())
    if not args.skip: 
        print("dense_ERI", time.time()-start)
        dense_ERI = mol.intor("int2e_sph", aosym="s1")
    print("distinct_ERI", time.time()-start)
    distinct_ERI = mol.intor("int2e_sph", aosym="s8")
    print("Minao init", time.time()-start)
    dm = pyscf.scf.hf.init_guess_by_minao(mol)         
    scale = HYB_B3LYP/2
    if not args.skip: 
        print("einsum J", time.time()-start)
        J = np.einsum("ijkl,ji->kl", dense_ERI, dm)
        print("einsum K", time.time()-start)
        K = np.einsum("ijkl,jk->il", dense_ERI, dm)
        print("J-K/2", time.time()-start)
        truth = J - K / 2 * HYB_B3LYP

    print("\n----------")
    print("nonzero_indices ", time.time()-start)
    nonzero_indices      = np.nonzero(distinct_ERI)[0].astype(np.uint64) # int64 by default, convert to uint64
    print("grap ERI values", time.time()-start)
    nonzero_distinct_ERI = distinct_ERI[nonzero_indices].astype(np.float32)
    nonzero_indices_x64 = nonzero_indices
    nonzero_indices_2d = np.stack(get_i_j(nonzero_indices, precision='x64'), axis=1) # ijkl x64 -> ij,kl x32 -- shape=(?, 2)
    assert np.equal(nonzero_indices_2d, nonzero_indices_2d.astype(np.uint32)).all()
    nonzero_indices_2d = nonzero_indices_2d.astype(np.uint32)
    nonzero_indices = nonzero_indices_2d
    print(nonzero_indices.shape)
    # ------------------------------ #
    if args.indexcheck:
        a = nonzero_indices_x64
        b = (nonzero_indices[:,0].astype(np.uint64)*(nonzero_indices[:,0].astype(np.uint64)+1)/2+nonzero_indices[:,1].astype(np.uint64)).astype(np.uint64)
        assert np.equal(a, b).all(), (a, b, np.nonzero(np.abs(a-b)))
        print('Check: 1D (ijkl) to 2D (ij,kl) decomposition is accurate')
        nonzero_indices_4d = np.concatenate(get_i_j(nonzero_indices_2d), axis=1) # ij,kl x32 -> i,j,k,l x32 -- shape=(?, 2)
        nonzero_indices_4d = nonzero_indices_4d.astype(np.uint32)
        a = nonzero_indices_x64
        b_ij = (nonzero_indices_4d[:,0].astype(np.uint64)*(nonzero_indices_4d[:,0].astype(np.uint64)+1)/2+nonzero_indices_4d[:,2].astype(np.uint64)).astype(np.uint64)
        b_kl = (nonzero_indices_4d[:,1].astype(np.uint64)*(nonzero_indices_4d[:,1].astype(np.uint64)+1)/2+nonzero_indices_4d[:,3].astype(np.uint64)).astype(np.uint64)
        b = (b_ij*(b_ij+1)/2+b_kl).astype(np.uint64)
        assert np.equal(a, b).all(), (a, b, np.nonzero(np.abs(a-b)), np.nonzero(np.abs(a-b))[0].shape)
        print('Check: 2D (ij,kl) to 4D (i,j,k,l) decomposition is accurate')
    # ------------------------------ #
    # orig
    if args.indexdim == 'x64':
        nonzero_indices = nonzero_indices_x64
    # ------------------------------ #
    if not args.skip: 
        print("dense: ", dense_ERI.shape, dense_ERI.nbytes/10**6)
    print("sparse: ", distinct_ERI.shape, distinct_ERI.nbytes/10**6)
    print("sparse+symmetric: ", nonzero_distinct_ERI.shape, nonzero_distinct_ERI.nbytes/10**6)
    print("per ipu", nonzero_distinct_ERI.nbytes/10**6/16*2) # obs: most time might be spent compiling loading ERI from H->device! 
    print("----------\n")

    # write code that profiles get_indices/take/segment_sum independently. 
    # TODO: fix broken test below. add seperate test for get_indices/np.take and segment_sum, potentially with custom poplar implementations
    if args.prof: 
        symmetry = 7 
        dm_indices = get_ijkl(nonzero_indices, symmetry, dm_indices_func_J)  

        size = nonzero_indices.shape[0]
        total_threads = (1472-1) * 6 
        remainder = size % total_threads
        if remainder != 0: 
            nonzero_indices = np.pad(nonzero_indices, (0, total_threads-remainder))
        nonzero_indices = nonzero_indices.reshape(total_threads, -1) 

        _dm_indices = np.asarray(_get_ijkl(nonzero_indices, symmetry, N))[:size]
        assert np.allclose(dm_indices, _dm_indices)
        exit()

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
    if len(nonzero_indices.shape) == 1:
        # ijkl x64 format
        if remainder != 0:
            nonzero_indices = np.pad(nonzero_indices, (0,nipu*batches-remainder))
            nonzero_distinct_ERI = np.pad(nonzero_distinct_ERI, (0,nipu*batches-remainder))
        print("reshape to 16 IPUs pmap", time.time()-start)
        nonzero_indices = nonzero_indices.reshape(nipu, batches, -1)
        nonzero_distinct_ERI = nonzero_distinct_ERI.reshape(nipu, batches, -1)
    else: # assuming == 2
        # ij,kl x32 format
        if remainder != 0:
            nonzero_indices = np.pad(nonzero_indices, ((0,nipu*batches-remainder), (0,0)))
            nonzero_distinct_ERI = np.pad(nonzero_distinct_ERI, (0,nipu*batches-remainder))
        print("reshape to 16 IPUs pmap", time.time()-start)
        nonzero_indices = nonzero_indices.reshape(nipu, batches, -1, 2)
        nonzero_distinct_ERI = nonzero_distinct_ERI.reshape(nipu, batches, -1)
    
    print(nonzero_indices.shape)
    print(nonzero_distinct_ERI.shape)

    print("call pmap", time.time()-start)
    print("[%i]"%mol.nao_nr())
    diff_JK = jax.pmap( sparse_symmetric_einsum, in_axes=(0,0,None,None, None), static_broadcasted_argnums=(3,), backend=backend, axis_name="p") (nonzero_distinct_ERI, nonzero_indices, dm, args.backend) 
    if args.skip: 
        exit()
    diff_JK = np.array(diff_JK[0])

    diff_JK = diff_JK.reshape(N, N)
    print(diff_JK.reshape(-1)[::51])
    print(truth.reshape(-1)[::51])
    print(np.max(np.abs(diff_JK.reshape(-1) - truth.reshape(-1))))
    assert np.allclose(diff_JK, truth, atol=1e-6)
    print("PASSED!")
