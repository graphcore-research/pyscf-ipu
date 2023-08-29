# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import pyscf 
import numpy as np 
import jax 
import jax.numpy as jnp 
import os.path as osp
from tessellate_ipu import create_ipu_tile_primitive, ipu_cycle_count, tile_map, tile_put_sharded, tile_put_replicated
from functools import partial 
jax.config.update('jax_platform_name', "cpu")
jax.config.update('jax_enable_x64', False) 
HYB_B3LYP = 0.2

def max_val(N):
    x_candidate = (-1 + jnp.sqrt(1 + 8*N)) // 2 # TODO: Check when this overflows and fix problem (e.g. N=500 => N^4/8 =7.8G > 2**32 for np.uint32).
    x_candidate = (x_candidate).astype(jnp.int32) 
    x = jnp.where(x_candidate * (x_candidate + 1) // 2 <= N, x_candidate, x_candidate - 1)
    return x
max_vals  = jax.vmap(max_val)
def get_i_j(val):
    i = max_val(val)
    j = val - i*(i+1)//2
    return i, j

def cpu_ijkl(value, symmetry, mask, f): 
    ij = max_val(value*mask[0])
    kl = value - ij*(ij+1)//2
    i, j = get_i_j(ij)
    k, l = get_i_j(kl)
    v      = f(i,j,k,l,symmetry) 
    return v
cpu_ijkl = jax.vmap(cpu_ijkl, in_axes=(0, None, None, None))

@partial(jax.jit, backend="ipu")
def ipu_ijkl(nonzero_indices, symmetry, N, mask):
    mask = mask.astype(jnp.int32)
    vertex_filename  = osp.join(osp.dirname(__file__), "compute_indices.cpp")
    compute_indices= create_ipu_tile_primitive(
            "SymmetryIndices",
            "SymmetryIndices",
            inputs=["value", "symmetry", "input_N", "mask", "start", "stop"],
            outputs={"out": 0},
            gp_filename=vertex_filename,
            perf_estimate=100,
    )

    size = np.prod(nonzero_indices.shape)
    total_threads = (1472-1) * 6 
    remainder = size % total_threads
    if remainder != 0: 
        nonzero_indices = jnp.pad(nonzero_indices, (0, total_threads-remainder))
    nonzero_indices = nonzero_indices.reshape(total_threads, -1) 

    tiles = tuple((np.arange(0,total_threads) % (1471) + 1).astype(np.int32).tolist())
    nonzero_indices = tile_put_sharded(nonzero_indices, tiles)
    symmetry = jnp.array(symmetry, dtype=jnp.int32).reshape(1,)
    symmetry = tile_put_replicated(symmetry,   tiles) 
    mask     = tile_put_replicated(mask,   tiles)
    N        = tile_put_replicated(N,   tiles)
    start    = tile_put_replicated(0,   tiles)
    stop     = tile_put_replicated(nonzero_indices.shape[1],   tiles)

    value = tile_map(compute_indices, nonzero_indices, symmetry, N, mask, start, stop)  

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

def sparse_symmetric_einsum(nonzero_distinct_ERI, nonzero_indices, dm, mask, backend):
    dm = dm.reshape(-1)
    diff_JK = jnp.zeros(dm.shape)
    N = int(np.sqrt(dm.shape[0]))

    mask      = (mask-1)*jnp.sum(diff_JK)*jnp.sum(dm) + mask  # TODO: check if we can remove. 
    
    if backend == "cpu":
        indices_func = lambda i,j,k,l,symmetry: jnp.array([i*N+j, j*N+i, i*N+j, j*N+i, k*N+l, l*N+k, k*N+l, l*N+k,
                                                        k*N+l, k*N+l, l*N+k, l*N+k, i*N+j, i*N+j, j*N+i, j*N+i,
                                                        k*N+j, k*N+i, l*N+j, l*N+i, i*N+l, i*N+k, j*N+l, j*N+k,
                                                        i*N+l, j*N+l, i*N+k, j*N+k, k*N+j, l*N+j, k*N+i, l*N+i])[symmetry]

    def iteration(symmetry, vals): 
        diff_JK, mask= vals 
        is_K_matrix = (symmetry >= 8)

        def sequentialized_iter(i, vals):
            # Generalized J/K computation: does J when symmetry is in range(0,8) and K when symmetry is in range(8,16)
            # Trade-off: Using one function leads to smaller always-live memory.
            diff_JK, mask = vals 

            indices = nonzero_indices[i]
            eris    = nonzero_distinct_ERI[i]

            if backend == "cpu": dm_indices = cpu_ijkl(indices, symmetry+is_K_matrix*8, mask, indices_func)  
            else:                dm_indices = ipu_ijkl(indices, symmetry+is_K_matrix*8, N, mask)  
            dm_values  = jnp.take(dm, dm_indices, axis=0) # causes peak 1 
            print(eris.shape, dm_values.shape, indices.shape)
            dm_values = dm_values.at[:].mul( eris ) #* mask  # this is prod, but re-use variable for inplace update. 
            mask       = (mask-1)*jnp.sum(dm_values) + mask 
            if backend == "cpu": ss_indices = cpu_ijkl(indices, symmetry+8+is_K_matrix*8, mask, indices_func) 
            else:                ss_indices = ipu_ijkl(indices, symmetry+8+is_K_matrix*8, N, mask) 
            diff_JK    = diff_JK + jax.ops.segment_sum(dm_values, ss_indices, N**2) * (-HYB_B3LYP/2)**is_K_matrix # causes peak 2 
            mask       = (mask-1)*jnp.sum(diff_JK) + mask 
            return [diff_JK, mask]

        batches = nonzero_indices.shape[0] # before pmap, tensor had shape (nipus, batches, -1) so [0]=batches after pmap
        diff_JK, mask = jax.lax.fori_loop(0, batches, sequentialized_iter, [diff_JK, mask]) 
        return [diff_JK, mask]

    diff_JK, mask = jax.lax.fori_loop(0, 16, iteration, [diff_JK, mask]) 
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
    args = parser.parse_args()
    backend = args.backend 

    natm = int(args.natm) 
    nipu = int(args.nipu)
    if backend == "cpu": nipu = 1

    start = time.time()

    mask = jnp.ones(1, dtype=jnp.int32)
    mol = pyscf.gto.Mole(atom="".join(f"C 0 {1.54*j} {1.54*i};" for i in range(natm) for j in range(natm))) 
    mol.build()
    N = mol.nao_nr()
    print("[%i]"%mol.nao_nr())
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
    nonzero_indices      = np.nonzero(distinct_ERI)[0].astype(np.int32)
    print("grap ERI values", time.time()-start)
    nonzero_distinct_ERI = distinct_ERI[nonzero_indices].astype(np.float32)
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
        dm_indices = get_ijkl(nonzero_indices, symmetry, mask, dm_indices_func_J)  

        size = nonzero_indices.shape[0]
        total_threads = (1472-1) * 6 
        remainder = size % total_threads
        if remainder != 0: 
            nonzero_indices = np.pad(nonzero_indices, (0, total_threads-remainder))
        nonzero_indices = nonzero_indices.reshape(total_threads, -1) 

        _dm_indices = np.asarray(_get_ijkl(nonzero_indices, symmetry, N, mask))[:size]
        assert np.allclose(dm_indices, _dm_indices)
        exit()

    # Compute number of repetitions per nonzero index and re-scale ERI. 
    print("compute eri normalization ", time.time()-start)
    rep = jax.jit(vmap_num_repetitions_fast, backend="cpu")(nonzero_indices)
    print("perform normalization ", time.time()-start)
    nonzero_distinct_ERI = nonzero_distinct_ERI / rep
    mask = np.ones(1)
    dm = dm.reshape(-1)
    diff_JK = np.zeros(dm.shape)

    print("pad remainder ", time.time()-start)
    print(nonzero_distinct_ERI.shape, nonzero_indices.shape)
    batches = int(args.batches) # perhaps make 10 batches? 
    remainder = nonzero_indices.shape[0] % (nipu*batches)
    if remainder != 0:
        nonzero_indices = np.pad(nonzero_indices, (0,nipu*batches-remainder))
        nonzero_distinct_ERI = np.pad(nonzero_distinct_ERI, (0,nipu*batches-remainder))

    print("reshape to 16 IPUs pmap", time.time()-start)
    nonzero_indices = nonzero_indices.reshape(nipu, batches, -1)
    nonzero_distinct_ERI = nonzero_distinct_ERI.reshape(nipu, batches, -1)
    print(nonzero_indices.shape)
    print(nonzero_distinct_ERI.shape)

    print("call pmap", time.time()-start)
    print("[%i]"%mol.nao_nr())
    diff_JK = jax.pmap( sparse_symmetric_einsum, in_axes=(0,0,None,None,None, None), static_broadcasted_argnums=(4,), backend=backend, axis_name="p") (nonzero_distinct_ERI, nonzero_indices, dm, mask, args.backend) 
    if args.skip: 
        exit()
    diff_JK = np.array(diff_JK[0])

    diff_JK = diff_JK.reshape(N, N)
    print(diff_JK.reshape(-1)[::51])
    print(truth.reshape(-1)[::51])
    print(np.max(np.abs(diff_JK.reshape(-1) - truth.reshape(-1))))
    assert np.allclose(diff_JK, truth, atol=1e-6)
    print("PASSED!")
