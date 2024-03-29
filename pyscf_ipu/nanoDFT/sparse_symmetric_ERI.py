# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import pyscf 
import numpy as np 
import jax 
import jax.numpy as jnp 
import os.path as osp
from functools import partial 
from icecream import ic
jax.config.update('jax_platform_name', "cpu")
#jax.config.update('jax_enable_x64', True) 
HYB_B3LYP = 0.2

def get_i_j(val):
    i = (np.sqrt(1 + 8*val.astype(np.uint64)) - 1)//2 # no need for floor, integer division acts as floor. 
    j = (((val - i) - (i**2 - val))//2)
    return i, j

def cpu_ijkl(value, symmetry, N, f):
    i, j, k, l = value[0].astype(np.uint32), value[1].astype(np.uint32), value[2].astype(np.uint32), value[3].astype(np.uint32)
    return f(i,j,k,l,symmetry,N)
cpu_ijkl = jax.vmap(cpu_ijkl, in_axes=(0, None, None, None))

@partial(jax.jit, backend="ipu")
def ipu_ijkl(nonzero_indices, symmetry, N):
    from tessellate_ipu import create_ipu_tile_primitive, ipu_cycle_count, tile_map, tile_put_sharded, tile_put_replicated
    vertex_filename  = osp.join(osp.dirname(__file__), "compute_indices.cpp")
    compute_indices= create_ipu_tile_primitive(
            "IndicesIJKL" ,
            "IndicesIJKL" ,
            inputs=["i_", "j_", "k_", "l_", "sym_", "N_", "start_", "stop_"], 
            outputs={"out_": 0},
            gp_filename=vertex_filename,
            perf_estimate=100,
    )
    size = nonzero_indices.shape[0]
    total_threads = (1472-1) * 6 
    remainder = size % total_threads

    i, j, k, l = [nonzero_indices[:, i].astype(np.uint32) for i in range(4)] 
    
    if remainder != 0: 
        i = jnp.pad(i, ((0, total_threads-remainder)))
        j = jnp.pad(j, ((0, total_threads-remainder)))
        k = jnp.pad(k, ((0, total_threads-remainder)))
        l = jnp.pad(l, ((0, total_threads-remainder)))

    i = i.reshape(total_threads, -1)
    j = j.reshape(total_threads, -1)
    k = k.reshape(total_threads, -1)
    l = l.reshape(total_threads, -1)
    
    stop = i.shape[1]

    tiles = tuple((np.arange(0,total_threads) % (1471) + 1).astype(np.uint32).tolist())
    symmetry = tile_put_replicated(jnp.array(symmetry, dtype=jnp.uint32),   tiles) 
    N        = tile_put_replicated(jnp.array(N, dtype=jnp.uint32),   tiles)
    start    = tile_put_replicated(jnp.array(0, dtype=jnp.uint32),   tiles)
    stop     = tile_put_replicated(jnp.array(stop, dtype=jnp.uint32),   tiles)

    i = tile_put_sharded(i, tiles)
    j = tile_put_sharded(j, tiles)
    k = tile_put_sharded(k, tiles)
    l = tile_put_sharded(l, tiles)
    value = tile_map(compute_indices, i, j, k, l, symmetry, N, start, stop)

    return value.array.reshape(-1)[:size]

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

def sparse_symmetric_einsum(nonzero_distinct_ERI, nonzero_indices, dm, backend):
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

    def iteration(symmetry, vals): 
        diff_JK = vals 
        is_K_matrix = (symmetry >= 8)

        def sequentialized_iter(i, vals):
            # Generalized J/K computation: does J when symmetry is in range(0,8) and K when symmetry is in range(8,16)
            # Trade-off: Using one function leads to smaller always-live memory.
            diff_JK = vals 

            indices = nonzero_indices[i]
            print(nonzero_indices.shape, indices.shape)

            indices = jax.lax.bitcast_convert_type(indices, np.int16).astype(np.int32)
            eris    = nonzero_distinct_ERI[i]

            if backend == "cpu": dm_indices = cpu_ijkl(indices, symmetry+is_K_matrix*8, N, indices_func).reshape(-1, 1)
            else:                dm_indices = ipu_ijkl(indices, symmetry+is_K_matrix*8, N)  .reshape(-1, 1)
            # dm_values = jnp.take(dm, indices, axis=0) # for our special case the 50 lines of code reduces to the one line below. 
            dm_values = jax.lax.gather(dm, dm_indices, dimension_numbers=dnums, slice_sizes=(1,), mode=jax.lax.GatherScatterMode.FILL_OR_DROP)

            dm_values = dm_values.at[:].mul( eris ) # this is prod, but re-use variable for inplace update. 
            
            if backend == "cpu": ss_indices = cpu_ijkl(indices, symmetry+8+is_K_matrix*8, N, indices_func) .reshape(-1,1)
            else:                ss_indices = ipu_ijkl(indices, symmetry+8+is_K_matrix*8, N).astype(np.int32).reshape(-1,1)
            # diff_JK = diff_JK + jax.lax.segment_sum( ...) # for our special case the 100 lines of code reduces to the one line below. 
            diff_JK = diff_JK + jax.lax.scatter_add(jnp.zeros((N**2,)),
                                            ss_indices, dm_values, 
                                            scatter_dnums, indices_are_sorted=True, unique_indices=True, mode=jax.lax.GatherScatterMode.FILL_OR_DROP)\
                                *(-HYB_B3LYP/2)**is_K_matrix
            
            return diff_JK

        batches = nonzero_indices.shape[0] # Before pmap, tensor had shape (nipus, batches, -1) so [0]=batches after pmap
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
    parser.add_argument('-nipu', default=16, type=int)
    parser.add_argument('-skip', action="store_true") 
    
    args = parser.parse_args()
    backend = args.backend 

    natm = int(args.natm) 
    nipu = int(args.nipu)
    if backend == "cpu": nipu = 1

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
    distinct_ERI[np.abs(distinct_ERI)<1e-9] = 0  # zero out stuff 
    dm = pyscf.scf.hf.init_guess_by_minao(mol)         
    scale = HYB_B3LYP/2
    if not args.skip: 
        J = np.einsum("ijkl,ji->kl", dense_ERI, dm)
        K = np.einsum("ijkl,jk->il", dense_ERI, dm)
        truth = J - K / 2 * HYB_B3LYP

    nonzero_indices      = np.nonzero(distinct_ERI)[0].astype(np.uint64) 
    nonzero_distinct_ERI = distinct_ERI[nonzero_indices].astype(np.float32)
    print("Nonzero Operations:", nonzero_indices.size*8*2/10**9, "[Giga]")
    ij, kl               = get_i_j(nonzero_indices)
    rep                  = num_repetitions_fast(ij, kl)
    nonzero_distinct_ERI = nonzero_distinct_ERI / rep
    dm                   = dm.reshape(-1)
    diff_JK              = np.zeros(dm.shape)

    batches  = int(args.batches) # perhaps make 10 batches? 
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
    nonzero_indices = np.vstack([i,j,k,l]).T.reshape(nipu, batches, -1, 4).astype(np.int16)
    nonzero_indices = jax.lax.bitcast_convert_type(nonzero_indices, np.float16)

    diff_JK = jax.pmap(sparse_symmetric_einsum, in_axes=(0,0,None,None), static_broadcasted_argnums=(3,), backend=backend, axis_name="p")(nonzero_distinct_ERI, nonzero_indices, dm, args.backend) 

    if args.skip: 
        exit()
    if args.nipu > 1:
        diff_JK = np.array(diff_JK[0])

    diff_JK = diff_JK.reshape(N, N)
    print(diff_JK.reshape(-1)[::51])
    print(truth.reshape(-1)[::51])
    print(np.max(np.abs(diff_JK.reshape(-1) - truth.reshape(-1))))
    assert np.allclose(diff_JK, truth, atol=1e-6)
    print("PASSED!")