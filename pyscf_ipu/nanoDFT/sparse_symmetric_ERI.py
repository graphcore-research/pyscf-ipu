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

from tessellate_ipu import create_ipu_tile_primitive, ipu_cycle_count, tile_map, tile_put_sharded, tile_put_replicated
vertex_filename  = osp.join(osp.dirname(__file__), "compute_indices.cpp")
compute_indices= [create_ipu_tile_primitive(
        "TemplatedIndicesIJKL<%i>"%a ,
        "TemplatedIndicesIJKL<%i>"%a ,
        inputs=["outshape", "nonzero_indices", "N_", "start_", "stop_"], 
        outputs={"out_": 0},
        gp_filename=vertex_filename,
        perf_estimate=100,
) for a in range(32)]

@partial(jax.jit, backend="ipu")
def ipu_ijkl(nonzero_indices, symmetry, N):
    nonzero_indices = nonzero_indices[:, 0]
    size          = nonzero_indices.shape[0]
    total_threads = (1472-1) * 6 
    remainder     = size % total_threads

    if remainder != 0: nonzero_indices = jnp.pad(nonzero_indices, ((0, total_threads-remainder), (0, 0)))

    nonzero_indices = nonzero_indices.reshape(total_threads, -1, 4)
    stop            = nonzero_indices.shape[1]
    outshape        = np.zeros((total_threads, stop), dtype=np.int32) # perhaps faster if jnp? 

    tiles = tuple((np.arange(0,total_threads) % (1471) + 1).astype(np.uint32).tolist())
    symmetry = tile_put_replicated(jnp.array(symmetry, dtype=jnp.uint32),   tiles) 
    N        = tile_put_replicated(jnp.array(N,        dtype=jnp.uint32),   tiles)
    start    = tile_put_replicated(jnp.array(0,        dtype=jnp.uint32),   tiles)
    stop     = tile_put_replicated(jnp.array(stop,     dtype=jnp.uint32),   tiles)

    outshape        = tile_put_sharded(outshape, tiles)
    nonzero_indices = tile_put_sharded(nonzero_indices, tiles)

    value = tile_map(compute_indices, outshape, nonzero_indices, symmetry, N, start, stop)

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
    dm      = dm.reshape(-1)
    diff_JK = jnp.zeros(dm.shape)

    N       = int(np.sqrt(dm.shape[0]))

    # Arguments for gather/scatter_add. 
    dnums         = jax.lax.GatherDimensionNumbers( offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,))
    scatter_dnums = jax.lax.ScatterDimensionNumbers( update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,))

    # Initialize memory layout.
    SKIP_TILES = 1 
    NUM_TILES  = 1471
    num        = nonzero_distinct_ERI.shape[1] 
    tiles      = tuple((np.arange(0,num) % (NUM_TILES) + SKIP_TILES).astype(np.uint32).tolist())
    eris       = tile_put_sharded(nonzero_distinct_ERI.T, tiles=tiles)

    nonzero_indices = nonzero_indices.reshape(batches, num, 1, 4)
    nonzero_indices = tile_put_sharded(jnp.transpose(nonzero_indices, (1, 0, 2, 3)), tiles=tiles)

    stop       = nonzero_indices.shape[2]
    outshape   = np.zeros((num, stop, 1), dtype=np.uint32) # perhaps faster if jnp? 

    multiplier = -HYB_B3LYP/2
    mult       = jnp.ones(N**2)*multiplier
    mult       = tile_put_replicated(mult, tiles=tiles)

    # dm is symmetric, only take the distinct entris 
    #dm = dm[:N*(N-1)//2]  
    #dm = dm[:N*N)//2]  

    dm_scaled  = tile_put_replicated(dm*multiplier, tiles)
    #dm_scaled  = dm*multiplier
    #dm         = tile_put_replicated(dm, tiles)
    print(dm.shape)
    dm = dm.reshape(2, 450)
    diff_JK    = tile_put_replicated(diff_JK, tiles=tiles)

    N         = tile_put_replicated(jnp.array(N,        dtype=jnp.uint32),   tiles)
    start     = tile_put_replicated(jnp.array(0,        dtype=jnp.uint32),   tiles)
    stop      = tile_put_replicated(jnp.array(stop,     dtype=jnp.uint32),   tiles)
    outshape  = tile_put_sharded(outshape, tiles)

    def iteration(i, diff_JK): 
        def sequentialized_iter(symmetry, diff_JK, dm, dm_scaled, is_K_matrix=False):
            # Step 1: dm_indices = ipu_ijkl(indices, symmetry+is_K_matrix*8, N).reshape(-1, 1)
            nonzero_indices_i = nonzero_indices[:, i] 
            dm_indices        = tile_map(compute_indices[symmetry + is_K_matrix*8], outshape, nonzero_indices_i, N, start, stop)
            dm_indices = tile_put_sharded(dm_indices.array-first_half*step, tiles)

            # Step 2: dm_values = jnp.take(dm, indices, axis=0) 
            dm_values = tile_map( jax.lax.gather_p,
                                    dm_scaled if is_K_matrix else dm, # faster with pre-scaled dm 
                                    dm_indices,
                                    dimension_numbers=dnums, slice_sizes=(1,), mode=jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS,
                                    unique_indices=False, indices_are_sorted=False, fill_value=None,
            )

            # Step 3: prod 
            dm_values = tile_map(jax.lax.mul_p, dm_values, eris[:, i])
            
            # Step 4: ss_indices = ipu_ijkl(nonzero_indices[i], symmetry+8+is_K_matrix*8, N).reshape(-1,1)
            ss_indices = tile_map(compute_indices[symmetry+8+is_K_matrix*8], outshape, nonzero_indices_i, N, start, stop)
            
            # Step 5: diff_JK = jax.lax.segment_sum(diff_JK, ss_indices, dm_values, ...)
            return tile_map(
                jax.lax.scatter_add_p,
                diff_JK,
                ss_indices,
                dm_values,
                dimension_numbers=scatter_dnums,
                indices_are_sorted=False,
                unique_indices=False,
                mode=jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS,
                update_jaxpr=None,
                update_consts=None,
            )

        #diff_JK = jax.lax.fori_loop(0, batches, sequentialized_iter, diff_JK) 
        sequentialized_iter_J = lambda x,y,z,a: sequentialized_iter(x,y,z,a,False) 
        sequentialized_iter_K = lambda x,y,z,a: sequentialized_iter(x,y,z,a,True) 
        #diff_JK = jax.lax.fori_loop(0, 8, sequentialized_iter_J, diff_JK) 
        #diff_JK = jax.lax.fori_loop(8, 16, sequentialized_iter_K, diff_JK) 

        #tile_shard_dm 
        # // may add a 10x here. 

        first_half = i < (batches//2)
        step = dm.shape[0]//2
        print(first_half, step)

        _dm_scaled = tile_put_replicated(dm[first_half]*multiplier, tiles)
        _dm        = tile_put_replicated(dm[first_half], tiles)
        #diff_JK   = tile_put_replicated(diff_JK, tiles=tiles)

        # this basically uses the same floats 16 times; 
        # we don't want to reload stuff for that, i.e., have them close to each other. 
        for z in range(8):
            diff_JK = sequentialized_iter_J(z, diff_JK, _dm, _dm_scaled)
        for z in range(8, 16):
            diff_JK = sequentialized_iter_K(z, diff_JK, _dm, _dm_scaled) 

        #dm_scaled = jnp.sum(dm_scaled.array, axis=0)
        #dm = dm.array

        #for i in range(batches):
        #    diff_JK = sequentialized_iter(i, diff_JK)
        #tile-reduce_sum here # |batch_size| more reduce_sums.  only of size N^2/|batch_size. 
        #  => only adds N^2 reduce_sum. 

        return diff_JK

    #diff_JK = jax.lax.fori_loop(0, 16, iteration, diff_JK) 
    #for i in range(16):
    #    diff_JK = iteration(i, diff_JK)
    diff_JK = jax.lax.fori_loop(0, batches, iteration, diff_JK) 
    #for i in range(batches):
    #    diff_JK = iteration(i, diff_JK)
    #return jax.lax.psum(diff_JK, axis_name="p")
    print(diff_JK.shape)
    diff_JK = diff_JK.array.sum(0)
    print(diff_JK.shape)
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

    #mol = pyscf.gto.Mole(atom="".join(f"C 0 {1.54*j} {1.54*i};" for i in range(natm) for j in range(natm))) 
    mol = pyscf.gto.Mole(atom="".join(f"C 0 {1.54*j} {1.54*i};" for i in range(2) for j in range(3))) 
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
    print(ij)
    # does transpose just correspond to exchange ij, kl? 
    # i want to exchange the order we visit the elements in, i.e., visit column by column. 

    # comput batch_size so we get as close as possible to  NUM_TILES*THREADS 
    rep                  = num_repetitions_fast(ij, kl)
    nonzero_distinct_ERI = nonzero_distinct_ERI / rep
    dm                   = dm.reshape(-1)
    diff_JK              = np.zeros(dm.shape)
    batches  = int(args.batches) # perhaps make 10 batches? 


    NUM_TILES = 1471 
    batches = int(nonzero_indices.shape[0]//(NUM_TILES*6))+1

    remainder = nonzero_indices.shape[0] % (nipu*batches)

    if remainder != 0:
        print(nipu*batches-remainder, ij.shape)
        ij = np.pad(ij, ((0,nipu*batches-remainder)))
        kl = np.pad(kl, ((0,nipu*batches-remainder)))
        nonzero_distinct_ERI = np.pad(nonzero_distinct_ERI, (0,nipu*batches-remainder))

    # Have column on single tile 
    # split so each tile gets a column, i.e., only needs one element from dm. 
    
    # can I arbirarily permute at this point? 

    print(ij.shape, kl.shape, nonzero_distinct_ERI.shape)
    #indxs = np.argsort(np.random.normal(0, 1, ij.shape[0]))
    indxs = np.argsort(kl)
    ij, kl, nonzero_distinct_ERI = ij[indxs], kl[indxs], nonzero_distinct_ERI[indxs]

    # the batching isn't perfectly uniform. 
    # for some reason, the matrix has more elemnts in earlier columns than later. 
    # perhaps this is due to all of the symmetry? 
    # this means basically we will touch th first entries of DM much more than the later entries of dm .
    # oh, but if we also utilize the symmetry in dm matrix I think this may even out! 
    # so might be perfectly balanced! 
    print(kl)

    import matplotlib.pyplot as plt 
    fig, ax = plt.subplots(1,2)
    ax[0].plot(kl)
    # we are indexing into entry (ij, kl); 
    # i want to sort so we do 

    # so dm is symmetric => only need to store half of it! 
    # dm = dm.reshape(-1)[:N*(N-1)//2]

    ij = ij.reshape(nipu, batches, -1)
    kl = kl.reshape(nipu, batches, -1)
    print(batches)
    for i in range(batches):
        print(kl[0,i].min(), kl[0, i].max())
        ax[1].plot(kl[0,i])
    plt.savefig("cols.jpg")


    nonzero_distinct_ERI = nonzero_distinct_ERI.reshape(nipu, batches, -1)

    i, j = get_i_j(ij.reshape(-1))
    k, l  = get_i_j(kl.reshape(-1))
    nonzero_indices = np.vstack([i,j,k,l]).T.reshape(nipu, batches, -1, 4).astype(np.int32)

    np.random.seed(42)
    # we can permute arbitraily! 
    #indxs = np.argsort(np.random.normal(0, 1, nonzero_indices.shape[2]))
    #indxs = np.argsort(np.abs(nonzero_distinct_ERI[0,0]))
    #indxs = np.argsort(np.abs(nonzero_indices[0,0]))
    indxs = np.arange(nonzero_indices.shape[2])

    print(nonzero_indices[0,0, :5])
    nonzero_indices = nonzero_indices[:, :, indxs]
    print(nonzero_indices[0,0, :5])
    print("--> %i : %i <--"%(batches, nonzero_indices.shape[2]))

    #nonzero_indices = jax.lax.bitcast_convert_type(nonzero_indices, np.float16)
    print(nonzero_distinct_ERI[0,0,:5])
    nonzero_distinct_ERI = nonzero_distinct_ERI[:, :, indxs] 
    print(nonzero_distinct_ERI[0,0,:5])

    diff_JK = jax.pmap(sparse_symmetric_einsum, in_axes=(0,0,None,None), static_broadcasted_argnums=(3,), backend=backend, axis_name="p")(nonzero_distinct_ERI, nonzero_indices, dm, args.backend) 
    #diff_JK = jax.pmap(sparse_symmetric_einsum, in_axes=(0,0,None,None), static_broadcasted_argnums=(3,), backend=backend, axis_name="p")(nonzero_distinct_ERI, nonzero_indices, dm, args.backend) 
    #diff_JK = jax.jit(sparse_symmetric_einsum(nonzero_distinct_ERI[0], nonzero_indices[0], dm, args.backend) 

    if args.skip: 
        exit()
    if args.nipu > 1:
        diff_JK = np.array(diff_JK[0])

    diff_JK = diff_JK.reshape(N, N)
    print(diff_JK.reshape(-1)[::51])
    print(truth.reshape(-1)[::51])  
    print(diff_JK.shape, truth.shape)
    print(np.max(np.abs(diff_JK.reshape(-1) - truth.reshape(-1))))

    # deal with diagonal entries N^2 using double precision? 
    import matplotlib.pyplot as plt 

    fig, ax= plt.subplots(1,1)
    diff = np.abs(diff_JK.reshape(-1)-truth.reshape(-1))
    real_diff = diff
    indxs = np.argsort(diff)
    diff = np.abs(diff_JK.reshape(-1)-truth.reshape(-1))/np.abs(truth.reshape(-1))
    diff[np.abs(truth.reshape(-1)) < 1e-9] = 0 
    #ax.plot(diff[indxs], 'x', ms=2, label="relative error")
    ax.plot(np.abs(truth.reshape(-1))[indxs], 'o', ms=2, label="real abs value")
    ax.plot(real_diff[indxs], label="absolute error")
    plt.legend()

    ax.set_yscale("log")

    plt.savefig("error_jk.jpg")
    

    assert np.allclose(diff_JK, truth, atol=1e-6) # maybe make image of abs/relative error? do on image of matrix aswell? 
    print("PASSED!")