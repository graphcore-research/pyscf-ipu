# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import pyscf 
import numpy as np 
import jax 
import jax.numpy as jnp 
import os.path as osp
from tessellate_ipu import create_ipu_tile_primitive, ipu_cycle_count, tile_map, tile_put_sharded, tile_put_replicated
from functools import partial 
from icecream import ic
from compute_eri_utils import prepare_integrals_2_inputs
jax.config.update('jax_platform_name', "cpu")
#jax.config.update('jax_enable_x64', True) 
HYB_B3LYP = 0.2

def get_i_j(val, xnp=np, dtype=np.uint64):
    i = (xnp.sqrt(1 + 8*val.astype(dtype)) - 1)//2 # no need for floor, integer division acts as floor. 
    j = (((val - i) - (i**2 - val))//2)
    return i, j

def cpu_ijkl(value, symmetry, f): 
    i, j, k, l = value[0].astype(np.uint32), value[1].astype(np.uint32), value[2].astype(np.uint32), value[3].astype(np.uint32)
    return f(i,j,k,l,symmetry)
cpu_ijkl = jax.vmap(cpu_ijkl, in_axes=(0, None, None))

@partial(jax.jit, backend="ipu")
def ipu_ijkl(nonzero_indices, symmetry, N):
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

def num_repetitions_fast_4d(i, j, k, l, xnp=np, dtype=np.uint64):
    # compute: repetitions = 2^((i==j) + (k==l) + (k==i and l==j or k==j and l==i))
    repetitions = 2**(
        xnp.equal(i,j).astype(dtype) + 
        xnp.equal(k,l).astype(dtype) + 
        (1 - ((1 - xnp.equal(k,i) * xnp.equal(l,j)) * 
        (1- xnp.equal(k,j) * xnp.equal(l,i))).astype(dtype))
    )
    return repetitions

def num_repetitions_fast(ij, kl):
    i, j = get_i_j(ij)
    k, l = get_i_j(kl)

    return num_repetitions_fast_4d(i, j, k, l)


indices_func = lambda i,j,k,l,symmetry: jnp.array([i*N+j, j*N+i, i*N+j, j*N+i, k*N+l, l*N+k, k*N+l, l*N+k,
                                                k*N+l, k*N+l, l*N+k, l*N+k, i*N+j, i*N+j, j*N+i, j*N+i,
                                                k*N+j, k*N+i, l*N+j, l*N+i, i*N+l, i*N+k, j*N+l, j*N+k,
                                                i*N+l, j*N+l, i*N+k, j*N+k, k*N+j, l*N+j, k*N+i, l*N+i])[symmetry]

def sparse_symmetric_einsum(nonzero_distinct_ERI, nonzero_indices, dm, backend):


    dm = dm.reshape(-1)
    diff_JK = jnp.zeros(dm.shape)
    N = int(np.sqrt(dm.shape[0]))

    def iteration(symmetry, vals): 
        diff_JK = vals 
        is_K_matrix = (symmetry >= 8)

        def sequentialized_iter(i, vals):
            # Generalized J/K computation: does J when symmetry is in range(0,8) and K when symmetry is in range(8,16)
            # Trade-off: Using one function leads to smaller always-live memory.
            diff_JK = vals 

            indices = nonzero_indices[i]

            indices = jax.lax.bitcast_convert_type(indices, np.int16).astype(np.int32)
            eris    = nonzero_distinct_ERI[i]
            print(indices.shape)

            if backend == "cpu": dm_indices = cpu_ijkl(indices, symmetry+is_K_matrix*8, indices_func)  
            else:                dm_indices = ipu_ijkl(indices, symmetry+is_K_matrix*8, N)  
            dm_values = jnp.take(dm, dm_indices, axis=0) 

            print('nonzero_distinct_ERI.shape', nonzero_distinct_ERI.shape)
            print('dm_values.shape', dm_values.shape)
            print('eris.shape', eris.shape)
            dm_values = dm_values.at[:].mul( eris ) # this is prod, but re-use variable for inplace update. 
            
            if backend == "cpu": ss_indices = cpu_ijkl(indices, symmetry+8+is_K_matrix*8, indices_func) 
            else:                ss_indices = ipu_ijkl(indices, symmetry+8+is_K_matrix*8, N) 
            diff_JK   = diff_JK + jax.ops.segment_sum(dm_values, ss_indices, N**2) * (-HYB_B3LYP/2)**is_K_matrix 
            
            return diff_JK

        batches = nonzero_indices.shape[0] # before pmap, tensor had shape (nipus, batches, -1) so [0]=batches after pmap
        diff_JK = jax.lax.fori_loop(0, batches, sequentialized_iter, diff_JK) 
        return diff_JK

    diff_JK = jax.lax.fori_loop(0, 16, iteration, diff_JK) 
    return jax.lax.psum(diff_JK, axis_name="p")



def compute_eri(mol, itol):
    input_floats, input_ints, input_ijkl, shapes, sizes, counts, indxs, indxs_inv, ao_loc = prepare_integrals_2_inputs(mol, itol)

    # Load vertex using TileJax.
    vertex_filename = osp.join(osp.dirname(__file__), "intor_int2e_sph.cpp")
    int2e_sph = create_ipu_tile_primitive(
            "poplar_int2e_sph",
            "poplar_int2e_sph",
            inputs=["ipu_floats", "ipu_ints", "ipu_ij", "ipu_output", "tile_g", "tile_idx", "tile_buf"],
            outputs={"ipu_output": 3, "tile_g": 4, "tile_idx": 5, "tile_buf": 6},
            gp_filename=vertex_filename,
            perf_estimate=100,
    )
    int2e_sph_forloop = create_ipu_tile_primitive(
                "poplar_int2e_sph_forloop",
                "poplar_int2e_sph_forloop",
                inputs=["ipu_floats", "ipu_ints", "ipu_ij", "ipu_output", "tile_g", "tile_idx", "tile_buf", "chunks", "integral_size"],
                outputs={"ipu_output": 3, "tile_g": 4, "tile_idx": 5, "tile_buf": 6},
                gp_filename=vertex_filename,
                perf_estimate=100,
        )

    all_outputs = []
    all_indices = []
    start_index = 0
    start, stop = 0, 0
    np.random.seed(42)

    NUM_TILES = 1472

    num_tiles = NUM_TILES-1
    num_threads = 6

    num_calls = len(indxs_inv)
    #print(num_calls)

    if num_calls < num_tiles:  tiles       = tuple((np.arange(num_calls)+1).tolist())
    else:                      tiles       = tuple((np.arange(num_tiles*num_threads)%(num_tiles)+1).tolist())

    # print('-> TILES:', tiles)

    tile_floats = tile_put_replicated(input_floats, tiles)
    tile_ints   = tile_put_replicated(input_ints,   tiles)

    print(sizes, counts)

    print('full count:', np.array(counts).sum())
    print('full mult count:', np.array(sizes).dot(np.array(counts)))
    # exit()

    for i, (size, count) in enumerate(zip(sizes, counts)):

        print('>>>', i, size, count)
        glen, nf, buflen = shapes[i]

        tiles       = tuple((np.arange(num_tiles*num_threads)%(num_tiles)+1).tolist())
        tile_g      = tile_put_replicated(jnp.empty(min(int(glen), 3888)+1),                     tiles)
        tile_idx    = tile_put_replicated(jnp.empty(max(256, min(int(nf*3), 3888)+1), dtype = jnp.int32) ,  tiles)
        tile_buf    = tile_put_replicated(jnp.empty(1080*4+1) ,                   tiles)

        chunk_size      = num_tiles * num_threads

        if count // chunk_size > 0:
            output = jnp.empty((len(tiles), count//chunk_size, size))
            output  = tile_put_sharded(   output,   tiles)

            _indices = []
            for j in range( count // (chunk_size) ):
                start, stop = j*chunk_size, (j+1)*chunk_size
                indices = np.array(input_ijkl[i][start:stop])
                _indices.append(indices.reshape(indices.shape[0], 1, indices.shape[1]))
            _indices = np.concatenate(_indices, axis=1)
            tile_indices = tile_put_sharded(_indices, tiles)

            chunks = tile_put_replicated( jnp.zeros(count//chunk_size), tiles)
            integral_size = tile_put_replicated( jnp.zeros(size), tiles)

            batched_out , _, _, _= tile_map(int2e_sph_forloop,
                                            tile_floats,
                                            tile_ints,
                                            tile_indices,
                                            output,
                                            tile_g,
                                            tile_idx,
                                            tile_buf,
                                            chunks,
                                            integral_size
                                            )
            batched_out = jnp.transpose(batched_out.array, (1, 0, 2)).reshape(-1, size)
            print('batched mode!')

        # do last iteration normally. -- assuming only one iteration for indices to be correct
        for j in range(count // chunk_size, count // (chunk_size) + 1):
            start, stop = j*chunk_size, (j+1)*chunk_size
            indices = np.array(input_ijkl[i][start:stop])
            if indices.shape[0] != len(tiles):
                    tiles = tuple((np.arange(indices.shape[0])%(num_tiles)+1).tolist())

            tile_ijkl   = tile_put_sharded(   indices ,     tiles)
            output = jnp.empty((len(tiles), size))
            output  = tile_put_sharded(   output+j,   tiles)

            # print('indices', indices)

            _output, _, _, _= tile_map(int2e_sph,
                tile_floats[:len(tiles)],
                tile_ints[:len(tiles)],
                tile_ijkl,
                output,
                tile_g[:len(tiles)],
                tile_idx[:len(tiles)],
                tile_buf[:len(tiles)]
            )
            print('???', j)
        
        if count//chunk_size>0:
            print(batched_out.shape)
            print(_indices.shape)
        print(_output.array.shape)
        print(np.array(indices).shape)

        if count//chunk_size>0:
            all_outputs.append(jnp.concatenate([batched_out, _output.array]))
            all_indices.append(np.concatenate([np.transpose(_indices, (1, 0, 2)).reshape(-1, 4), indices]))
        else:
            all_outputs.append(_output.array)
            all_indices.append(indices)

        start = stop

    return all_outputs, all_indices, ao_loc

def compute_diff_jk(dm, mol, nprog, nbatch, itol, backend):
    dm = dm.reshape(-1)
    diff_JK = jnp.zeros(dm.shape)
    N = int(np.sqrt(dm.shape[0])) 

    all_eris, all_indices, ao_loc = compute_eri(mol, itol)

    BLOCK_ERI_SIZE = np.sum(np.array([eri.shape[0] for eri in all_eris]))

    overlap_bookkeeping = {}
    comp_distinct_ERI_list = [None]*BLOCK_ERI_SIZE
    comp_distinct_idx_list = [None]*BLOCK_ERI_SIZE
    comp_do_list = [None]*BLOCK_ERI_SIZE
    comp_list_index = 0

    print('[a.shape for a in all_eris]', [a.shape for a in all_eris])
    print('[a.shape for a in all_indices]', [a.shape for a in all_indices])

    for eri, idx in zip(all_eris, all_indices):
        print(eri.shape)
        i = idx[:, 0]
        j = idx[:, 1]
        k = idx[:, 2]
        l = idx[:, 3]

        di = ao_loc[i+1] - ao_loc[i]
        dj = ao_loc[j+1] - ao_loc[j]
        dk = ao_loc[k+1] - ao_loc[k]
        dl = ao_loc[l+1] - ao_loc[l]

        i0 = ao_loc[i] - ao_loc[0]
        j0 = ao_loc[j] - ao_loc[0]
        k0 = ao_loc[k] - ao_loc[0]
        l0 = ao_loc[l] - ao_loc[0]

        for ind in range(eri.shape[0]):

            _di = di[ind]
            _dj = dj[ind]
            _dk = dk[ind]
            _dl = dl[ind]

            _i0 = i0[ind]
            _j0 = j0[ind]
            _k0 = k0[ind]
            _l0 = l0[ind]
            
            def ijkl2c(i, j, k, l):
                if i<j: i,j = j,i
                if k<l: k,l = l,k
                ij = i*(i+1)//2 + j
                kl = k*(k+1)//2 + l
                if ij < kl: ij,kl = kl,ij
                c = ij*(ij+1)//2 + kl
                return c

            block_idx = np.mgrid[
                _i0:(_i0+_di),
                _j0:(_j0+_dj),
                _k0:(_k0+_dk),
                _l0:(_l0+_dl)].transpose(4, 3, 2, 1, 0).astype(np.int16)

            block_c = [ijkl2c(ijkl[0],ijkl[1], ijkl[2], ijkl[3]) for ijkl in block_idx.reshape(-1, 4)]
            block_do = np.zeros((_dl*_dk*_dj*_di))
            for ci, c in enumerate(block_c):
                block_do[ci] = int(c not in overlap_bookkeeping)
                overlap_bookkeeping[c] = True
                            
            comp_distinct_idx_list[comp_list_index] = block_idx.reshape(-1, 4)
            comp_do_list[comp_list_index] = block_do
            comp_list_index += 1
    
    
    comp_distinct_idx = np.concatenate(comp_distinct_idx_list)
    comp_do = np.concatenate(comp_do_list)
    
    remainder = comp_distinct_idx.shape[0] % (nprog*nbatch)

    if remainder != 0:
        print('padding', remainder, nprog*nbatch-remainder, comp_distinct_idx.shape)
        comp_distinct_idx = np.pad(comp_distinct_idx, ((0, nprog*nbatch-remainder), (0, 0)))
        comp_do = np.pad(comp_do, ((0, nprog*nbatch-remainder)))
        all_eris.append(jnp.zeros((nprog*nbatch-remainder), dtype=jnp.float32))
    
    comp_distinct_ERI = jnp.concatenate([eri.reshape(-1) for eri in all_eris]).reshape(nprog, nbatch, -1)
    comp_distinct_idx = comp_distinct_idx.reshape(nprog, nbatch, -1, 4)
    comp_do = comp_do.reshape(nprog, nbatch, -1)
    comp_distinct_ERI *= comp_do

    print('comp_distinct_ERI.shape', comp_distinct_ERI.shape)
    print('comp_distinct_idx.shape', comp_distinct_idx.shape)

    # compute repetitions caused by 8x symmetry when computing from the distinct_ERI form and scale accordingly
    drep                      = num_repetitions_fast_4d(comp_distinct_idx[:, :, :, 0], comp_distinct_idx[:, :, :, 1], comp_distinct_idx[:, :, :, 2], comp_distinct_idx[:, :, :, 3], xnp=np, dtype=np.uint32)
    comp_distinct_ERI         = comp_distinct_ERI / drep

    # int16 storage supported but not slicing; use conversion trick to enable slicing
    comp_distinct_idx = jax.lax.bitcast_convert_type(comp_distinct_idx, jnp.float16)
    
    diff_JK = jax.pmap(sparse_symmetric_einsum, in_axes=(0,0,None,None), static_broadcasted_argnums=(3,), backend=backend, axis_name="p")(comp_distinct_ERI, comp_distinct_idx, dm, backend)

    return diff_JK

if __name__ == "__main__":
    import time 
    import argparse 
    parser = argparse.ArgumentParser(prog='', description='', epilog='')
    parser.add_argument('-backend', default="cpu"),
    parser.add_argument('-natm', default=3),
    parser.add_argument('-test', action="store_true")
    parser.add_argument('-prof', action="store_true")
    parser.add_argument('-batches', default=5, type=int)
    parser.add_argument('-nipu', default=16, type=int)
    parser.add_argument('-skip', action="store_true") 
    parser.add_argument('-itol', default=1e-9, type=float)
    
    args = parser.parse_args()
    backend = args.backend 

    natm = int(args.natm) 
    nipu = int(args.nipu)
    if backend == "cpu": nipu = 1

    start = time.time()

    mol = pyscf.gto.Mole(atom="".join(f"C 0 {1.54*j} {1.54*i};" for i in range(natm) for j in range(natm))) 
    # mol = pyscf.gto.Mole(atom="".join(f"C 0 {1.54*i} {1.54*i};" for i in range(natm))) 
    #mol = pyscf.gto.Mole(atom="".join(f"C 0 {15.4*j} {15.4*i};" for i in range(1) for j in range(75))) 
    mol.build()
    N = mol.nao_nr()
    print("N %i"%mol.nao_nr())
    if not args.skip: dense_ERI = mol.intor("int2e_sph", aosym="s1")
    distinct_ERI = mol.intor("int2e_sph", aosym="s8")

    print('distinct_ERI.shape', distinct_ERI.shape)
    if not args.skip:
        print('dense_ERI', dense_ERI.astype(np.float32).nbytes/10**6, "MB  ", np.prod(dense_ERI.shape), dense_ERI.shape)
    print('distinct_ERI', distinct_ERI.astype(np.float32).nbytes/10**6, "MB  ", np.prod(distinct_ERI.shape), distinct_ERI.shape)
    print("")

    # distinct_ERI[np.abs(distinct_ERI)<1e-9] = 0  # zero out stuff 
    dm = pyscf.scf.hf.init_guess_by_minao(mol)         
    scale = HYB_B3LYP/2
    if not args.skip: 
        J = np.einsum("ijkl,ji->kl", dense_ERI, dm)
        K = np.einsum("ijkl,jk->il", dense_ERI, dm)
        truth = J - K / 2 * HYB_B3LYP

    nonzero_indices      = np.nonzero(distinct_ERI)[0].astype(np.uint64) 
    nonzero_distinct_ERI = distinct_ERI[nonzero_indices].astype(np.float32)
    ij, kl               = get_i_j(nonzero_indices)
    rep                  = num_repetitions_fast(ij, kl)
    nonzero_distinct_ERI = nonzero_distinct_ERI / rep
    dm                   = dm.reshape(-1)
    diff_JK              = np.zeros(dm.shape)

    # ------------------------------------ #

    diff_JK = jax.jit(compute_diff_jk, backend=backend, static_argnames=['mol', 'nprog', 'nbatch', 'itol', 'backend'])(dm, mol, args.nipu, args.batches, args.itol, args.backend)

    # ------------------------------------ #

    # diff_JK = jax.pmap(sparse_symmetric_einsum, in_axes=(0,0,None,None), static_broadcasted_argnums=(3,), backend=backend, axis_name="p")(nonzero_distinct_ERI, nonzero_indices, dm, args.backend) 

    if args.skip: 
        exit()
    if args.nipu > 1:
        diff_JK = np.array(diff_JK[0])
    else:
        diff_JK = np.array(diff_JK) # avoid multiple profiles

    diff_JK = diff_JK.reshape(N, N)
    print(diff_JK.reshape(-1)[::51])
    print(truth.reshape(-1)[::51])
    print(np.max(np.abs(diff_JK.reshape(-1) - truth.reshape(-1))))
    assert np.allclose(diff_JK, truth, atol=1e-6)
    print("PASSED!")