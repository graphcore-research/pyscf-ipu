# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
# todo
import pyscf 
import numpy as np 
import jax 
import jax.numpy as jnp 
import os.path as osp
from tessellate_ipu import create_ipu_tile_primitive, ipu_cycle_count, tile_map, tile_put_sharded, tile_put_replicated
from functools import partial 
from icecream import ic
from tqdm import tqdm
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

vertex_filename  = osp.join(osp.dirname(__file__), "compute_indices.cpp")
compute_indices= create_ipu_tile_primitive(
        "IndicesIJKL" ,
        "IndicesIJKL" ,
        inputs=["i_", "j_", "k_", "l_", "sym_", "N_", "start_", "stop_"], 
        outputs={"out_": 0},
        gp_filename=vertex_filename,
        perf_estimate=100,
)

@partial(jax.jit, backend="ipu")
def ipu_ijkl(nonzero_indices, symmetry, N):
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

def get_shapes(input_ijkl, bas):
    i_sh, j_sh, k_sh, l_sh  = input_ijkl[0]
    BAS_SLOTS = 8
    NPRIM_OF = 2
    NCTR_OF = 3
    ANG_OF = 1
    GSHIFT = 4

    i_ctr   = bas.reshape(-1)[BAS_SLOTS * i_sh + NCTR_OF]
    j_ctr   = bas.reshape(-1)[BAS_SLOTS * j_sh + NCTR_OF]
    k_ctr   = bas.reshape(-1)[BAS_SLOTS * k_sh + NCTR_OF]
    l_ctr   = bas.reshape(-1)[BAS_SLOTS * l_sh + NCTR_OF]

    i_l     = bas.reshape(-1)[BAS_SLOTS * i_sh + ANG_OF]
    j_l     = bas.reshape(-1)[BAS_SLOTS * j_sh + ANG_OF]
    k_l     = bas.reshape(-1)[BAS_SLOTS * k_sh + ANG_OF]
    l_l     = bas.reshape(-1)[BAS_SLOTS * l_sh + ANG_OF]

    nfi  = (i_l+1)*(i_l+2)/2
    nfj  = (j_l+1)*(j_l+2)/2
    nfk  = (k_l+1)*(k_l+2)/2
    nfl  = (l_l+1)*(l_l+2)/2
    nf = nfi * nfk * nfl * nfj
    n_comp = 1

    nc = i_ctr * j_ctr * k_ctr * l_ctr
    lenl = nf * nc * n_comp
    lenk = nf * i_ctr * j_ctr * k_ctr * n_comp
    lenj = nf * i_ctr * j_ctr * n_comp
    leni = nf * i_ctr * n_comp
    len0 = nf * n_comp

    ng = [0, 0, 0, 0, 0, 1, 1, 1]

    IINC=0
    JINC=1
    KINC=2
    LINC=3

    li_ceil = i_l + ng[IINC]
    lj_ceil = j_l + ng[JINC]
    lk_ceil = k_l + ng[KINC]
    ll_ceil = l_l + ng[LINC]
    nrys_roots = (li_ceil + lj_ceil + lk_ceil + ll_ceil)/2 + 1


    ibase = li_ceil > lj_ceil
    kbase = lk_ceil > ll_ceil
    if (nrys_roots <= 2):
        ibase = 0
        kbase = 0
    if (kbase) :
        dlk = lk_ceil + ll_ceil + 1
        dll = ll_ceil + 1
    else:
        dlk = lk_ceil + 1
        dll = lk_ceil + ll_ceil + 1

    if (ibase) :
        dli = li_ceil + lj_ceil + 1
        dlj = lj_ceil + 1
    else :
        dli = li_ceil + 1
        dlj = li_ceil + lj_ceil + 1

    g_size     = nrys_roots * dli * dlk * dll * dlj
    gbits        = ng[GSHIFT]
    leng = g_size*3*((1<<gbits)+1)

    len = leng + lenl + lenk + lenj + leni + len0

    return len, nf

def compute_diff_jk(dm, mol, nprog, nbatch, tolerance, backend):
    dm = dm.reshape(-1)
    diff_JK = jnp.zeros(dm.shape)
    N = int(np.sqrt(dm.shape[0])) 

    # 50mb     100mb 
    #all_eris, all_indices, ao_loc = compute_eri(mol, itol)
    atm, bas, env   = mol._atm, mol._bas, mol._env
    n_atm, n_bas, N = atm.shape[0], bas.shape[0], mol.nao_nr()
    ao_loc          = np.cumsum(np.concatenate([np.zeros(1), (bas[:,1]*2+1) * bas[:,3] ])).astype(np.int32)
    n_ao_loc        = np.prod(ao_loc.shape)

    n_buf, n_eri, n_env = 1, 1, np.prod(env.shape) # TODO: Try to remove this. 

    # Step 1. Compute indices where ERI is non-zero due to geometry (pre-screening). 
    # Below computes: np.max([ERI[a,b,a,b] for a,b in zip(tril_idx[0], tril_idx[1])])
    ERI_s8 = mol.intor("int2e_sph", aosym="s8") # TODO: make custom libcint code compute this. 
    lst_abab = np.zeros((N*(N+1)//2), dtype=np.float32)
    lst_ab   = np.zeros((N*(N+1)//2, 2), dtype=np.int32)
    tril_idx = np.tril_indices(N)
    for c, (a, b) in tqdm(enumerate(zip(tril_idx[0], tril_idx[1]))):
        index_ab_s8 = a*(a+1)//2 + b
        index_s8 = index_ab_s8*(index_ab_s8+3)//2
        abab = np.abs(ERI_s8[index_s8])
        # lst.append((abab, a,b))
        lst_abab[c] = abab
        lst_ab[c, :] = (a, b)
    abab_max = np.max(lst_abab)
    considered_indices = set([(a,b) for abab, (a, b) in tqdm(zip(lst_abab, lst_ab)) if abab*abab_max >= tolerance**2])
    print('n_bas', n_bas)
    print('ao_loc', ao_loc)

    # Step 2: Remove zeros by orthogonality and match indices with shells. 
    # Precompute Fill input_ijkl and output_sizes with the necessary indices.
    n_upper_bound = (n_bas*(n_bas-1))**2
    input_ijkl    = np.zeros((n_upper_bound, 4), dtype=np.int32)
    output_sizes  = np.zeros((n_upper_bound, 5))

    sym_pattern =  np.array([(i+3)%5!=0 for i in range(N)])
    nonzero_seed = sym_pattern
    num_calls = 0
    
    for i in tqdm(range(n_bas)): # consider all shells << all ijkl 
        for j in range(i+1):
            for k in range(i, n_bas):
                for l in range(k+1):
                    di, dj, dk, dl = [ao_loc[z+1] - ao_loc[z] for z in [i,j,k,l]]

                    found_nonzero = False
                    # check i,j boxes
                    for bi in range(ao_loc[i], ao_loc[i+1]):
                        for bj in range(ao_loc[j], ao_loc[j+1]):
                            if (bi, bj) in considered_indices: # if ij box is considered
                                # check if kl pairs are considered
                                for bk in range(ao_loc[k], ao_loc[k+1]):
                                    if bk>=bi: # apply symmetry - tril fade vertical
                                        mla = ao_loc[l]
                                        if bk == bi:
                                            mla = max(bj, ao_loc[l]) # apply symmetry - tril fade horizontal
                                        for bl in range(mla, ao_loc[l+1]):
                                            if (bk, bl) in considered_indices:
                                                # apply grid pattern to find final nonzeros
                                                if ~(nonzero_seed[bi] ^ nonzero_seed[bj]) ^ (nonzero_seed[bk] ^ nonzero_seed[bl]):
                                                    found_nonzero = True
                                                    break
                                    if found_nonzero: break
                            if found_nonzero: break
                        if found_nonzero: break
                    if not found_nonzero: continue 

                    input_ijkl[num_calls] = [i, j, k, l]
                    output_sizes[num_calls] = [di, dj, dk, dl, di*dj*dk*dl]
                    num_calls += 1

    input_ijkl   = input_ijkl[:num_calls, :]
    output_sizes = output_sizes[:num_calls, :]

    # Prepare IPU inputs.
    # Merge all int/float inputs in seperate arrays.
    input_floats = env.reshape(1, -1)
    input_ints = np.hstack([n_eri, n_buf, n_atm, n_bas, n_env, n_ao_loc, ao_loc.reshape(-1), atm.reshape(-1), bas.reshape(-1)])

    sizes, counts = np.unique(output_sizes[:, -1], return_counts=True)
    sizes, counts = sizes.astype(np.int32), counts.astype(np.int32)

    indxs      = np.argsort(output_sizes[:, -1])
    input_ijkl          = input_ijkl[indxs]

    sizes, counts  = np.unique(output_sizes[:, -1], return_counts=True)
    sizes, counts = sizes.astype(np.int32), counts.astype(np.int32)
    start_index = 0
    inputs = []
    shapes = []
    for i, (size, count) in enumerate(zip(sizes, counts)):
        a = input_ijkl[start_index: start_index+count]
        tuples = tuple(map(tuple, a))
        inputs.append(tuples)
        start_index += count

    input_ijkl = tuple(inputs)

    for i in range(len(sizes)):
        shapes.append(get_shapes(inputs[i], bas))

    shapes, sizes, counts = tuple(shapes), tuple(sizes.tolist()), counts.tolist()

    # Load vertex using TileJax.
    vertex_filename = osp.join(osp.dirname(__file__), "intor_int2e_sph.cpp")
    int2e_sph_forloop = create_ipu_tile_primitive(
                "poplar_int2e_sph_forloop",
                "poplar_int2e_sph_forloop",
                inputs=["ipu_floats", "ipu_ints", "ipu_ij", "ipu_output", "tile_g", "tile_idx", "tile_buf", "chunks", "integral_size"],
                outputs={"ipu_output": 3, "tile_g": 4, "tile_idx": 5, "tile_buf": 6},
                gp_filename=vertex_filename,
                perf_estimate=100,
        )

    all_eris = []
    all_indices = []
    start_index = 0
    np.random.seed(42)

    NUM_TILES = 1472

    num_tiles = NUM_TILES-1
    num_threads = 6

    if num_calls < num_tiles:  tiles       = tuple((np.arange(num_calls)+1).tolist())
    else:                      tiles       = tuple((np.arange(num_tiles*num_threads)%(num_tiles)+1).tolist())

    tile_floats = tile_put_replicated(input_floats, tiles)
    tile_ints   = tile_put_replicated(input_ints,   tiles)

    print(sizes, counts)

    print('full count:', np.array(counts).sum())
    print('full mult count:', np.array(sizes).dot(np.array(counts)))

    for i, (size, count) in enumerate(zip(sizes, counts)):
        print('>>>', i, size, count) # the small test case won't test this. 
        glen, nf = shapes[i]

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

            chunks = tile_put_replicated( jnp.array(count//chunk_size, dtype=jnp.uint32), tiles)
            integral_size = tile_put_replicated( jnp.array(size, dtype=jnp.uint32), tiles)

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

            chunks = tile_put_replicated( jnp.array(1, dtype=jnp.uint32), tiles)
            integral_size = tile_put_replicated( jnp.array(size, dtype=jnp.uint32), tiles)

            _output, _, _, _= tile_map(int2e_sph_forloop,
                tile_floats[:len(tiles)],
                tile_ints[:len(tiles)],
                tile_ijkl,
                output,
                tile_g[:len(tiles)],
                tile_idx[:len(tiles)],
                tile_buf[:len(tiles)],
                chunks,
                integral_size
            )
            print('???', j)
        
        if count//chunk_size>0:
            print(batched_out.shape)
            print(_indices.shape)
        print(_output.array.shape)
        print(np.array(indices).shape)

        if count//chunk_size>0:
            all_eris.append(jnp.concatenate([batched_out, _output.array]))
            all_indices.append(np.concatenate([np.transpose(_indices, (1, 0, 2)).reshape(-1, 4), indices]))
        else:
            all_eris.append(_output.array)
            all_indices.append(indices)

        start = stop

    print('[a.shape for a in all_eris]', [a.shape for a in all_eris])
    print('[a.shape for a in all_indices]', [a.shape for a in all_indices])

    temp = 0
    for zip_counter, (eri, idx) in enumerate(zip(all_eris, all_indices)):
        # go from our memory layout to mol.intor("int2e_sph", "s8")

        num_shells, shell_size = eri.shape # save original tensor shape

        def compute_full_shell_idx(idx):
            comp_distinct_idx_list = []
            for ind in range(eri.shape[0]):
                i, j, k, l         = [idx[ind, z] for z in range(4)]
                _di, _dj, _dk, _dl = ao_loc[i+1] - ao_loc[i], ao_loc[j+1] - ao_loc[j], ao_loc[k+1] - ao_loc[k], ao_loc[l+1] - ao_loc[l]
                _i0, _j0, _k0, _l0 = ao_loc[i], ao_loc[j], ao_loc[k], ao_loc[l]
                block_idx = np.mgrid[
                    _i0:(_i0+_di),
                    _j0:(_j0+_dj),
                    _k0:(_k0+_dk),
                    _l0:(_l0+_dl)].transpose(4, 3, 2, 1, 0) #.astype(np.int16)
                                
                comp_distinct_idx_list.append(block_idx.reshape(-1, 4))
            comp_distinct_idx = np.concatenate(comp_distinct_idx_list)
            return comp_distinct_idx
        
        comp_distinct_idx = compute_full_shell_idx(idx)

        ijkl_arr = np.sum([np.prod(np.array(a).shape) for a in input_ijkl])
        print('input_ijkl.nbytes/1e6', ijkl_arr*2/1e6)
        print('comp_distinct_idx.nbytes/1e6', comp_distinct_idx.astype(np.int16).nbytes/1e6)
        
        remainder = (eri.shape[0]) % (nprog*nbatch)

        # unused for nipu==batches==1
        if remainder != 0:
            print('padding', remainder, nprog*nbatch-remainder, comp_distinct_idx.shape)
            comp_distinct_idx = np.pad(comp_distinct_idx.reshape(-1, shell_size, 4), ((0, (nprog*nbatch-remainder)), (0, 0), (0, 0))).reshape(-1, 4)
            eri = jnp.pad(eri, ((0, nprog*nbatch-remainder), (0, 0)))
            idx = jnp.pad(idx, ((0, nprog*nbatch-remainder), (0, 0)))
            
        comp_distinct_ERI = eri.reshape(nprog, nbatch, -1)
        comp_distinct_idx = comp_distinct_idx.reshape(nprog, nbatch, -1, 4)
        idx = idx.reshape(nprog, nbatch, -1, 4)


        # nonzero_distinct_ERI, nonzero_indices, dm, backend = comp_distinct_ERI[0], comp_distinct_idx[0], dm, backend
        nonzero_distinct_ERI, nonzero_indices, dm, backend = comp_distinct_ERI[0], idx[0], dm, backend
        
        
        dm = dm.reshape(-1)
        diff_JK = jnp.zeros(dm.shape)
        N = int(np.sqrt(dm.shape[0]))

        def foreach_batch(i, vals): 
            diff_JK, nonzero_indices, ao_loc = vals 

            eris    = nonzero_distinct_ERI[i].reshape(-1)

            if False:

                indices = nonzero_indices[i]
                # # indices = jax.lax.bitcast_convert_type(indices, np.int16).astype(np.int32)
                indices = indices.astype(jnp.int32)                

                print('eris.shape', eris.shape)
                print('indices.shape', indices.shape)

            else:
                # Compute offsets and sizes
                idx = nonzero_indices[i]
                _i, _j, _k, _l     = [idx[:, z] for z in range(4)]
                _di, _dj, _dk, _dl = [(ao_loc[z+1] - ao_loc[z]).reshape(-1, 1) for z in [_i, _j, _k, _l]]
                _i0, _j0, _k0, _l0 = [ao_loc[z].reshape(-1, 1) for z in [_i, _j, _k, _l]]

                def gen_shell_idx(idx_sh):
                    idx_sh = idx_sh.reshape(-1, shell_size)
                    # Compute the indices
                    ind_i = (idx_sh                 ) % _di + _i0
                    ind_j = (idx_sh // (_di)        ) % _dj + _j0
                    ind_k = (idx_sh // (_di*_dj)    ) % _dk + _k0
                    ind_l = (idx_sh // (_di*_dj*_dk)) % _dl + _l0
                    print('>>', ind_i.shape)
                    # Update the array with the computed indices
                    return jnp.stack([ind_i.reshape(-1), ind_j.reshape(-1), ind_k.reshape(-1), ind_l.reshape(-1)], axis=1)
                
                indices = gen_shell_idx(jnp.arange((eris.shape[0]))) # <<<<<<<<<<<<<<<<<<<<<<<<<

            print('eris.shape', eris.shape)
            print('indices.shape', indices.shape)

            # compute repetitions caused by 8x symmetry when computing from the distinct_ERI form and scale accordingly
            drep = num_repetitions_fast_4d(indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3], xnp=jnp, dtype=jnp.uint32)
            eris = eris / drep
            

            def foreach_symmetry(sym, vals):
                # Generalized J/K computation: does J when symmetry is in range(0,8) and K when symmetry is in range(8,16)
                # Trade-off: Using one function leads to smaller always-live memory.
                is_K_matrix = (sym >= 8)
                diff_JK = vals 

                if backend == "cpu": dm_indices = cpu_ijkl(indices, sym+is_K_matrix*8, indices_func)  
                else:                dm_indices = ipu_ijkl(indices, sym+is_K_matrix*8, N)  
                dm_values = jnp.take(dm, dm_indices, axis=0) 

                print('indices.shape', indices.shape)
                print('dm_values.shape', dm_values.shape)
                print('eris.shape', eris.shape)
                dm_values = dm_values.at[:].mul( eris ) # this is prod, but re-use variable for inplace update. 
                
                if backend == "cpu": ss_indices = cpu_ijkl(indices, sym+8+is_K_matrix*8, indices_func) 
                else:                ss_indices = ipu_ijkl(indices, sym+8+is_K_matrix*8, N) 
                diff_JK   = diff_JK + jax.ops.segment_sum(dm_values, ss_indices, N**2) * (-HYB_B3LYP/2)**is_K_matrix 
                
                return diff_JK
           
            diff_JK = jax.lax.fori_loop(0, 16, foreach_symmetry, diff_JK) 
            
            return (diff_JK, nonzero_indices, ao_loc)

        batches = nonzero_indices.shape[0] # before pmap, tensor had shape (nipus, batches, -1) so [0]=batches after pmap
        
        diff_JK, _, _ = jax.lax.fori_loop(0, batches, foreach_batch, (diff_JK, nonzero_indices, ao_loc)) 

        temp += diff_JK

    return temp

if __name__ == "__main__":
    import time 
    import argparse 
    parser = argparse.ArgumentParser(prog='', description='', epilog='')
    parser.add_argument('-backend', default="cpu"),
    parser.add_argument('-natm', default=3),
    parser.add_argument('-test', action="store_true")
    parser.add_argument('-prof', action="store_true")
    parser.add_argument('-batches', default=5, type=int)
    parser.add_argument('-nipu', default=1, type=int)
    parser.add_argument('-skip', action="store_true") 
    parser.add_argument('-itol', default=1e-9, type=float)
    
    args = parser.parse_args()
    backend = args.backend 

    natm = int(args.natm) 
    nipu = int(args.nipu)
    if backend == "cpu": nipu = 1

    start = time.time()

    #mol = pyscf.gto.Mole(atom="".join(f"C 0 {1.54*j} {1.54*i};" for i in range(natm) for j in range(natm))) # sto-3g by default
    # mol = pyscf.gto.Mole(atom="".join(f"C 0 {1.54*j} {1.54*i};" for i in range(1) for j in range(2)), basis="sto3g") 
    mol = pyscf.gto.Mole(atom="".join(f"C 0 {1.54*j} {1.54*i};" for i in range(natm) for j in range(natm)), basis="sto3g") 
    #mol = pyscf.gto.Mole(atom="".join(f"C 0 {1.54*j} {1.54*i};" for i in range(1) for j in range(1)), basis="def2-TZVPPD") 
    #mol = pyscf.gto.Mole(atom="".join(f"C 0 {1.54*j} {1.54*i};" for i in range(1) for j in range(2)), basis="6-31G*") 
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

    diff_JK = jax.jit(compute_diff_jk, backend=backend, static_argnames=['mol', 'nprog', 'nbatch', 'tolerance', 'backend'])(dm, mol, args.nipu, args.batches, args.itol, args.backend)

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