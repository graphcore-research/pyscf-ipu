# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
# todo
import pyscf 
import numpy as np 
import jax 
import jax.numpy as jnp 
import os.path as osp
from tessellate_ipu import create_ipu_tile_primitive, ipu_cycle_count, tile_map, tile_put_sharded, tile_put_replicated
from functools import partial, lru_cache
from icecream import ic
from tqdm import tqdm
import utils
from utils import process_mol_str

jax.config.update('jax_platform_name', "cpu")
#jax.config.update('jax_enable_x64', True) 
HYB_B3LYP = 0.2

vertex_filename  = osp.join(osp.dirname(__file__), "compute_indices.cpp")
compute_indices= create_ipu_tile_primitive(
        "IndicesIJKL" ,
        "IndicesIJKL" ,
        inputs=["i_", "j_", "k_", "l_", "sym_", "N_", "start_", "stop_"], 
        outputs={"out_": 0},
        gp_filename=vertex_filename,
        perf_estimate=100,
)

def cpu_ijkl(value, symmetry, N): 
    f = lambda i,j,k,l,symmetry: jnp.array([i*N+j, j*N+i, i*N+j, j*N+i, k*N+l, l*N+k, k*N+l, l*N+k,
                                            k*N+l, k*N+l, l*N+k, l*N+k, i*N+j, i*N+j, j*N+i, j*N+i,
                                            k*N+j, k*N+i, l*N+j, l*N+i, i*N+l, i*N+k, j*N+l, j*N+k,
                                            i*N+l, j*N+l, i*N+k, j*N+k, k*N+j, l*N+j, k*N+i, l*N+i])[symmetry]
    i, j, k, l = value[0].astype(np.uint32), value[1].astype(np.uint32), value[2].astype(np.uint32), value[3].astype(np.uint32)
    return f(i,j,k,l,symmetry)
cpu_ijkl = jax.vmap(cpu_ijkl, in_axes=(0, None, None))

@partial(jax.jit, backend="ipu")
def ipu_ijkl(nonzero_indices, symmetry, N):
    size = nonzero_indices.shape[0]
    total_threads = (1472-1) * 6 
    remainder = size % total_threads

    if remainder != 0:
        nonzero_indices = jnp.pad(nonzero_indices, ((0, total_threads-remainder), (0, 0)))

    i, j, k, l = [nonzero_indices[:, x].astype(np.uint32).reshape(total_threads, -1) for x in range(4)] 

    tiles = tuple((np.arange(0,total_threads) % (1471) + 1).astype(np.uint32).tolist())
    symmetry = tile_put_replicated(jnp.array(symmetry, dtype=jnp.uint32),   tiles) 
    N        = tile_put_replicated(jnp.array(N, dtype=jnp.uint32),   tiles)
    start    = tile_put_replicated(jnp.array(0, dtype=jnp.uint32),   tiles)
    stop     = tile_put_replicated(jnp.array(i.shape[1], dtype=jnp.uint32),   tiles)

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

# gen_shells is designed to be called multiple times to simplify the rest of the code, but compute shells only once (lru_cache)
@lru_cache(maxsize=None) # cache results
def gen_shells(mol, tolerance, ndevices, fast_shells=True):
    N = mol.nao_nr()
    bas = mol._bas
    n_bas = bas.shape[0]
    ao_loc = np.cumsum(np.concatenate([np.zeros(1), (bas[:,1]*2+1) * bas[:,3] ])).astype(np.int32)

    # Step 1. Compute up to n(n+1)/2 considered indices (O(N^2))
    if tolerance > 0:
        # Compute indices where ERI is non-zero due to geometry (pre-screening). 
        # Below computes: np.max([ERI[a,b,a,b] for a,b in zip(tril_idx[0], tril_idx[1])])
        ERI_s8 = mol.intor("int2e_sph", aosym="s8") # TODO: make custom libcint code compute this. 
        lst_abab = np.zeros((N*(N+1)//2), dtype=np.float32)
        lst_ab   = np.zeros((N*(N+1)//2, 2), dtype=np.int32)
        tril_idx = np.tril_indices(N)
        for c, (a, b) in tqdm(enumerate(zip(tril_idx[0], tril_idx[1]))):
            index_ab_s8 = a*(a+1)//2 + b
            index_s8 = index_ab_s8*(index_ab_s8+3)//2
            abab = np.abs(ERI_s8[index_s8])
            lst_abab[c] = abab
            lst_ab[c, :] = (a, b)
        abab_max = np.max(lst_abab)
        considered_indices_list = [(a,b) for abab, (a, b) in tqdm(zip(lst_abab, lst_ab)) if abab*abab_max >= tolerance**2]
        considered_indices_array = np.array(considered_indices_list)
        considered_indices = set(considered_indices_list)
    else:
        lst_ab   = np.zeros((N*(N+1)//2, 2), dtype=np.int32)
        tril_idx = np.tril_indices(N)
        considered_indices_list = [(a, b) for a, b in zip(tril_idx[0], tril_idx[1])]
        considered_indices_array = np.array(considered_indices_list)
        considered_indices = set(considered_indices_list)
    print('n_bas', n_bas)
    print('ao_loc', ao_loc)

    if fast_shells:
        import cpp_shell_gen
        print('Computing shells', end=' ')
        test_input_ijkl, test_output_sizes, num_calls = cpp_shell_gen.compute_indices(n_bas, ao_loc, considered_indices)
        print('done.')
        print(test_input_ijkl.shape)
        print(test_output_sizes.shape)
        
        input_ijkl    = test_input_ijkl[np.any(test_input_ijkl>=0, axis=1)] #test_input_ijkl[:num_calls, :]
        output_sizes  = test_output_sizes[np.any(test_output_sizes>=0, axis=1)] #test_output_sizes[:num_calls, :]
        assert input_ijkl.shape[0] == num_calls, (input_ijkl.shape[0], num_calls)
        assert output_sizes.shape[0] == num_calls, (output_sizes.shape[0], num_calls)
    else:
        # Step 2: Remove zeros by orthogonality and match indices with shells. 
        # Precompute Fill input_ijkl and output_sizes with the necessary indices.
        ortho_pattern =  np.array([(i+3)%5!=0 for i in range(N)]) # hardcoded
        n_upper_bound = (n_bas*(n_bas-1))**2
        input_ijkl    = np.zeros((n_upper_bound, 4), dtype=np.int32)
        output_sizes  = np.zeros((n_upper_bound, 5))

        num_calls = 0

        for i in tqdm(range(n_bas), desc="Computing shells"): # consider all shells << all ijkl 
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
                                                    if False and not ortho_pattern[bi] ^ ortho_pattern[bj] ^ ortho_pattern[bk] ^ ortho_pattern[bl]:
                                                        found_nonzero = True
                                                        break
                                                    else: 
                                                        found_nonzero = True
                                                        break
                                        if found_nonzero: break
                                if found_nonzero: break
                            if found_nonzero: break
                        if not found_nonzero: continue 

                        input_ijkl[num_calls] = [i, j, k, l]
                        output_sizes[num_calls] = [di, dj, dk, dl, di*dj*dk*dl]
                        num_calls += 1

        input_ijkl    = input_ijkl[:num_calls, :]
        output_sizes  = output_sizes[:num_calls, :]

    sizes, counts = [tuple(out.astype(np.int32).tolist()) for out in np.unique(output_sizes[:, -1], return_counts=True)]
    
    input_ijkl    = input_ijkl[np.argsort(output_sizes[:, -1])]
    input_ijkl = [tuple(map(tuple, input_ijkl[start_index:start_index+count])) for start_index, count in zip(np.cumsum(np.concatenate([[0], counts[:-1]])), counts)]

    print('before [len(ijkl) for ijkl in input_ijkl]', [len(ijkl) for ijkl in input_ijkl])

    new_counts = [0]*len(counts)
    for tup_id in range(len(input_ijkl)):
        pmap_remainder = len(input_ijkl[tup_id]) % ndevices
        print('pmap_remainder', pmap_remainder)
        padding = 0
        if pmap_remainder > 0:
            input_ijkl[tup_id] += tuple([[-1, -1, -1, -1]]*(ndevices-pmap_remainder)) # -1s are accounted for later after eri computation
            padding = (ndevices-pmap_remainder)
        new_counts[tup_id] = counts[tup_id]+padding
    counts = tuple(new_counts)

    print('after [len(ijkl) for ijkl in input_ijkl]', [len(ijkl) for ijkl in input_ijkl])

    input_ijkl = tuple(input_ijkl)

    shapes = tuple([get_shapes(input_ijkl[i], bas) for i in range(len(sizes))])

    return input_ijkl, sizes, counts, shapes

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

    return len, nf, nc

vertex_filename = osp.join(osp.dirname(__file__), "intor_int2e_sph_condensed.cpp")
int2e_sph_forloop = create_ipu_tile_primitive(
            "poplar_int2e_sph_forloop",
            "poplar_int2e_sph_forloop",
            inputs=["ipu_floats", "ipu_ints", "ipu_ij", "ipu_output", "tile_g", "tile_idx", "tile_buf", "chunks", "integral_size", "gctr_buf"],
            outputs={"ipu_output": 3, "tile_g": 4, "tile_idx": 5, "tile_buf": 6},
            gp_filename=vertex_filename,
            perf_estimate=100,
    )

def compute_diff_jk(input_ijkl_slice, dm, mol, nbatch, tolerance, ndevices, backend):
    # if backend == "ipu": dm, start = utils.get_ipu_cycles(dm)

    dm = dm.reshape(-1)
    diff_JK = jnp.zeros(dm.shape)
    N = int(np.sqrt(dm.shape[0])) 

    # 50mb     100mb 
    #all_eris, all_indices, ao_loc = compute_eri(mol, itol)
    atm, bas, env   = mol._atm, mol._bas, mol._env
    n_atm, n_bas, N = atm.shape[0], bas.shape[0], mol.nao_nr()
    ao_loc          = np.cumsum(np.concatenate([np.zeros(1), (bas[:,1]*2+1) * bas[:,3] ])).astype(np.int32)
    n_ao_loc        = np.prod(ao_loc.shape)

    # Prepare IPU inputs.
    # Merge all int/float inputs in seperate arrays.
    input_floats  = env.reshape(1, -1)
    input_ints    = np.hstack([0, 0, n_atm, n_bas, 0, n_ao_loc, ao_loc.reshape(-1), atm.reshape(-1), bas.reshape(-1)]) # todo 0s not used (n_buf, n_eri, n_env)

    input_ijkl, sizes, counts, shapes = gen_shells(mol, tolerance, ndevices, fast_shells=True)

    all_eris = []
    all_indices = []
    np.random.seed(42) # is this needed?

    NUM_TILES = 1472

    num_tiles = NUM_TILES-1
    num_threads = 6

    num_calls = np.sum(counts)
    if num_calls < num_tiles:  tiles = tuple((np.arange(num_calls)+1).tolist())
    else:                      tiles = tuple((np.arange(num_tiles*num_threads)%(num_tiles)+1).tolist())

    tile_floats   = tile_put_replicated(input_floats, tiles)
    tile_ints     = tile_put_replicated(input_ints,   tiles)

    slice_indices = np.cumsum([0]+[c*4//ndevices for c in counts])
    print('slice_indices', slice_indices)

    for i, (size, count) in enumerate(zip(sizes, counts)):
        print('>>>', size, count)
        slice_offset = (count//ndevices)*jax.lax.axis_index('p')
        slice_count = count // ndevices
        slice_ijkl = input_ijkl_slice[slice_indices[i]:slice_indices[i+1]].reshape(-1, 4)

        glen, nf, nc = shapes[i]
        chunk_size  = num_tiles * num_threads
        num_full_batches = slice_count//chunk_size

        tiles         = tuple((np.arange(num_tiles*num_threads)%(num_tiles)+1).tolist())
        tile_g        = tile_put_replicated(jnp.empty(min(int(glen), 3888)+1), tiles)
        tile_idx      = tile_put_replicated(jnp.empty(256, dtype=jnp.int32), tiles) # condensed version only # sto3g, 631g
        tile_buf      = tile_put_replicated(jnp.empty(81), tiles) # condensed version only # sto3g, 631g
        integral_size = tile_put_replicated(jnp.array(size, dtype=jnp.uint32), tiles)
        tile_gctr_buf = tile_put_replicated(jnp.empty((81+1), dtype=jnp.float32), tiles) # condensed version only # sto3g, 631g
        
        def batched_compute(start, stop, chunk_size, tiles):
            assert (stop-start) < chunk_size or (stop-start) % chunk_size == 0
            num_batches = max(1, (stop-start)//chunk_size)
            idx = jnp.array(slice_ijkl[start:stop]).reshape(-1, num_batches, 4) # contains -1s for padding shells
            out , _, _, _= tile_map(int2e_sph_forloop,
                                    tile_floats[:len(tiles)],
                                    tile_ints[:len(tiles)],
                                    tile_put_sharded(idx, tiles),
                                    tile_put_sharded(jnp.empty((len(tiles), num_batches, size)), tiles),
                                    tile_g[:len(tiles)],
                                    tile_idx[:len(tiles)],
                                    tile_buf[:len(tiles)],
                                    tile_put_replicated(jnp.array(num_batches, dtype=jnp.uint32), tiles),
                                    integral_size[:len(tiles)],
                                    tile_gctr_buf[:len(tiles)])
            return out.array.reshape(-1, size), jnp.maximum(idx.reshape(-1, 4), 0) # account for -1s, convert to 0s

        if num_full_batches > 0: f_out, f_idx = batched_compute(0, num_full_batches*chunk_size, chunk_size, tiles)
        else: f_out, f_idx = np.array([]).reshape(0, size), np.array([]).reshape(0, 4)

        tiles         = tuple((np.arange(slice_count-num_full_batches*chunk_size)%(num_tiles)+1).tolist())
        out, idx      = batched_compute(num_full_batches*chunk_size, slice_count, chunk_size, tiles)

        all_eris.append(jnp.concatenate([f_out, out]))
        all_indices.append(jnp.concatenate([f_idx, idx])) #.astype(jnp.uint8))

    print('[a.shape for a in all_eris]', [a.shape for a in all_eris])
    print('[a.shape for a in all_indices]', [a.shape for a in all_indices])

    total_diff_JK = 0
    for zip_counter, (eri, idx) in enumerate(zip(all_eris, all_indices)):
        num_shells, shell_size = eri.shape # save original tensor shape

        remainder = (eri.shape[0]) % (nbatch)

        # pad tensors; unused for nipu==batches==1
        if remainder != 0:
            eri = jnp.pad(eri, ((0, nbatch-remainder), (0, 0)))
            idx = jnp.pad(idx, ((0, nbatch-remainder), (0, 0)))
            
        nonzero_distinct_ERI = eri.reshape(nbatch, -1)
        nonzero_indices = idx.reshape(nbatch, -1, 4)

        dm = dm.reshape(-1)
        diff_JK = jnp.zeros(dm.shape)
        N = int(np.sqrt(dm.shape[0]))

        def foreach_batch(i, vals):
            diff_JK, nonzero_indices, ao_loc = vals 

            # Compute offsets and sizes
            batch_idx = nonzero_indices[i]
            _i, _j, _k, _l     = [batch_idx[:, z].astype(jnp.uint32) for z in range(4)]
            _di, _dj, _dk, _dl = [(ao_loc[z+1] - ao_loc[z]).reshape(-1, 1) for z in [_i, _j, _k, _l]]
            _i0, _j0, _k0, _l0 = [ao_loc[z].reshape(-1, 1) for z in [_i, _j, _k, _l]]

            def gen_shell_idx(idx_sh):
                # Compute the indices
                ind_i = (idx_sh                 ) % _di + _i0
                ind_j = (idx_sh // (_di)        ) % _dj + _j0
                ind_k = (idx_sh // (_di*_dj)    ) % _dk + _k0
                ind_l = (idx_sh // (_di*_dj*_dk)) % _dl + _l0

                # Update the array with the computed indices
                return jnp.stack([ind_i.reshape(-1), ind_j.reshape(-1), ind_k.reshape(-1), ind_l.reshape(-1)], axis=1)
            
            eris = nonzero_distinct_ERI[i].reshape(-1)
            indices = gen_shell_idx(jnp.arange((eris.shape[0])).reshape(-1, shell_size))

            # compute repetitions caused by 8x symmetry when computing from the distinct_ERI form and scale accordingly
            drep = num_repetitions_fast_4d(indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3], xnp=jnp, dtype=jnp.uint32)
            eris = eris / drep

            def foreach_symmetry(sym, vals):
                # Generalized J/K computation: does J when symmetry is in range(0,8) and K when symmetry is in range(8,16)
                # Trade-off: Using one function leads to smaller always-live memory.
                is_K_matrix = (sym >= 8)
                diff_JK = vals 

                if backend == "cpu": dm_indices = cpu_ijkl(indices, sym+is_K_matrix*8)  
                else:                dm_indices = ipu_ijkl(indices, sym+is_K_matrix*8, N)  
                dm_values = jnp.take(dm, dm_indices, axis=0) 

                print('indices.shape', indices.shape)
                print('dm_values.shape', dm_values.shape)
                print('eris.shape', eris.shape)
                dm_values = dm_values.at[:].mul( eris ) # this is prod, but re-use variable for inplace update. 
                
                if backend == "cpu": ss_indices = cpu_ijkl(indices, sym+8+is_K_matrix*8) 
                else:                ss_indices = ipu_ijkl(indices, sym+8+is_K_matrix*8, N) 
                diff_JK   = diff_JK + jax.ops.segment_sum(dm_values, ss_indices, N**2) * (-HYB_B3LYP/2)**is_K_matrix 
                
                return diff_JK
           
            diff_JK = jax.lax.fori_loop(0, 16, foreach_symmetry, diff_JK) 
            
            return (diff_JK, nonzero_indices, ao_loc)
        
        diff_JK, _, _ = jax.lax.fori_loop(0, nbatch, foreach_batch, (diff_JK, nonzero_indices, ao_loc)) 

        total_diff_JK += diff_JK

    # return total_diff_JK

    jk_psum = jax.lax.psum(total_diff_JK, axis_name="p")

    # cycles = -1
    # if backend == "ipu": 
    #     jk_psum, stop = utils.get_ipu_cycles(jk_psum)
    #     cycles = (stop.array-start.array)[0,0]

    # return jk_psum, cycles

    return jk_psum, 0

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
    parser.add_argument('-basis', default="6-311G", type=str)
    parser.add_argument('-fast_shells', action="store_true")
    
    args = parser.parse_args()
    backend = args.backend 

    natm = int(args.natm) 
    nipu = int(args.nipu)
    if backend == "cpu": nipu = 1

    start = time.time()

    # mol = pyscf.gto.Mole(atom="".join(f"C 0 {1.54*j} {1.54*i};" for i in range(natm) for j in range(natm)), basis=args.basis) 
    mol = pyscf.gto.Mole(atom=process_mol_str('bench10'), basis=args.basis)
    #mol = pyscf.gto.Mole(atom="".join(f"C 0 {1.54*j} {1.54*i};" for i in range(natm) for j in range(natm))) # sto-3g by default
    # mol = pyscf.gto.Mole(atom="".join(f"C 0 {1.54*j} {1.54*i};" for i in range(1) for j in range(2)), basis="sto3g") 
    #mol = pyscf.gto.Mole(atom="".join(f"C 0 {1.54*j} {1.54*i};" for i in range(natm) for j in range(natm)), basis="sto3g") 
    # mol = pyscf.gto.Mole(atom="".join(f"C 0 {1.54*j} {1.54*i};" for i in range(1) for j in range(2)), basis=args.basis) 
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

    dm                   = dm.reshape(-1)
    diff_JK              = np.zeros(dm.shape)

    # ------------------------------------ #

    input_ijkl, sizes, counts, shapes = gen_shells(mol, args.itol, args.nipu, fast_shells=args.fast_shells)
    sliced_input_ijkl = np.concatenate([np.array(ijkl, dtype=int).reshape(args.nipu, -1) for ijkl in input_ijkl], axis=-1)

    diff_JK, cycles = jax.pmap(compute_diff_jk, in_axes=(0, None, None, None, None, None, None), static_broadcasted_argnums=(2, 3, 4, 5, 6), backend=backend, axis_name="p")(sliced_input_ijkl, dm, mol, args.batches, args.itol, args.nipu, args.backend) 
    if backend == "ipu": print("Cycle Count: ", cycles/10**6, "[M]")

    # ------------------------------------ #

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