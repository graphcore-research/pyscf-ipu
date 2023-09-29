# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import numpy as np 

def inverse_permutation(a):
    b = np.arange(a.shape[0])
    b[a] = b.copy()
    return b

def get_shapes(input_ijkl, bas):
    i_sh, j_sh, k_sh, l_sh  = input_ijkl[0]
    BAS_SLOTS = 8
    NPRIM_OF = 2
    NCTR_OF = 3
    ANG_OF = 1
    GSHIFT = 4

    i_prim  = bas.reshape(-1)[BAS_SLOTS*i_sh + NPRIM_OF]
    j_prim  = bas.reshape(-1)[BAS_SLOTS*j_sh + NPRIM_OF]
    k_prim  = bas.reshape(-1)[BAS_SLOTS*k_sh + NPRIM_OF]
    l_prim  = bas.reshape(-1)[BAS_SLOTS*l_sh + NPRIM_OF]

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
    nf = nfi * nfk * nfl * nfj;
    n_comp = 1

    nc = i_ctr * j_ctr * k_ctr * l_ctr;
    lenl = nf * nc * n_comp;
    lenk = nf * i_ctr * j_ctr * k_ctr * n_comp;
    lenj = nf * i_ctr * j_ctr * n_comp;
    leni = nf * i_ctr * n_comp;
    len0 = nf * n_comp;

    ng = [0, 0, 0, 0, 0, 1, 1, 1];

    IINC=0
    JINC=1
    KINC=2
    LINC=3

    li_ceil = i_l + ng[IINC]
    lj_ceil = j_l + ng[JINC]
    lk_ceil = k_l + ng[KINC]
    ll_ceil = l_l + ng[LINC]
    nrys_roots = (li_ceil + lj_ceil + lk_ceil + ll_ceil)/2 + 1


    ibase = li_ceil > lj_ceil;
    kbase = lk_ceil > ll_ceil;
    if (nrys_roots <= 2):
        ibase = 0;
        kbase = 0;
    if (kbase) :
        dlk = lk_ceil + ll_ceil + 1;
        dll = ll_ceil + 1;
    else:
        dlk = lk_ceil + 1;
        dll = lk_ceil + ll_ceil + 1;

    if (ibase) :
        dli = li_ceil + lj_ceil + 1;
        dlj = lj_ceil + 1;
    else :
        dli = li_ceil + 1;
        dlj = li_ceil + lj_ceil + 1;

    g_stride_i = nrys_roots;
    g_stride_k = nrys_roots * dli;
    g_stride_l = nrys_roots * dli * dlk;
    g_stride_j = nrys_roots * dli * dlk * dll;
    g_size     = nrys_roots * dli * dlk * dll * dlj;
    gbits        = ng[GSHIFT];
    leng = g_size*3*((1<<gbits)+1);

    len = leng + lenl + lenk + lenj + leni + len0;

    di = i_l * 2 + 1;
    dj = j_l * 2 + 1;
    dk = k_l * 2 + 1;
    dl = l_l * 2 + 1;

    ni = (i_l*2+1) * i_ctr;
    nj = (j_l*2+1) * j_ctr;
    nk = (k_l*2+1) * k_ctr;
    nl = (l_l*2+1) * l_ctr;
    nfik = nfi * nfk;
    nfikl = nfik * nfl;
    dlj = dl * dj;
    ofj = ni * dj;

    ofk = ni * nj * dk;
    ofl = ni * nj * nk * dl;
    buflen = nfikl*dj;

    return len, nf, buflen

def prepare_integrals_2_inputs(mol, itol):
    # Shapes/sizes.
    atm, bas, env   = mol._atm, mol._bas, mol._env
    n_atm, n_bas, N = atm.shape[0], bas.shape[0], mol.nao_nr()
    ao_loc          = np.cumsum(np.concatenate([np.zeros(1), (bas[:,1]*2+1) * bas[:,3] ])).astype(np.int32)
    n_ao_loc        = np.prod(ao_loc.shape)
    shls_slice      = (0, n_bas, 0, n_bas, 0, n_bas, 0, n_bas)
    shape           = [1, N, N, N, N]

    # Initialize tensors for CPU libcint computation.
    buf     = np.zeros(np.prod(shape)*2)
    out     = np.zeros(shape)
    eri     = np.zeros(shape).reshape(-1)
    ipu_eri = np.zeros(shape).reshape(-1)

    dtype = np.float32 #hardcoded
    buf, out, eri, ipu_eri, env = buf.astype(dtype), out.astype(dtype), eri.astype(dtype), ipu_eri.astype(dtype), env.astype(dtype)

    # The padded shape used to store output from all tiles.
    n_buf, n_eri, n_env = 81, 81, np.prod(env.shape)
    if mol.basis == "6-31G*": # has known error; please open github issue if you want to use 6-31G*
            n_buf = 5**4
            n_eri = 5**4

    # Compute how many distinct integrals after 8x symmetry.
    num_calls = 0
    for i in range(n_bas):
        for j in range(i+1):
            for k in range(i, n_bas):
                for l in range(k+1):
                    # almost all 8-fold symmetry rules (except horizontal fade)
                    num_calls+=1
    print('num_calls', num_calls)

    # Input/outputs for calling the IPU vertex.
    input_ijkl   = np.zeros((num_calls, 4),     dtype=np.int32)
    cpu_output   = np.zeros((num_calls, n_eri), dtype=np.float32)
    output_sizes = np.zeros((num_calls, 5))

    USE_TOLERANCE_THRESHOLD = True
    screened_indices_s8_4d = []

    if USE_TOLERANCE_THRESHOLD:
        tolerance = itol
        print('computing ERI s8 to sample N*(N+1)/2 values... ', end='')
        ERI_s8 = mol.intor('int2e_sph', aosym='s8')
        print('done')

        # sample symmetry pattern and do safety check
        # if N % 2 == 0:
        #     nonzero_seed = ERI[N-1, N-1, :N//2, 0] != 0
        #     nonzero_seed = np.concatenate([nonzero_seed, np.flip(nonzero_seed)])
        # else:
        #     nonzero_seed = ERI[N-1, N-1, :(N+1)//2, 0] != 0
        #     nonzero_seed = np.concatenate([nonzero_seed, np.flip(nonzero_seed[:-1])])
        # if not np.equal(nonzero_seed, ERI[N-1, N-1, :, 0]!=0).all():
        #     print('# -------------------------------------------------------------- #')
        #     print('# WARNING: Experimental symmetry pattern sample is inconsistent. #')
        #     print('pred', nonzero_seed)
        #     print('real', ERI[N-1, N-1, :, 0]!=0)
        #     print('# -------------------------------------------------------------- #')

    # hardcoded symmetry pattern
    sym_pattern =  np.array([(i+3)%5!=0 for i in range(N)])
    nonzero_seed = sym_pattern

    if USE_TOLERANCE_THRESHOLD:
        # find max value
        I_max = 0
        tril_idx = np.tril_indices(N)
        for a, b in zip(tril_idx[0], tril_idx[1]):
            index_ab_s8 = a*(a+1)//2 + b
            index_s8 = index_ab_s8*(index_ab_s8+3)//2
            abab = np.abs(ERI_s8[index_s8])
            if abab > I_max:
                I_max = abab

   # collect candidate pairs for s8
    considered_indices = []
    tril_idx = np.tril_indices(N)
    for a, b in zip(tril_idx[0], tril_idx[1]):
        if USE_TOLERANCE_THRESHOLD:
            index_ab_s8 = a*(a+1)//2 + b
            index_s8 = index_ab_s8*(index_ab_s8+3)//2
            abab = np.abs(ERI_s8[index_s8])
            if abab*I_max>=tolerance**2:
                considered_indices.append((a, b)) # collect candidate pairs for s8
        else:
            considered_indices.append((a, b)) # collect candidate pairs for s8
    considered_indices = set(considered_indices)

    print('n_bas', n_bas)
    print('ao_loc', ao_loc)

    # Fill input_ijkl and output_sizes with the necessary indices.
    n_items = 0
    n_all_integrals = 0
    n_new_integrals = 0
    for i in range(n_bas):
        for j in range(i+1):
            for k in range(i, n_bas):
                for l in range(k+1):
                    di = ao_loc[i+1] - ao_loc[i]
                    dj = ao_loc[j+1] - ao_loc[j]
                    dk = ao_loc[k+1] - ao_loc[k]
                    dl = ao_loc[l+1] - ao_loc[l]

                    ia, ib = ao_loc[i], ao_loc[i+1]
                    ja, jb = ao_loc[j], ao_loc[j+1]
                    ka, kb = ao_loc[k], ao_loc[k+1]
                    la, lb = ao_loc[l], ao_loc[l+1]

                    n_all_integrals += di*dj*dk*dl

                    found_nonzero = False
                    # check i,j boxes
                    for bi in range(ia, ib):
                        for bj in range(ja, jb):
                            if (bi, bj) in considered_indices: # if ij box is considered
                                # check if kl pairs are considered
                                for bk in range(ka, kb):
                                    if bk>=bi: # apply symmetry - tril fade vertical
                                        mla = la
                                        if bk == bi:
                                            mla = max(bj, la) # apply symmetry - tril fade horizontal
                                        for bl in range(mla, lb):
                                            if (bk, bl) in considered_indices:
                                                # apply grid pattern to find final nonzeros
                                                if ~(nonzero_seed[bi] ^ nonzero_seed[bj]) ^ (nonzero_seed[bk] ^ nonzero_seed[bl]):
                                                    found_nonzero = True
                                                    break
                                    if found_nonzero: break
                            if found_nonzero: break
                        if found_nonzero: break
                    if not found_nonzero: continue 

                    n_new_integrals += di*dj*dk*dl

                    input_ijkl[n_items] = [i, j, k, l]

                    output_sizes[n_items] = [di, dj, dk, dl, di*dj*dk*dl]

                    n_items += 1
    print('!!! saved', num_calls - n_items, 'calls i.e.', num_calls, '->', n_items)
    print('!!! saved', n_all_integrals - n_new_integrals, 'integrals i.e.', n_all_integrals, '->', n_new_integrals)

    num_calls = n_items
    input_ijkl = input_ijkl[:num_calls, :]
    cpu_output = cpu_output[:num_calls, :]
    output_sizes = output_sizes[:num_calls, :]

    # Prepare IPU inputs.
    # Merge all int/float inputs in seperate arrays.
    input_floats = env.reshape(1, -1)
    input_ints   = np.zeros((1, 6+n_ao_loc +n_atm*6+n_bas*8), dtype=np.int32)
    start, stop = 0, 6
    input_ints[:, start:stop] = np.array( [n_eri, n_buf, n_atm, n_bas, n_env, n_ao_loc] )
    start, stop = start+6, stop+n_ao_loc
    input_ints[:, start:stop] = ao_loc.reshape(-1)
    start, stop = start+n_ao_loc, stop + n_atm*6
    input_ints[:, start:stop] = atm.reshape(-1)
    start, stop = start+n_atm*6, stop + n_bas*8
    input_ints[:, start:stop] = bas.reshape(-1)

    sizes, counts  = np.unique(output_sizes[:, -1], return_counts=True)
    sizes, counts = sizes.astype(np.int32), counts.astype(np.int32)

    indxs               = np.argsort(output_sizes[:, -1])
    sorted_output_sizes = output_sizes[indxs]
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

    tuple_ijkl = tuple(inputs)
    input_ijkl = inputs

    for i in range(len(sizes)):
        shapes.append(get_shapes(input_ijkl[i], bas))

    return input_floats, input_ints, tuple_ijkl, tuple(shapes), tuple(sizes.tolist()), counts.tolist(), ao_loc, num_calls
