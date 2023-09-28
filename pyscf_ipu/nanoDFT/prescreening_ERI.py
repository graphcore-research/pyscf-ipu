import time
import pyscf
import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm

class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name,)
        print('Elapsed: %s' % (time.time() - self.tstart))

def plot4D(x, N, name='', norm=None):
    if norm is None:
        norm = mpl.colors.Normalize(vmin=0, vmax=np.max(x))

    fig, axs = plt.subplots(N, N)
    for i in range(N):
        for j in range(N):
            axs[i, j].imshow(x[i,j,:,:], norm=norm, interpolation='none')
            axs[i, j].set_ylabel(f'i={i}')
            axs[i, j].set_xlabel(f'j={j}')
            axs[i, j].axis("off")
    
    for ax in axs.flat:
        ax.label_outer()

    fig.suptitle(name)
    fig.savefig(name+'.png', dpi=300)

def get_i_j(val, xnp=np, dtype=np.uint64):
    i = (xnp.sqrt(1 + 8*val.astype(dtype)) - 1)//2 # no need for floor, integer division acts as floor. 
    j = (((val - i) - (i**2 - val))//2)
    return i, j

def c2ijkl(c):
    ij, kl = get_i_j(c)
    i, j = get_i_j(ij)
    k, l = get_i_j(kl)
    return (int(i), int(j), int(k), int(l))

def reconstruct_ERI(ERI, nonzero_idx, N, sym=True, enhance=False):
    rec_ERI = np.zeros((N, N, N, N))
    for ijkl in nonzero_idx:
        rec_ERI[ijkl[0], ijkl[1], ijkl[2], ijkl[3]] = ERI[ijkl[0], ijkl[1], ijkl[2], ijkl[3]]
        if sym:
            rec_ERI[ijkl[1], ijkl[0], ijkl[2], ijkl[3]] = ERI[ijkl[1], ijkl[0], ijkl[2], ijkl[3]]
            rec_ERI[ijkl[0], ijkl[1], ijkl[3], ijkl[2]] = ERI[ijkl[0], ijkl[1], ijkl[3], ijkl[2]]
            rec_ERI[ijkl[1], ijkl[0], ijkl[3], ijkl[2]] = ERI[ijkl[1], ijkl[0], ijkl[3], ijkl[2]]
            rec_ERI[ijkl[2], ijkl[3], ijkl[0], ijkl[1]] = ERI[ijkl[2], ijkl[3], ijkl[0], ijkl[1]]
            rec_ERI[ijkl[2], ijkl[3], ijkl[1], ijkl[0]] = ERI[ijkl[2], ijkl[3], ijkl[1], ijkl[0]]
            rec_ERI[ijkl[3], ijkl[2], ijkl[0], ijkl[1]] = ERI[ijkl[3], ijkl[2], ijkl[0], ijkl[1]]
            rec_ERI[ijkl[3], ijkl[2], ijkl[1], ijkl[0]] = ERI[ijkl[3], ijkl[2], ijkl[1], ijkl[0]]

        if enhance:
            rec_ERI[ijkl[0], ijkl[1], ijkl[2], ijkl[3]] = np.abs(ERI[ijkl[0], ijkl[1], ijkl[2], ijkl[3]]) + 100
            if sym:
                rec_ERI[ijkl[1], ijkl[0], ijkl[2], ijkl[3]] = np.abs(ERI[ijkl[1], ijkl[0], ijkl[2], ijkl[3]]) + 100
                rec_ERI[ijkl[0], ijkl[1], ijkl[3], ijkl[2]] = np.abs(ERI[ijkl[0], ijkl[1], ijkl[3], ijkl[2]]) + 100
                rec_ERI[ijkl[1], ijkl[0], ijkl[3], ijkl[2]] = np.abs(ERI[ijkl[1], ijkl[0], ijkl[3], ijkl[2]]) + 100
                rec_ERI[ijkl[2], ijkl[3], ijkl[0], ijkl[1]] = np.abs(ERI[ijkl[2], ijkl[3], ijkl[0], ijkl[1]]) + 100
                rec_ERI[ijkl[2], ijkl[3], ijkl[1], ijkl[0]] = np.abs(ERI[ijkl[2], ijkl[3], ijkl[1], ijkl[0]]) + 100
                rec_ERI[ijkl[3], ijkl[2], ijkl[0], ijkl[1]] = np.abs(ERI[ijkl[3], ijkl[2], ijkl[0], ijkl[1]]) + 100
                rec_ERI[ijkl[3], ijkl[2], ijkl[1], ijkl[0]] = np.abs(ERI[ijkl[3], ijkl[2], ijkl[1], ijkl[0]]) + 100
    return rec_ERI

# -------------------------------------------------------------- #

natm = 3
mol = pyscf.gto.Mole(atom="".join(f"C 0 {1.54*j} {1.54*i};" for i in range(natm) for j in range(natm))) 
# mol = pyscf.gto.Mole(atom="".join(f"C 0 {1.54*i} {1.54*i};" for i in range(natm)))
# atype = 'C' if natm % 2 == 0 else 'Mg'
# mol = pyscf.gto.Mole(atom="".join(f"{atype} 0 {1.54*i} {1.54*i};" for i in range(natm))) 

# mol = pyscf.gto.Mole(atom=[["C", (0,0,0)], ["C", (10,2,3)]])
# mol = pyscf.gto.Mole(atom="".join(f"C {10*i} {2*i} {3*i};" for i in range(natm)))

mol.build()
ERI = mol.intor("int2e_sph", aosym="s1")
ERI_s8 = mol.intor("int2e_sph", aosym="s8")
N = mol.nao_nr()

sym_pattern =  np.array([(i+3)%5!=0 for i in range(N)])
sym_pattern_mat = ~(sym_pattern.reshape(N, 1) ^ sym_pattern.reshape(1, N))

norm = mpl.colors.Normalize(vmin=0, vmax=1)

ERI_nonzeros = np.abs(ERI)>0

ERI_patterns = np.zeros((N, N, N, N))

ERI_pattern_errors = np.zeros((N, N, N, N))

ERI_differences = np.zeros((N, N, N, N))

for i in tqdm(range(N)):
    for j in range(N):
        ERI_slice = ERI_nonzeros[i, j, :, :]
        local_sym_pattern_mat = sym_pattern_mat ^ (sym_pattern_mat[i] ^ sym_pattern_mat[j])
        differences = ERI_slice^local_sym_pattern_mat
        ERI_patterns[i, j, :, :] = local_sym_pattern_mat
        ERI_pattern_errors[i, j, :, :] = np.not_equal(differences & local_sym_pattern_mat, differences)
        ERI_differences[i, j, :, :] = ~differences # invert for consistency (yellow -> nonzero)
        assert np.equal(differences & local_sym_pattern_mat, differences).all()
print('PASSED: sym_pattern_mat works on this ERI!')

if False:
    plot4D(ERI_nonzeros, N, 'ERI_dense_nonzeros')
    plot4D(ERI_nonzeros.swapaxes(1,2), N, 'ERI_dense_nonzeros_swap')
    plot4D(ERI_patterns, N, 'ERI_patterns')
    plot4D(ERI_pattern_errors, N, 'ERI_pattern_errors')
    plot4D(ERI_differences, N, 'ERI_differences')
    plot4D(ERI_differences.swapaxes(1,2), N, 'ERI_differences_swap')

# plot4D(np.log(np.abs(ERI)+1), N, 'ERI_dense')

# -------------------------------------------------------------- #

# all_indices_4d = [ ]
# for a in range(N):
#   for b in range(N):
#     for c in range(N):
#       for d in range(N):
#         abcd      = np.abs(ERI[a,b,c,d])
#         sqrt_abab = np.sqrt(np.abs(ERI[a,b,a,b]))
#         sqrt_cdcd = np.sqrt(np.abs(ERI[c,d,c,d]))
#         assert abcd-1e9 <= sqrt_abab*sqrt_cdcd # add 1e-9 atol 
#         all_indices_4d.append((a,b,c,d))

# # find max value
# I_max = 0
# for a in range(N):
#   for b in range(N):
#     abab      = np.abs(ERI[a,b,a,b])
#     if abab > I_max:
#         I_max = abab

# find max value
I_max = 0
tril_idx = np.tril_indices(N)
for a, b in zip(tril_idx[0], tril_idx[1]):
    abab = np.abs(ERI[a,b,a,b])
    if abab > I_max:
        I_max = abab

tolerance = 1e-9

# ERI[np.abs(ERI)<tolerance] = 0 
# true_nonzero_indices = np.nonzero( ERI.reshape(-1) )[0]
# true_nonzero_indices_4d = [np.unravel_index(c, (N, N, N, N)) for c in true_nonzero_indices]
# test_nonzero_indices = [(x[0], x[1], x[2], x[3]) for x in np.vstack(np.nonzero(ERI)).T]
# assert np.equal(true_nonzero_indices_4d, test_nonzero_indices).all()

# ERI_s8[np.abs(ERI_s8)<tolerance] = 0
print('compute s8 nonzeros...')
true_nonzero_indices_s8 = np.nonzero( ERI_s8.reshape(-1) )[0]
# print('convert s8 nonzeros to 4D...')
# true_nonzero_indices_s8_4d = [c2ijkl(c) for c in true_nonzero_indices_s8]

print('--------------------------------')
print('N', N)
print('I_max', I_max)
print('ERI.reshape(-1).shape', ERI.reshape(-1).shape)
print('ERI_s8.shape', ERI_s8.shape)
# print('len(all_indices_4d)', len(all_indices_4d))
print('--------------------------------')

# -------------------------------------------------------------- #
# Strategy 0
if True:
    with Timer('Strategy 0'):
        screened_indices_s8_4d = []
        
        # sample symmetry pattern and do safety check
        if N % 2 == 0:
            nonzero_seed = ERI[N-1, N-1, :N//2, 0] != 0
            nonzero_seed = np.concatenate([nonzero_seed, np.flip(nonzero_seed)])
        else:
            nonzero_seed = ERI[N-1, N-1, :(N+1)//2, 0] != 0
            nonzero_seed = np.concatenate([nonzero_seed, np.flip(nonzero_seed[:-1])])

        if not np.equal(nonzero_seed, ERI[N-1, N-1, :, 0]!=0).all():
            print('# -------------------------------------------------------------- #')
            print('# WARNING: Experimental symmetry pattern sample is inconsistent. #')
            # print('pred', nonzero_seed)
            # print('real', ERI[N-1, N-1, :, 0]!=0)
            print('# -------------------------------------------------------------- #')

        nonzero_seed = sym_pattern.copy()
        print('forcing sym_pattern')

        ERI_considered = np.zeros((N, N, N, N))

        # collect candidate pairs for s8
        considered_indices = []
        tril_idx = np.tril_indices(N)
        for a, b in zip(tril_idx[0], tril_idx[1]):
            abab = np.abs(ERI[a,b,a,b])
            ERI_considered[a, b, a, b] = 1
            if abab*I_max>=tolerance**2:
                considered_indices.append((a, b)) # collect candidate pairs for s8
                ERI_considered[a, b, a, b] = 2

        ERI_considered_full = np.zeros((N, N, N, N))

        # generate s8 indices
        for index, ab in enumerate(considered_indices):
            a, b = ab
            for cd in considered_indices[index:]:
                c, d = cd
                ERI_considered_full[a, b, c, d] = 1
                if ~(nonzero_seed[b] ^ nonzero_seed[a]) ^ (nonzero_seed[d] ^ nonzero_seed[c]):
                    screened_indices_s8_4d.append((a, b, c, d))
        
    if False:
        plot4D(ERI_considered, N, 'ERI_considered')
        plot4D(ERI_considered.swapaxes(1,2), N, 'ERI_considered_swap')
        plot4D(ERI_considered_full, N, 'ERI_considered_full')
        plot4D(ERI_considered_full.swapaxes(1,2), N, 'ERI_considered_full_swap')

    print('len(considered_indices)', len(considered_indices))
    print('len(screened_indices_s8_4d)', len(screened_indices_s8_4d))
    print('len(true_nonzero_indices_s8)', len(true_nonzero_indices_s8))
    print('nonzero_seed', nonzero_seed)
    rec_ERI = reconstruct_ERI(ERI, screened_indices_s8_4d, N)
    absdiff = np.abs(ERI-rec_ERI)
    print('avg error:', np.mean(absdiff))
    print('std error:', np.std(absdiff))
    print('max error:', np.max(absdiff))
    print('tol', tolerance)

    # check_s8 = [(item in screened_indices_s8_4d) for item in true_nonzero_indices_s8_4d]
    # print ('[(item in screened_indices_4d) for item in true_nonzero_indices_s8_4d]', 'PASS' if np.array(check_s8).all() else 'FAIL')

    print('---')

# -------------------------------------------------------------- #
# Strategy 1
with Timer('Strategy 1'):
    screened_indices_s8_4d = []
    
    # sample symmetry pattern and do safety check
    if N % 2 == 0:
        nonzero_seed = ERI[N-1, N-1, :N//2, 0] != 0
        nonzero_seed = np.concatenate([nonzero_seed, np.flip(nonzero_seed)])
    else:
        nonzero_seed = ERI[N-1, N-1, :(N+1)//2, 0] != 0
        nonzero_seed = np.concatenate([nonzero_seed, np.flip(nonzero_seed[:-1])])

    if not np.equal(nonzero_seed, ERI[N-1, N-1, :, 0]!=0).all():
        print('# -------------------------------------------------------------- #')
        print('# WARNING: Experimental symmetry pattern sample is inconsistent. #')
        # print('pred', nonzero_seed)
        # print('real', ERI[N-1, N-1, :, 0]!=0)
        print('# -------------------------------------------------------------- #')

    nonzero_seed = sym_pattern.copy()
    print('forcing sym_pattern')

    ERI_considered = np.zeros((N, N, N, N))

    # collect candidate pairs for s8
    considered_indices = []
    tril_idx = np.tril_indices(N)
    for a, b in zip(tril_idx[0], tril_idx[1]):
        abab = np.abs(ERI[a,b,a,b])
        ERI_considered[a, b, a, b] = 1
        if abab*I_max>=tolerance**2:
            considered_indices.append((a, b)) # collect candidate pairs for s8
            ERI_considered[a, b, a, b] = 2
    considered_indices = set(considered_indices)

    ERI_considered_full = np.zeros((N, N, N, N))
    
    if False:
        plot4D(ERI_considered, N, 'ERI_considered')
        plot4D(ERI_considered.swapaxes(1,2), N, 'ERI_considered_swap')

    t_size = (1, 2, 5, 3)
    for ia in range(0, N, t_size[0]):
        for ja in range(0, N, t_size[1]):
            for ka in range(0, N, t_size[2]):
                for la in range(0, N, t_size[3]):
                    ib = ia + t_size[0]
                    jb = ja + t_size[1]
                    kb = ka + t_size[2]
                    lb = la + t_size[3]

                    found_nonzero = False
                    # check i,j boxes
                    for bi in range(ia, ib):
                        for bj in range(ja, jb):
                            if (bi, bj) in considered_indices: # if ij box is considered
                                # check if kl pairs are considered
                                for bk in range(ka, kb):
                                    # mla = la
                                    if bk>=bi: # apply symmetry - tril fade vertical
                                        mla = la
                                        if bk == bi:
                                            mla = max(bj, la)
                                        for bl in range(mla, lb):
                                            if (bk, bl) in considered_indices:
                                                ERI_considered_full[bi, bj, bk, bl] = 1
                                                # apply grid pattern to find final nonzeros
                                                if ~(nonzero_seed[bi] ^ nonzero_seed[bj]) ^ (nonzero_seed[bk] ^ nonzero_seed[bl]):
                                                    # found_nonzero = True
                                                    screened_indices_s8_4d.append((bi, bj, bk, bl))
                                                    # break
                                    # if found_nonzero: break
                            # if found_nonzero: break
                        # if found_nonzero: break
                    # if not found_nonzero: continue
                    

if False:
    plot4D(ERI_considered_full, N, 'ERI_considered_full_v1')

print('len(considered_indices)', len(considered_indices))
print('len(screened_indices_s8_4d)', len(screened_indices_s8_4d))
print('len(true_nonzero_indices_s8)', len(true_nonzero_indices_s8))
print('nonzero_seed', nonzero_seed)
rec_ERI = reconstruct_ERI(ERI, screened_indices_s8_4d, N)
absdiff = np.abs(ERI-rec_ERI)
print('avg error:', np.mean(absdiff))
print('std error:', np.std(absdiff))
print('max error:', np.max(absdiff))
print('tol', tolerance)

# check_s8 = [(item in screened_indices_s8_4d) for item in true_nonzero_indices_s8_4d]
# print ('[(item in screened_indices_4d) for item in true_nonzero_indices_s8_4d]', 'PASS' if np.array(check_s8).all() else 'FAIL')

print('---')

exit()
# -------------------------------------------------------------- #
# Strategy 1
if False:
    with Timer('Strategy 1'):
        screened_indices_4d = []

        # collect candidate pairs for s1
        considered_indices = []
        for a in range(N):
            for b in range(N):
                abab = np.abs(ERI[a,b,a,b])
                if abab*I_max>=tolerance**2:
                    considered_indices.append((a, b))

        # generate s1 indices
        for ab in considered_indices:
            a, b = ab
            for cd in considered_indices:
                c, d = cd
                screened_indices_4d.append((a, b, c, d))

    print('len(considered_indices)', len(considered_indices))
    print('len(screened_indices_4d)', len(screened_indices_4d))
    print('len(true_nonzero_indices_4d)', len(true_nonzero_indices_4d))
    rec_ERI = reconstruct_ERI(ERI, screened_indices_4d, N)
    absdiff = np.abs(ERI-rec_ERI)
    print('avg error:', np.mean(absdiff))
    print('avg error:', np.std(absdiff))
    print('max error:', np.max(absdiff))
    print('tol', tolerance)

    # check_s1 = [(item in screened_indices_4d) for item in true_nonzero_indices_4d]
    # print ('[(item in screened_indices_4d) for item in true_nonzero_indices_4d]', 'PASS' if np.array(check_s1).all() else 'FAIL')

    print('---')

# -------------------------------------------------------------- #
if False:
    # Strategy 2
    with Timer('Strategy 2'):
        screened_indices_s8_4d = []

        # collect candidate pairs for s8
        considered_indices = []
        for a in range(N):
            for b in range(a, N):
                abab = np.abs(ERI[a,b,a,b])
                if abab*I_max>=tolerance**2:
                    considered_indices.append((a, b)) # collect candidate pairs for s8

        # generate s8 indices
        for ab in considered_indices:
            a, b = ab
            for cd in considered_indices:
                c, d = cd
                # if b<=d:
                screened_indices_s8_4d.append((d, c, b, a))

    print('len(considered_indices)', len(considered_indices))
    print('len(screened_indices_s8_4d)', len(screened_indices_s8_4d))
    print('len(true_nonzero_indices_s8_4d)', len(true_nonzero_indices_s8_4d))
    rec_ERI = reconstruct_ERI(ERI, screened_indices_s8_4d, N, sym=False, enhance=1)
    absdiff = np.abs(ERI-rec_ERI)
    print('avg error:', np.mean(absdiff))
    print('std error:', np.std(absdiff))
    print('max error:', np.max(absdiff))
    # plot4D(np.abs(rec_ERI), N, 'ERI_strat2')
    unneeded_indices_s8_4d = [idx for idx in screened_indices_s8_4d if idx not in true_nonzero_indices_4d]
    # print('screened_indices_s8_4d', screened_indices_s8_4d)
    # print('true_nonzero_indices_4d', true_nonzero_indices_4d)
    # print('unneeded_indices_s8_4d', unneeded_indices_s8_4d)
    unn_ERI = reconstruct_ERI(ERI, unneeded_indices_s8_4d, N, sym=False, enhance=1)
    print('max unn:', np.max(unn_ERI))
    # plot4D(np.abs(unn_ERI), N, 'ERI_strat2_unneeded')

    tru_ERI = reconstruct_ERI(ERI, true_nonzero_indices_s8_4d, N, sym=False, enhance=1)
    print('max tru:', np.max(tru_ERI))
    plot4D(np.abs(tru_ERI), N, 'ERI_strat2_true_dbg')
    plot4D(np.abs(tru_ERI.swapaxes(1,2)), N, 'ERI_strat2_true_dbgswap')

    # check_s8 = [(item in screened_indices_s8_4d) for item in true_nonzero_indices_s8_4d]
    # print ('[(item in screened_indices_4d) for item in true_nonzero_indices_s8_4d]', 'PASS' if np.array(check_s8).all() else 'FAIL')

    print('---')

# -------------------------------------------------------------- #
# Strategy 3
with Timer('Strategy 3'):
    screened_indices_s8_4d = []

    # collect candidate pairs for s8
    considered_indices = []
    tril_idx = np.tril_indices(N)
    for a, b in zip(tril_idx[0], tril_idx[1]):
        abab = np.abs(ERI[a,b,a,b])
        if abab*I_max>=tolerance**2:
            considered_indices.append((a, b)) # collect candidate pairs for s8

    # generate s8 indices
    for ab in considered_indices:
        a, b = ab
        for cd in considered_indices:
            c, d = cd
            if a>=c:
                screened_indices_s8_4d.append((a, b, c, d))

print('len(considered_indices)', len(considered_indices))
print('len(screened_indices_s8_4d)', len(screened_indices_s8_4d))
print('len(true_nonzero_indices_s8_4d)', len(true_nonzero_indices_s8_4d))
rec_ERI = reconstruct_ERI(ERI, screened_indices_s8_4d, N)
absdiff = np.abs(ERI-rec_ERI)
print('avg error:', np.mean(absdiff))
print('avg error:', np.std(absdiff))
print('max error:', np.max(absdiff))
print('tol', tolerance)

# check_s8 = [(item in screened_indices_s8_4d) for item in true_nonzero_indices_s8_4d]
# print ('[(item in screened_indices_4d) for item in true_nonzero_indices_s8_4d]', 'PASS' if np.array(check_s8).all() else 'FAIL')

print('---')

# -------------------------------------------------------------- #
# Strategy 4
with Timer('Strategy 4'):
    screened_indices_s8_4d = []

    # collect candidate pairs for s8
    considered_indices = []
    for a in range(N):
        for b in range(a, N):
            abab = np.abs(ERI[a,b,a,b])
            if abab*I_max>=tolerance**2:
                considered_indices.append((a, b)) # collect candidate pairs for s8

    # generate s8 indices
    for ab in considered_indices:
        a, b = ab
        for cd in considered_indices:
            c, d = cd
            if b<=d:
                screened_indices_s8_4d.append((d, c, b, a))
screened_indices_s8_4d = set(screened_indices_s8_4d)
for i, j, k, l in tqdm(screened_indices_s8_4d):
    if k != l: assert (i,j,l,k) not in screened_indices_s8_4d, (i,j,k,l)
    if i != j: assert (j,i,k,l) not in screened_indices_s8_4d, (i,j,k,l)
    if set([i, j, k, l]) == 4: assert (j,i,l,k) not in screened_indices_s8_4d, (i,j,k,l)

    if set([i, j, k, l]) == 4: assert (k,l,i,j) not in screened_indices_s8_4d, (i,j,k,l)
    if set([i, j, k, l]) == 4: assert (l,k,i,j) not in screened_indices_s8_4d, (i,j,k,l)
    if set([i, j, k, l]) == 4: assert (k,l,j,i) not in screened_indices_s8_4d, (i,j,k,l)
    if set([i, j, k, l]) == 4: assert (l,k,j,i) not in screened_indices_s8_4d, (i,j,k,l)

print('len(considered_indices)', len(considered_indices))
print('len(screened_indices_s8_4d)', len(screened_indices_s8_4d))
print('len(true_nonzero_indices_s8_4d)', len(true_nonzero_indices_s8_4d))
rec_ERI = reconstruct_ERI(ERI, screened_indices_s8_4d, N)
absdiff = np.abs(ERI-rec_ERI)
print('avg error:', np.mean(absdiff))
print('std error:', np.std(absdiff))
print('max error:', np.max(absdiff))
print('tol', tolerance)

# check_s8 = [(item in screened_indices_s8_4d) for item in true_nonzero_indices_s8_4d]
# print ('[(item in screened_indices_4d) for item in true_nonzero_indices_s8_4d]', 'PASS' if np.array(check_s8).all() else 'FAIL')

print('---')

# -------------------------------------------------------------- #
# Strategy 5
with Timer('Strategy 5'):
    screened_indices_s8_4d = []
    
    # sample symmetry pattern and do safety check
    if N % 2 == 0:
        nonzero_seed = ERI[N-1, N-1, :N//2, 0] != 0
        nonzero_seed = np.concatenate([nonzero_seed, np.flip(nonzero_seed)])
    else:
        nonzero_seed = ERI[N-1, N-1, :(N+1)//2, 0] != 0
        nonzero_seed = np.concatenate([nonzero_seed, np.flip(nonzero_seed[:-1])])
    if not np.equal(nonzero_seed, ERI[N-1, N-1, :, 0]!=0).all():
        print('# -------------------------------------------------------------- #')
        print('# WARNING: Experimental symmetry pattern sample is inconsistent. #')
        # print('pred', nonzero_seed)
        # print('real', ERI[N-1, N-1, :, 0]!=0)
        print('# -------------------------------------------------------------- #')

    # print('test:')
    # for k in range(N):
    #     for l in range(k+1):
    #         is_nonzero = ~(nonzero_seed[k] ^ nonzero_seed[l]) # not XOR
    #         print(is_nonzero, end=' ')
    #     print()
    # exit()


    # collect candidate pairs for s8
    considered_indices = []
    for a in range(N):
        for b in range(a, N):
            abab = np.abs(ERI[a,b,a,b])
            if abab*I_max>=tolerance**2:
                considered_indices.append((a, b)) # collect candidate pairs for s8

    # generate s8 indices
    for index, ab in enumerate(considered_indices):
        a, b = ab
        for cd in considered_indices[index:]:
            c, d = cd
            if ~(nonzero_seed[b] ^ nonzero_seed[a]) ^ (nonzero_seed[d] ^ nonzero_seed[c]):
                screened_indices_s8_4d.append((d, c, b, a))

print('len(considered_indices)', len(considered_indices))
print('len(screened_indices_s8_4d)', len(screened_indices_s8_4d))
print('len(true_nonzero_indices_s8_4d)', len(true_nonzero_indices_s8_4d))
print('nonzero_seed', nonzero_seed)
rec_ERI = reconstruct_ERI(ERI, screened_indices_s8_4d, N)
absdiff = np.abs(ERI-rec_ERI)
print('avg error:', np.mean(absdiff))
print('std error:', np.std(absdiff))
print('max error:', np.max(absdiff))
print('tol', tolerance)

# check_s8 = [(item in screened_indices_s8_4d) for item in true_nonzero_indices_s8_4d]
# print ('[(item in screened_indices_4d) for item in true_nonzero_indices_s8_4d]', 'PASS' if np.array(check_s8).all() else 'FAIL')

print('---')



# -------------------------------------------------------------- #

print('--------------------------------')