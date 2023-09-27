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
            axs[i, j].imshow(x[i,j], norm=norm)
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
# mol = pyscf.gto.Mole(atom="".join(f"C 0 {0.054*i} {0.54*i};" for i in range(natm)))
# atype = 'C' if natm % 2 == 0 else 'Mg'
# mol = pyscf.gto.Mole(atom="".join(f"{atype} 0 {1.54*i} {1.54*i};" for i in range(natm))) 

# mol = pyscf.gto.Mole(atom=[["C", (0,0,0)], ["C", (10,2,3)]])
# mol = pyscf.gto.Mole(atom="".join(f"C {10*i} {2*i} {3*i};" for i in range(natm)))

mol.build()
ERI = mol.intor("int2e_sph", aosym="s1")
ERI_s8 = mol.intor("int2e_sph", aosym="s8")
N = mol.nao_nr()

# plot4D(np.log(np.abs(ERI)+1), N, 'ERI_dense')

# -------------------------------------------------------------- #

all_indices_4d = [ ]
for a in range(N):
  for b in range(N):
    for c in range(N):
      for d in range(N):
        abcd      = np.abs(ERI[a,b,c,d])
        sqrt_abab = np.sqrt(np.abs(ERI[a,b,a,b]))
        sqrt_cdcd = np.sqrt(np.abs(ERI[c,d,c,d]))
        assert abcd-1e9 <= sqrt_abab*sqrt_cdcd # add 1e-9 atol 
        all_indices_4d.append((a,b,c,d))

# find max value
I_max = 0
for a in range(N):
  for b in range(N):
    abab      = np.abs(ERI[a,b,a,b])
    if abab > I_max:
        I_max = abab

tolerance = 1e-7

# ERI[np.abs(ERI)<tolerance] = 0 
true_nonzero_indices = np.nonzero( ERI.reshape(-1) )[0]
true_nonzero_indices_4d = [np.unravel_index(c, (N, N, N, N)) for c in true_nonzero_indices]
# test_nonzero_indices = [(x[0], x[1], x[2], x[3]) for x in np.vstack(np.nonzero(ERI)).T]
# assert np.equal(true_nonzero_indices_4d, test_nonzero_indices).all()

# ERI_s8[np.abs(ERI_s8)<tolerance] = 0
true_nonzero_indices_s8 = np.nonzero( ERI_s8.reshape(-1) )[0]
true_nonzero_indices_s8_4d = [c2ijkl(c) for c in true_nonzero_indices_s8]

print('--------------------------------')
print('N', N)
print('I_max', I_max)
print('ERI.reshape(-1).shape', ERI.reshape(-1).shape)
print('ERI_s8.shape', ERI_s8.shape)
print('len(all_indices_4d)', len(all_indices_4d))
print('--------------------------------')

# -------------------------------------------------------------- #
# Strategy 1
with Timer('Strategy 1'):
    screened_indices_4d = []

    # collect candidate pairs for s1
    considered_indices = []
    for a in range(N):
        for b in range(N):
            abab = np.abs(ERI[a,b,a,b])
            if abab*I_max>=tolerance:
                considered_indices.append((a, b))
            else:
                print('>>', abab, I_max)

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
                if abab*I_max>=tolerance:
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
        if abab*I_max>=tolerance:
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
            if abab*I_max>=tolerance:
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
    if not np.equal(nonzero_seed, ERI[N-1, N-1, :, 0]).all():
        print('# -------------------------------------------------------------- #')
        print('# WARNING: Experimental symmetry pattern sample is inconsistent. #')
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
            if abab*I_max>=tolerance:
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

# check_s8 = [(item in screened_indices_s8_4d) for item in true_nonzero_indices_s8_4d]
# print ('[(item in screened_indices_4d) for item in true_nonzero_indices_s8_4d]', 'PASS' if np.array(check_s8).all() else 'FAIL')

print('---')

# -------------------------------------------------------------- #

print('--------------------------------')