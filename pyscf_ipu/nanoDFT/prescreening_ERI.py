import time
import pyscf
import numpy as np 

class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name,)
        print('Elapsed: %s' % (time.time() - self.tstart))

def get_i_j(val, xnp=np, dtype=np.uint64):
    i = (xnp.sqrt(1 + 8*val.astype(dtype)) - 1)//2 # no need for floor, integer division acts as floor. 
    j = (((val - i) - (i**2 - val))//2)
    return i, j

def c2ijkl(c):
    ij, kl = get_i_j(c)
    i, j = get_i_j(ij)
    k, l = get_i_j(kl)
    return (int(i), int(j), int(k), int(l))

# -------------------------------------------------------------- #

natm = 5
# mol = pyscf.gto.Mole(atom="".join(f"C 0 {1.54*j} {1.54*i};" for i in range(natm) for j in range(natm))) 

# mol = pyscf.gto.Mole(atom=[["C", (0,0,0)], ["C", (10,2,3)]])
mol = pyscf.gto.Mole(atom="".join(f"C {10*i} {2*i} {3*i};" for i in range(natm)))

mol.build()
ERI = mol.intor("int2e_sph", aosym="s1")
ERI_s8 = mol.intor("int2e_sph", aosym="s8")
N = mol.nao_nr()

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

ERI[np.abs(ERI)<tolerance] = 0 
true_nonzero_indices = np.nonzero( ERI.reshape(-1) )[0]
true_nonzero_indices_4d = [np.unravel_index(c, (N, N, N, N)) for c in true_nonzero_indices]

ERI_s8[np.abs(ERI_s8)<tolerance] = 0
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

    # generate s1 indices
    for ab in considered_indices:
        a, b = ab
        for cd in considered_indices:
            c, d = cd
            screened_indices_4d.append((a, b, c, d))

print('len(considered_indices)', len(considered_indices))
print('len(screened_indices_4d)', len(screened_indices_4d))
print('len(true_nonzero_indices_4d)', len(true_nonzero_indices_4d))

check_s1 = [(item in screened_indices_4d) for item in true_nonzero_indices_4d]
assert np.array(check_s1).all()
print('PASSED [(item in screened_indices_4d) for item in true_nonzero_indices_4d]!')

print('---')

# -------------------------------------------------------------- #
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
            if b<=d:
                screened_indices_s8_4d.append((d, c, b, a))

print('len(considered_indices)', len(considered_indices))
print('len(screened_indices_s8_4d)', len(screened_indices_s8_4d))
print('len(true_nonzero_indices_s8_4d)', len(true_nonzero_indices_s8_4d))

check_s8 = [(item in screened_indices_s8_4d) for item in true_nonzero_indices_s8_4d]
assert np.array(check_s8).all()
print('PASSED [(item in screened_indices_4d) for item in true_nonzero_indices_s8_4d]!')

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

check_s8 = [(item in screened_indices_s8_4d) for item in true_nonzero_indices_s8_4d]
assert np.array(check_s8).all()
print('PASSED [(item in screened_indices_4d) for item in true_nonzero_indices_s8_4d]!')

print('---')

# -------------------------------------------------------------- #

print('--------------------------------')