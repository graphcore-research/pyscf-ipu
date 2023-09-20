import pyscf
import numpy as np 
import ctypes
import ctypes
import numpy
from pyscf import lib
libcgto = lib.load_library('libcgto')

ANG_OF     = 1
NPRIM_OF   = 2
NCTR_OF    = 3
KAPPA_OF   = 4
PTR_EXP    = 5
PTR_COEFF  = 6
BAS_SLOTS  = 8
NGRIDS     = 11
PTR_GRIDS  = 12


def make_loc(bas, key):
    if 'cart' in key:
        l = bas[:,ANG_OF]
        dims = (l+1)*(l+2)//2 * bas[:,NCTR_OF]
    elif 'sph' in key:
        dims = (bas[:,ANG_OF]*2+1) * bas[:,NCTR_OF]
    else:  # spinor
        l = bas[:,ANG_OF]
        k = bas[:,KAPPA_OF]
        dims = (l*4+2) * bas[:,NCTR_OF]
        dims[k<0] = (l[k<0] * 2 + 2) * bas[k<0,NCTR_OF]
        dims[k>0] = (l[k>0] * 2    ) * bas[k>0,NCTR_OF]

    ao_loc = numpy.empty(len(dims)+1, dtype=numpy.int32)
    ao_loc[0] = 0
    dims.cumsum(dtype=numpy.int32, out=ao_loc[1:])
    return ao_loc

def getints4c(intor_name, atm, bas, env, N, shls_slice=None, comp=1,
              aosym='s1', ao_loc=None, cintopt=None, out=None):
    ao_loc = make_loc(bas, intor_name)

    nbas = bas.shape[0]
    shls_slice = (0, nbas, 0, nbas, 0, nbas, 0, nbas)
    shape = [comp, N, N, N, N] 

    out = numpy.ndarray(shape, buffer=out)

    prescreen = lib.c_null_ptr()
    cintopt = lib.c_null_ptr()
    libcgto.GTOnr2e_fill_drv(libcgto.int2e_ip1_sph, libcgto.GTOnr2e_fill_s1, prescreen,
        out.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(comp),
        (ctypes.c_int*8)(*shls_slice),
        ao_loc.ctypes.data_as(ctypes.c_void_p), 
        cintopt,
        atm.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(atm.shape[0]), 
        bas.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nbas), 
        env.ctypes.data_as(ctypes.c_void_p))
    return out

mol = pyscf.gto.Mole(atom="C 0 0 0; C 0 0 1;", basis="sto3g")
mol.build()

truth = mol.intor("int2e_ip1")
us = getints4c("int2e_ip1_sph", mol._atm, mol._bas, mol._env, mol.nao_nr(), None, 3, "s1", None, None, None)

print(np.max(np.abs(truth-us)))
assert np.allclose(truth, us )
print("PASSED")

