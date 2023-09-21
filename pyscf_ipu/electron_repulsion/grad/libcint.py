import pyscf
import numpy as np 
import ctypes
import ctypes
import numpy
from pyscf import lib
libcgto = numpy.ctypeslib.load_library("libcint.so", "") 

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

mol = pyscf.gto.Mole(atom="C 0 0 0; C 0 0 1;", basis="6-31G*")
mol.build()
def getints2c(intor_name, N, atm, bas, env, shls_slice=None, comp=1, hermi=0,
              ao_loc=None, cintopt=None, out=None):
    natm = atm.shape[0]
    nbas = bas.shape[0]
    shls_slice = (0, nbas, 0, nbas)
    ao_loc = make_loc(bas, intor_name)

    shape = (N, N, comp)
    prefix = 'GTO'

    dtype = numpy.double
    drv_name = prefix + 'int2c'

    mat = numpy.ndarray(shape, dtype, out, order='F')
    cintopt = None 

    fn = getattr(libcgto, drv_name)
    fn(getattr(libcgto, intor_name), mat.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(comp), ctypes.c_int(hermi),
        (ctypes.c_int*4)(*(shls_slice[:4])),
        ao_loc.ctypes.data_as(ctypes.c_void_p), cintopt,
        atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(natm),
        bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nbas),
        env.ctypes.data_as(ctypes.c_void_p))

    mat = numpy.rollaxis(mat, -1, 0)
    if comp == 1:
        mat = mat[0]
    return mat

def intor1e(self, intor, N, comp=None, hermi=0, aosym='s1', out=None, shls_slice=None, grids=None):
    return getints2c(intor+"_sph", N, self._atm, self._bas, self._env, shls_slice, comp, hermi, None, None, out)

N = mol.nao_nr()

print("one electron forward pass")
truth      = mol.intor_symmetric('int1e_kin')              # (N,N)
us         = intor1e(mol,'int1e_kin', N, 1)              # (N,N)
print(np.max(np.abs(truth-us)))
assert np.allclose(truth, us)
truth      = mol.intor_symmetric('int1e_nuc')              # (N,N)
us         = intor1e(mol,'int1e_nuc', N, 1)              # (N,N)
print(np.max(np.abs(truth-us)))
assert np.allclose(truth, us)
truth      = mol.intor_symmetric('int1e_ovlp')             # (N,N)
us         = intor1e(mol, 'int1e_ovlp', N, 1)             # (N,N)
print(np.max(np.abs(truth-us)))
assert np.allclose(truth, us)

print("one electron backward ")
truth = - mol.intor('int1e_ipovlp', comp=3)
us = -intor1e(mol,'int1e_ipovlp', N, comp=3)
print(np.max(np.abs(truth-us)))
assert np.allclose(truth, us)
truth = - mol.intor('int1e_ipkin', comp=3)
us = - intor1e(mol, 'int1e_ipkin', N, comp=3)
print(np.max(np.abs(truth-us)))
assert np.allclose(truth, us)

truth = - mol.intor('int1e_ipnuc',  comp=3)
us = - intor1e(mol,'int1e_ipnuc', N,  comp=3)
print(np.max(np.abs(truth-us)))
assert np.allclose(truth, us)

#mol.intor('int1e_iprinv', comp=3)
truth      = mol.intor('int1e_iprinv')             
us         = intor1e(mol, "int1e_iprinv", N, 3)
print(np.max(np.abs(truth-us)))
assert np.allclose(truth, us)


def getints4c(intor_name, atm, bas, env, N, shls_slice=None, comp=1,
              aosym='s1', ao_loc=None, cintopt=None, out=None):
    c_atm = atm.ctypes.data_as(ctypes.c_void_p)
    c_bas = bas.ctypes.data_as(ctypes.c_void_p)
    c_env = env.ctypes.data_as(ctypes.c_void_p)
    natm = atm.shape[0]
    nbas = bas.shape[0]
    ao_loc = make_loc(bas, intor_name)

    shls_slice = (0, nbas, 0, nbas, 0, nbas, 0, nbas)
    
    shape = [comp, N, N, N, N] 
    
    drv = libcgto.GTOnr2e_fill_drv
    fill = getattr(libcgto, 'GTOnr2e_fill_'+aosym)
    out = numpy.ndarray(shape, buffer=out)

    cintopt = None 
    prescreen = lib.c_null_ptr()
    drv(getattr(libcgto, intor_name), fill, prescreen,
        out.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(comp),
        (ctypes.c_int*8)(*shls_slice),
        ao_loc.ctypes.data_as(ctypes.c_void_p), cintopt,
        c_atm, ctypes.c_int(natm), c_bas, ctypes.c_int(nbas), c_env)

    if comp == 1:
        out = out[0]
    return out

def intor(self, intor, N, comp=None, hermi=0, aosym='s1', out=None, shls_slice=None, grids=None):
    return getints4c(intor, self._atm, self._bas, self._env, N, None, comp, "s1", None, None, None)

truth = mol.intor("int2e_sph")
us = intor(mol, "int2e_sph", N, 1)
print(truth.shape, us.shape)

print(np.max(np.abs(truth-us)))
assert np.allclose(truth, us )

truth = mol.intor("int2e_ip1")
us = intor(mol, "int2e_ip1_sph", N, 3)
print(np.max(np.abs(truth-us)))
assert np.allclose(truth, us )


print("PASSED")






exit()

input()

from functools import partial
import os.path as osp
import jax 
import jax.numpy as jnp 
@partial(jax.jit, backend="ipu")
def grad(a):
    from tessellate_ipu import create_ipu_tile_primitive, ipu_cycle_count, tile_map, tile_put_sharded, tile_put_replicated
    vertex_filename  = osp.join(osp.dirname(__file__), "grad.cpp")
    grad = create_ipu_tile_primitive(
            "Grad" ,
            "Grad" ,
            inputs=["n"], 
            outputs={"out": 0},
            gp_filename=vertex_filename,
            perf_estimate=100,
    )
    a= tile_put_replicated(jnp.array(a, dtype=jnp.float32),   (1,3,7)) 

    value = tile_map(grad, a)

    return value.array

print(grad(123.7))