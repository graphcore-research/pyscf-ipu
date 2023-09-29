# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import os
#os.environ['TF_POPLAR_FLAGS'] = """
#  --executable_cache_path=/tmp/ipu-ef-cache
#  --show_progress_bar=true
#"""
import pyscf
import numpy as np 
import ctypes
import ctypes
import numpy
from pyscf import lib
from icecream import ic 
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

#mol = pyscf.gto.Mole(atom="C 0 0 0; C 0 0 1;", basis="6-31G*")
mol = pyscf.gto.Mole(atom="H 0 0 0; H 0 0 1;", basis="sto3g")
#mol = pyscf.gto.Mole(atom="He 0 0 0; ", basis="STO3G")
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

    #mat = numpy.ndarray(shape, dtype, out, order='F')
    mat = numpy.zeros(shape, dtype=dtype, order="F")#, dtype, out, order='F')
    cintopt = None 

    # type 
    float32 = "#define dtype float" in open("_libcint.c", "r").read()
    if float32: 
        mat = mat.astype(np.float32)
        env = env.astype(np.float32)

    #ic(drv_name, intor_name, comp, hermi)
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


INT1E_KIN = 0
INT1E_NUC = 1
INT1E_OVLP = 2

INT1E_KIN_IP = 3
INT1E_NUC_IP = 4
INT1E_OVLP_IP = 5

from functools import partial
import os.path as osp
import jax 
import jax.numpy as jnp 
@partial(jax.jit, backend="ipu", static_argnums=(0,1,2,3,4,5,6,7,8))
def grad(self, intor, which_integral, N, comp=None, hermi=0, aosym='s1', out=None, shls_slice=None, grids=None):
    from tessellate_ipu import create_ipu_tile_primitive, ipu_cycle_count, tile_map, tile_put_sharded, tile_put_replicated
    vertex_filename  = osp.join(osp.dirname(__file__), "grad.cpp")
    #mat, shls_slice, ao_loc, atm, bas, env
    grad = create_ipu_tile_primitive(
            "Grad" ,
            "Grad" ,
            inputs=["mat", "shls_slice", "ao_loc", "atm", "bas", "env", "natm", "nbas", "which_integral"], 
            outputs={"out": 0},
            gp_filename=vertex_filename,
            perf_estimate=100,
    )
    
    
    intor_name, N, atm, bas, env, shls_slice, comp, hermi, ao_loc, cintopt, out=\
        intor+"_sph", N, self._atm, self._bas, self._env, shls_slice, comp, hermi, None, None, out

    natm = atm.shape[0]
    nbas = bas.shape[0]
    shls_slice = (0, nbas, 0, nbas)
    ao_loc = make_loc(bas, intor_name)

    shape = (N, N, comp)
    prefix = 'GTO'

    dtype = numpy.double
    drv_name = prefix + 'int2c'

    mat = numpy.ndarray(shape, dtype, out, order='F')

    # type 
    float32 = "#define dtype float" in open("_libcint.c", "r").read()
    if float32: 
        mat = mat.astype(np.float32)
        env = env.astype(np.float32)

    if comp == 3:
        mat = np.transpose(np.zeros(shape), (2,0,1))
    else:
        mat = np.zeros(shape)

    mat = tile_put_replicated(np.array(mat, dtype=jnp.float32),   (1,)) 
    shls_slice = tile_put_replicated(np.array(shls_slice[:4], dtype=jnp.int32),   (1,)) 
    ao_loc = tile_put_replicated(np.array(ao_loc, dtype=jnp.int32),   (1,)) 
    atm = tile_put_replicated(np.array(atm, dtype=jnp.int32),   (1,)) 
    bas = tile_put_replicated(np.array(bas, dtype=jnp.int32),   (1,)) 
    env = tile_put_replicated(np.array(env, dtype=jnp.float32),   (1,)) 
    natm = tile_put_replicated(np.array(natm, dtype=jnp.int32),   (1,)) 
    nbas = tile_put_replicated(np.array(nbas, dtype=jnp.int32),   (1,)) 

    which_integral = tile_put_replicated(np.array(which_integral, dtype=jnp.int32),   (1,)) 

    value = tile_map(grad, mat, shls_slice, ao_loc, atm, bas, env, natm, nbas, which_integral)

    result = value.array[0]

    return result 


us    =  intor1e(mol, 'int1e_kin', N, comp=1)
truth =  mol.intor('int1e_kin', comp=1)
ovlp =   np.asarray( grad(mol, "int1e_kin", INT1E_KIN,  N, 1))
print(np.max(np.abs(ovlp.reshape(-1)-truth.reshape(-1))))

us    =  intor1e(mol, 'int1e_ovlp', N, comp=1)
truth =  mol.intor('int1e_ovlp', comp=1)
ovlp =   np.asarray( grad(mol, "int1e_ovlp", INT1E_OVLP,  N, 1))
print(np.max(np.abs(ovlp.reshape(-1)-truth.reshape(-1))))

us    =  intor1e(mol, 'int1e_nuc', N, comp=1)
truth =  mol.intor('int1e_nuc', comp=1)
nuc =   np.asarray( grad(mol, "int1e_nuc", INT1E_NUC,  N, 1))
print(np.max(np.abs(nuc.reshape(-1)-truth.reshape(-1))))

us    = - intor1e(mol, 'int1e_ipkin', N, comp=3)
truth = - mol.intor('int1e_ipkin', comp=3)
ovlp = - np.transpose( np.asarray( grad(mol, "int1e_ipkin", INT1E_KIN_IP,  N, 3)), (0, 2, 1))
print(np.max(np.abs(ovlp.reshape(-1)-truth.reshape(-1))))

us    = - intor1e(mol, 'int1e_ipovlp', N, comp=3)
truth = - mol.intor('int1e_ipovlp', comp=3)
print(truth.shape)
ovlp = - np.transpose( np.asarray( grad(mol, "int1e_ipovlp", INT1E_OVLP_IP,  N, 3)), (0, 2, 1))
print(np.max(np.abs(ovlp.reshape(-1)-truth.reshape(-1))))

us    = - intor1e(mol, 'int1e_ipnuc', N, comp=3)
truth = - mol.intor('int1e_ipnuc', comp=3)
print(truth.shape)
nuc = - np.transpose( np.asarray( grad(mol, "int1e_ipnuc", INT1E_NUC_IP,  N, 3)), (0, 2, 1))
print(np.max(np.abs(nuc.reshape(-1)-truth.reshape(-1))))


'''truth = - mol.intor('int1e_ipovlp', comp=3).reshape(-1)
us = - intor1e(mol, 'int1e_ipovlp', N, comp=3).reshape(-1)
print(truth)
print(us)
ovlp = np.asarray(grad(mol, "int1e_ipovlp",  INT1E_OVLP_IP,  N, 3)).reshape(-1)
print(ovlp)
print(np.max(np.abs(ovlp-truth)))'''

exit()


print("one electron forward pass")
truth      = mol.intor_symmetric('int1e_kin')              # (N,N)
us         = intor1e(mol,'int1e_kin', N, 1)              # (N,N)
print(np.max(np.abs(truth-us)))
print(np.allclose(truth, us))
# got kin and nuc, next overlap, then derivatives. 
# perhaps then split over tiles => make faster. 
kin      = mol.intor_symmetric('int1e_kin').reshape(-1)              # (N,N)
nuc      = mol.intor_symmetric('int1e_nuc').reshape(-1)              # (N,N)
ovlp     = mol.intor_symmetric('int1e_ovlp').reshape(-1)              # (N,N)

kin1  = intor1e(mol, 'int1e_kin', N, 1).reshape(-1)             # (N,N)
nuc1  = intor1e(mol, 'int1e_nuc', N, 1).reshape(-1)             # (N,N)
ovlp1 = intor1e(mol, 'int1e_ovlp', N, 1).reshape(-1)             # (N,N)

kin2 = np.asarray(grad(mol, "int1e_kin",  INT1E_KIN,  N, 1)).reshape(-1)
nuc2 = np.asarray(grad(mol, "int1e_nuc",  INT1E_NUC,  N, 1)).reshape(-1)
ovlp2 = np.asarray(grad(mol, "int1e_ovlp", INT1E_OVLP, N, 1)).reshape(-1)

print(kin)
print(kin1)
print(kin2)

print(nuc)
print(nuc1)
print(nuc2)

print(ovlp)
print(ovlp1)
print(ovlp2)

exit()



truth      = mol.intor_symmetric('int1e_nuc')              # (N,N)
us         = intor1e(mol,'int1e_nuc', N, 1)              # (N,N)
print(np.max(np.abs(truth-us)))
#assert np.allclose(truth, us)
print(np.allclose(truth, us))
truth      = mol.intor_symmetric('int1e_ovlp')             # (N,N)
us         = intor1e(mol, 'int1e_ovlp', N, 1)             # (N,N)
print(np.max(np.abs(truth-us)))
print(np.allclose(truth, us))
#assert np.allclose(truth, us)

print("\none electron backward ")
truth = - mol.intor('int1e_ipovlp', comp=3)
us = -intor1e(mol,'int1e_ipovlp', N, comp=3)
print(np.max(np.abs(truth-us)))
#assert np.allclose(truth, us)
print(np.allclose(truth, us))
truth = - mol.intor('int1e_ipkin', comp=3)
us = - intor1e(mol, 'int1e_ipkin', N, comp=3)
print(np.max(np.abs(truth-us)))
#assert np.allclose(truth, us)
print(np.allclose(truth, us))

truth = - mol.intor('int1e_ipnuc',  comp=3)
us = - intor1e(mol,'int1e_ipnuc', N,  comp=3)
print(np.max(np.abs(truth-us)))
#assert np.allclose(truth, us)
print(np.allclose(truth, us))

#mol.intor('int1e_iprinv', comp=3)
truth      = mol.intor('int1e_iprinv')             
us         = intor1e(mol, "int1e_iprinv", N, 3)
print(np.max(np.abs(truth-us)))
#assert np.allclose(truth, us)
print(np.allclose(truth, us))


def getints4c(intor_name, atm, bas, env, N, shls_slice=None, comp=1,
            aosym='s1', ao_loc=None, cintopt=None, out=None):
    print(intor_name)
    c_atm = atm.ctypes.data_as(ctypes.c_void_p)
    c_bas = bas.ctypes.data_as(ctypes.c_void_p)
    natm = atm.shape[0]
    nbas = bas.shape[0]
    ao_loc = make_loc(bas, intor_name)

    shls_slice = (0, nbas, 0, nbas, 0, nbas, 0, nbas)
    
    shape = [comp, N, N, N, N] 
    
    drv = libcgto.GTOnr2e_fill_drv
    fill = getattr(libcgto, 'GTOnr2e_fill_'+aosym)
    out = numpy.ndarray(shape, buffer=out)

    # type 
    float32 = "#define dtype float" in open("_libcint.c", "r").read()
    if float32: 
        out = out.astype(np.float32)
        env = env.astype(np.float32)

    c_env = env.ctypes.data_as(ctypes.c_void_p)

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

print(np.max(np.abs(truth-us)))
#assert np.allclose(truth, us )
print(np.allclose(truth, us))

truth = mol.intor("int2e_ip1")
us = intor(mol, "int2e_ip1_sph", N, 3)
print(np.max(np.abs(truth-us)))
#assert np.allclose(truth, us )
print(np.allclose(truth, us))

print("PASSED")
