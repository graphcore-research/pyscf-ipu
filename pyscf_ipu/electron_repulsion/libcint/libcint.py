# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
# [x] Refactor into {def int2e_sph, def int1e_nuc, ...}. 
# [ ] Add tile-mapping of integral computation. 
# [ ] Consider how to interface this code into nanoDFT. 
# [ ] Remove hard-coding of tensor (i.e. move shape computation to python/jax.trace). 
import os
import pyscf
import numpy as np 
import ctypes
import ctypes
import numpy
from pyscf import lib
from icecream import ic 
from functools import partial
import os.path as osp
import jax 
import jax.numpy as jnp 

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

INT1E_KIN     = 0
INT1E_NUC     = 1
INT1E_OVLP    = 2
INT1E_KIN_IP  = 3
INT1E_NUC_IP  = 4
INT1E_OVLP_IP = 5
INT2E_SPH     = 6
INT2E_IP1_SPH = 7

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
    float32 = "#define dtype float" in open("libcint.c", "r").read()
    if float32: 
        mat = mat.astype(np.float32)
        env = env.astype(np.float32)

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

def cpu_intor1e(self, intor, N, comp=None, hermi=0, aosym='s1', out=None, shls_slice=None, grids=None):
    return getints2c(intor+"_sph", N, self._atm, self._bas, self._env, shls_slice, comp, hermi, None, None, out)

@partial(jax.jit, backend="ipu", static_argnums=(0,1,2,3,4,5,6,7,8))
def ipu_intor1e(self, intor, which_integral, N, comp=None, hermi=0, aosym='s1', out=None, shls_slice=None, grids=None):
    from tessellate_ipu import create_ipu_tile_primitive, ipu_cycle_count, tile_map, tile_put_sharded, tile_put_replicated
    vertex_filename  = osp.join(osp.dirname(__file__), "libcint.cpp")
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

    dtype = numpy.double

    mat = numpy.ndarray(shape, dtype, out, order='F')

    # type 
    float32 = "#define dtype float" in open("libcint.c", "r").read()
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

def cpu_getints4c(intor_name, atm, bas, env, N, shls_slice=None, comp=1,
            aosym='s1', ao_loc=None, cintopt=None, out=None, which_integral=-1):
    c_atm = atm.ctypes.data_as(ctypes.c_void_p)
    c_bas = bas.ctypes.data_as(ctypes.c_void_p)
    natm = atm.shape[0]
    nbas = bas.shape[0]
    ao_loc = make_loc(bas, intor_name)

    shls_slice = (0, nbas, 0, nbas, 0, nbas, 0, nbas)
    
    shape = [comp, N, N, N, N] 
    
    drv = libcgto.GTOnr2e_fill_drv
    fill = getattr(libcgto, 'GTOnr2e_fill_'+aosym)
    #out = numpy.ndarray(shape, buffer=out)
    out = numpy.zeros(shape)

    # type 
    float32 = "#define dtype float" in open("libcint.c", "r").read()
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
        c_atm, ctypes.c_int(natm), c_bas, ctypes.c_int(nbas), c_env, which_integral)

    if comp == 1:
        out = out[0]
    return out

def cpu_intor2e(self, intor, N, comp=None, hermi=0, aosym='s1', out=None, shls_slice=None, grids=None, which_integral=-1):
    return cpu_getints4c(intor, self._atm, self._bas, self._env, N, None, comp, "s1", None, None, None, which_integral)

def ipu_getints4c(intor_name, atm, bas, env, N, shls_slice=None, comp=1,
            aosym='s1', ao_loc=None, cintopt=None, out=None, which_integral=-1):
    natm = atm.shape[0]
    nbas = bas.shape[0]

    shls_slice = (0, nbas, 0, nbas, 0, nbas, 0, nbas)
    
    shape = [comp, N, N, N, N] 
    
    drv = libcgto.GTOnr2e_fill_drv
    fill = getattr(libcgto, 'GTOnr2e_fill_'+aosym)
    out = numpy.ndarray(shape, buffer=out)

    # type 
    float32 = "#define dtype float" in open("libcint.c", "r").read()
    if float32: 
        out = out.astype(np.float32)
        env = env.astype(np.float32)


    from tessellate_ipu import create_ipu_tile_primitive, ipu_cycle_count, tile_map, tile_put_sharded, tile_put_replicated
    vertex_filename  = osp.join(osp.dirname(__file__), "libcint.cpp")
    grad = create_ipu_tile_primitive(
            "Int2e" ,
            "Int2e" ,
            inputs=["mat", "shls_slice", "ao_loc", "atm", "bas", "env", "natm", "nbas", "which_integral", "comp"], 
            outputs={"out": 0},
            gp_filename=vertex_filename,
            perf_estimate=100,
    )

    natm = atm.shape[0]
    nbas = bas.shape[0]

    prefix = 'GTO'

    float32 = "#define dtype float" in open("libcint.c", "r").read()
    if float32: 
        out = out.astype(np.float32)
        env = env.astype(np.float32)


    out = tile_put_replicated(jnp.array(out, dtype=jnp.float32),   (1,)) 
    shls_slice = tile_put_replicated(jnp.array(shls_slice, dtype=jnp.int32),   (1,)) 
    ao_loc = tile_put_replicated(jnp.array(ao_loc, dtype=jnp.int32),   (1,)) 
    atm = tile_put_replicated(jnp.array(atm, dtype=jnp.int32),   (1,)) 
    bas = tile_put_replicated(jnp.array(bas, dtype=jnp.int32),   (1,)) 
    env = tile_put_replicated(jnp.array(env, dtype=jnp.float32),   (1,)) 
    natm = tile_put_replicated(jnp.array(natm, dtype=jnp.int32),   (1,)) 
    nbas = tile_put_replicated(jnp.array(nbas, dtype=jnp.int32),   (1,)) 
    comp = tile_put_replicated(jnp.array(comp, dtype=jnp.int32),   (1,)) 

    which_integral = tile_put_replicated(np.array(which_integral, dtype=jnp.int32),   (1,)) 

    value = tile_map(grad, out, shls_slice, ao_loc, atm, bas, env, natm, nbas, which_integral, comp)

    out = value.array 

    if comp == 1:
        out = out[0]
    return out

def ipu_intor2e(self, intor, N, comp=None, hermi=0, aosym='s1', out=None, shls_slice=None, grids=None, which_integral=-1):
    ao_loc = make_loc(mol._bas, intor)
    return np.asarray(jax.jit(ipu_getints4c, static_argnums=(0,4,5,6,7,9,10,11))
                        (intor, self._atm, self._bas, self._env, N, None, comp, "s1", ao_loc, None, None, which_integral))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(prog='popcint', description='Libcint compiled to poplar. ')
    parser.add_argument('-nuc', action="store_true")
    parser.add_argument('-kin', action="store_true")
    parser.add_argument('-ovlp', action="store_true")
    parser.add_argument('-eri', action="store_true")
    parser.add_argument('-nucgrad', action="store_true")
    parser.add_argument('-kingrad', action="store_true")
    parser.add_argument('-ovlpgrad', action="store_true")
    parser.add_argument('-erigrad', action="store_true")
    parser.add_argument('-all', action="store_true")
    parser.add_argument("-basis", type=str, default="sto3g")
    args = parser.parse_args()

    mol = pyscf.gto.Mole(atom="H 0 0 0; H 0 0 1; ", basis=args.basis)
    mol.build()
    N = mol.nao_nr()
    print("[N=%i]"%N)

    def test(truth, us, str):
        error = np.max(us.reshape(-1)-truth.reshape(-1))
        print(str, error)
        if error > 1e-6:
            print(us.reshape(-1))
            print(truth.reshape(-1))

    if args.nuc or args.all: 
        print("\n[Nuclear Integral]")
        us    =  cpu_intor1e(mol, 'int1e_nuc', N, comp=1)
        truth =  mol.intor('int1e_nuc', comp=1)
        test(us, truth, "CPU: \t")
        us    = np.asarray( ipu_intor1e(mol, "int1e_nuc", INT1E_NUC,  N, 1))
        test(us, truth, "IPU: \t")

    if args.kin or args.all: 
        print("\n[Kinetic Integral]")
        us    =  cpu_intor1e(mol, 'int1e_kin', N, comp=1)
        truth =  mol.intor('int1e_kin', comp=1)
        test(us, truth, "CPU: \t")
        us =   np.asarray( ipu_intor1e(mol, "int1e_kin", INT1E_KIN,  N, 1))
        test(us, truth, "IPU: \t")
 
    if args.ovlp or args.all:
        print("\n[Overlap Integral]")
        us    =  cpu_intor1e(mol, 'int1e_ovlp', N, comp=1)
        truth =  mol.intor('int1e_ovlp', comp=1)
        test(us, truth, "CPU: \t")
        us    = np.asarray( ipu_intor1e(mol, "int1e_ovlp", INT1E_OVLP,  N, 1))
        test(us, truth, "IPU: \t")

    if args.nucgrad or args.all:
        print("\n[Grad Nuclear]")
        us    = - cpu_intor1e(mol, 'int1e_ipnuc', N, comp=3)
        truth = - mol.intor('int1e_ipnuc', comp=3)
        test(us, truth, "CPU: \t")
        us =  - np.transpose(np.asarray( ipu_intor1e(mol, "int1e_ipnuc", INT1E_NUC_IP,  N, 3)), (0,2,1))
        test(us, truth, "IPU: \t")

    if args.kingrad or args.all:
        print("\n[Grad Kinetic]")
        us    = - cpu_intor1e(mol, 'int1e_ipkin', N, comp=3)
        truth = - mol.intor('int1e_ipkin', comp=3)
        test(us, truth, "CPU: \t")
        us    =  - np.transpose(np.asarray( ipu_intor1e(mol, "int1e_ipkin", INT1E_KIN_IP,  N, 3)), (0,2,1))
        test(us, truth, "IPU: \t")

    if args.ovlpgrad or args.all:
        print("\n[Grad Overlap]")
        us    = - cpu_intor1e(mol, 'int1e_ipovlp', N, comp=3)
        truth = - mol.intor('int1e_ipovlp', comp=3)
        test(us, truth, "CPU: \t")
        us    =  - np.transpose(np.asarray( ipu_intor1e(mol, "int1e_ipovlp", INT1E_OVLP_IP,  N, 3)), (0,2,1))
        test(us, truth, "IPU: \t")
        
    if args.eri or args.all: 
        print("\n[Electron Repulsion Integral]")
        truth = mol.intor("int2e_sph")
        us    = cpu_intor2e(mol, "int2e_sph", N, 1, which_integral=INT2E_SPH)
        test(us, truth, "CPU: \t")
        us    = ipu_intor2e(mol, "int2e_sph", N, 1, which_integral=INT2E_SPH)
        test(us, truth, "IPU: \t")

    if args.erigrad or args.all:
        print("\n[Grad of Electron Repulsion Integral]")
        truth = mol.intor("int2e_ip1_sph")
        us    = cpu_intor2e(mol, "int2e_ip1_sph", N, 3, which_integral=INT2E_IP1_SPH)
        test(us, truth, "CPU: \t")
        us    = ipu_intor2e(mol, "int2e_ip1_sph", N, 3, which_integral=INT2E_IP1_SPH)
        test(us, truth, "IPU: \t")

