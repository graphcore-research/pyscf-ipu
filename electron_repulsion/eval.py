import numpy as np
from pyscf import gto, dft
import numpy 

import ctypes
import unittest
import numpy
from pyscf import lib, gto, df
from pyscf.gto.eval_gto import BLKSIZE

import os, sys
import warnings
import tempfile
import functools
import itertools
import inspect
import collections
import ctypes
import numpy
import h5py
from threading import Thread
from multiprocessing import Queue, Process
try:
    from concurrent.futures import ThreadPoolExecutor
except ImportError:
    ThreadPoolExecutor = None

from pyscf.lib import param
from pyscf import __config__

if h5py.version.version[:4] == '2.2.':
    sys.stderr.write('h5py-%s is found in your environment. '
                     'h5py-%s has bug in threading mode.\n'
                     'Async-IO is disabled.\n' % ((h5py.version.version,)*2))

c_double_p = ctypes.POINTER(ctypes.c_double)
c_int_p = ctypes.POINTER(ctypes.c_int)
c_null_ptr = ctypes.POINTER(ctypes.c_void_p)

def load_library(libname):
  from pyscf import __path__ as ext_modules
  for path in ext_modules:
      libpath = os.path.join(path, 'lib')
      if os.path.isdir(libpath):
          for files in os.listdir(libpath):
              if files.startswith(libname):
                  print(libname, libpath)
                  return numpy.ctypeslib.load_library(libname, libpath)
  raise

#libcgto = load_library('libdft')
#libcgto = numpy.ctypeslib.load_library("libdft", "/nethome/alexm/.local/lib/python3.8/site-packages/pyscf/lib/")
libcgto = numpy.ctypeslib.load_library("eval", "")

print(libcgto.int2e_sph)


# Define the molecule geometry
#mol = gto.M(atom='''O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587; C 1 2 3; N 2 3 1; H -1 0 0;''', basis='sto-3g')
#mol = gto.M(atom='''O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587; C 1 2 3; N 2 3 1; H -1 0 0;''', basis='631g')
mol = gto.M(atom='''O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587; C 1 2 3; N 2 3 1; H -1 0 0;''', basis='631g*')

# Generate a grid
grid = dft.gen_grid.Grids(mol)
grid.level = 3  # level of grid spacing, higher number means finer grid
grid.build()

# Evaluate the atomic orbitals on the grid
#ao_values = dft.numint.eval_ao(mol, grid.coords)
target = mol.eval_gto('GTOval_cart_deriv1' if mol.cart else 'GTOval_sph_deriv1', grid.coords, 4)
#us = mol.eval_gto('GTOval_cart_deriv1' if mol.cart else 'GTOval_sph_deriv1', grid.coords, 4)


def eval_gto(mol, eval_name, coords,
             comp=1, shls_slice=None, non0tab=None, ao_loc=None, out=None):
    atm = numpy.asarray(mol._atm, dtype=numpy.int32, order='C')
    bas = numpy.asarray(mol._bas, dtype=numpy.int32, order='C')
    env = numpy.asarray(mol._env, dtype=numpy.double, order='C')
    coords = numpy.asarray(coords, dtype=numpy.double, order='F')
    natm = atm.shape[0]
    nbas = bas.shape[0]
    ngrids = coords.shape[0]
    ao_loc = gto.moleintor.make_loc(bas, eval_name)

    shls_slice = (0, nbas)
    sh0, sh1 = shls_slice
    nao = ao_loc[sh1] - ao_loc[sh0]
    ao = numpy.ndarray((comp,nao,ngrids), buffer=out)

    non0tab = numpy.ones(((ngrids+BLKSIZE-1)//BLKSIZE,nbas), dtype=numpy.uint8)

    print(eval_name)
    drv = getattr(libcgto, eval_name)
    drv(ctypes.c_int(ngrids),
        (ctypes.c_int*2)(*shls_slice), ao_loc.ctypes.data_as(ctypes.c_void_p),
        ao.ctypes.data_as(ctypes.c_void_p),
        coords.ctypes.data_as(ctypes.c_void_p),
        non0tab.ctypes.data_as(ctypes.c_void_p),
        atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(natm),
        bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nbas),
        env.ctypes.data_as(ctypes.c_void_p))


    # do micro test 


    ao = numpy.swapaxes(ao, -1, -2)
    return ao



# move for loop to python 
def eval_gto_python(mol, eval_name, coords,
             target, comp=1, shls_slice=None, non0tab=None, ao_loc=None, out=None):
    atm = numpy.asarray(mol._atm, dtype=numpy.int32, order='C')
    bas = numpy.asarray(mol._bas, dtype=numpy.int32, order='C')
    env = numpy.asarray(mol._env, dtype=numpy.double, order='C') # lol, the memory layout is switched... o hthat's annoying! 
    coords = numpy.asarray(coords, dtype=numpy.double, order='F') # lol 
    natm = atm.shape[0]
    nbas = bas.shape[0]
    ngrids = coords.shape[0]
    ao_loc = gto.moleintor.make_loc(bas, eval_name)

    shls_slice = (0, nbas)
    sh0, sh1 = shls_slice
    nao = ao_loc[sh1] - ao_loc[sh0]
    ao = numpy.ndarray((comp,nao,ngrids), buffer=out)

    non0tab = numpy.ones(((ngrids+BLKSIZE-1)//BLKSIZE,nbas), dtype=numpy.uint8)

    print(eval_name)
    if False: 
        drv = getattr(libcgto, eval_name)
        drv(ctypes.c_int(ngrids),
            (ctypes.c_int*2)(*shls_slice), ao_loc.ctypes.data_as(ctypes.c_void_p),
            ao.ctypes.data_as(ctypes.c_void_p),
            coords.ctypes.data_as(ctypes.c_void_p),
            non0tab.ctypes.data_as(ctypes.c_void_p),
            atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(natm),
            bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nbas),
            env.ctypes.data_as(ctypes.c_void_p))


    # do micro test 
    #extern "C" void GTOeval_sph_iter(size_t nao, size_t ngrids, size_t bgrids, int *shls_slice, int *ao_loc, double *ao, double *coord, uint8_t *non0table, int *atm, int natm, int *bas, int nbas, double *env)
    NPRIMAX         = 40
    #BLKSIZE        = 56 # defined somewhere else?
    nblk = (ngrids+BLKSIZE-1) // BLKSIZE
    k = 0
    shloc = np.zeros(shls_slice[1]-shls_slice[0]+1, dtype=np.int32)

    # compute nshblk.

    # slots of bas
    ATOM_OF         =0
    ANG_OF          =1
    NPRIM_OF        =2
    NCTR_OF         =3
    KAPPA_OF        =4
    PTR_EXP         =5
    PTR_COEFF       =6
    RESERVE_BASLOT  =7
    BAS_SLOTS       =8
    sh0 = shls_slice[0];
    sh1 = shls_slice[1];
    #ish, nshblk, lastatm;
    shloc[0] = sh0;
    nshblk = 1;
    lastatm = bas.reshape(-1)[BAS_SLOTS*sh0+ATOM_OF];
    for ish in range(sh0, sh1):
        if lastatm != bas.reshape(-1)[BAS_SLOTS*ish+ATOM_OF]:
                lastatm = bas.reshape(-1)[BAS_SLOTS*ish+ATOM_OF];
                shloc[nshblk] = ish;
                nshblk += 1
        
    shloc[nshblk] = sh1;
    print(shloc)

    target = numpy.swapaxes(target, -1, -2)
    #[BLKSIZE=56][ngrids=72288][nao=48][ncart=512][NPRIMAX=40][nblk=1291][nshblk=6]shloc[1]=6
    print(BLKSIZE, ngrids, nao, NPRIMAX, nblk, nshblk, shloc[1]) # no ncart?
    #56 72288 48 40 1291 6 6;; values look ok! 

    #         for (k = 0; k < nblk*nshblk; k++) {
    from tqdm import tqdm 
    pbar = tqdm(range(0, nblk*nshblk))
    for k in pbar:
        iloc = k // nblk
        
        ish = shloc[iloc];

        aoff = ao_loc[ish] - ao_loc[sh0];
        ib = k - iloc * nblk;
        ip = ib * BLKSIZE;
        bgrids = min(ngrids-ip, BLKSIZE);
        #GTOeval_sph_iter(nao, Ngrids, bgrids, shloc+iloc, ao_loc, ao+aoff*Ngrids+ip, coord+ip, non0table+ib*nbas, atm, natm, bas, nbas, env);
        #ao+aoff*Ngrids+ip
        pbar.set_description("iloc=%i ao[%i:%i] coords[%i:%i]"%(iloc, aoff*ngrids+ip, ao.size, ip, coords.size))
        print(ao.shape,coords.shape, ip)
        libcgto.GTOeval_sph_iter(
            ctypes.c_int(mol.nao_nr()),
            ctypes.c_int(ngrids),
            ctypes.c_int(bgrids),
            ctypes.c_int(shloc[iloc]), 
            ctypes.c_int(shloc[iloc+1]), 
            ao_loc.ctypes.data_as(ctypes.c_void_p),
            ao.reshape(-1)[aoff*ngrids+ip:].ctypes.data_as(ctypes.c_void_p),
            coords[ip:].ctypes.data_as(ctypes.c_void_p),
            non0tab.ctypes.data_as(ctypes.c_void_p),
            atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(natm),
            bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nbas),
            env.ctypes.data_as(ctypes.c_void_p))


        #if k % 100 == 0: assert np.allclose(ao, target)

        #print(ao)
    # GTOeval_sph_iter(nao, Ngrids, bgrids, shloc+iloc, ao_loc, ao+aoff*Ngrids+ip, coord+ip, non0table+ib*nbas, atm, natm, bas, nbas, env);


    ao = numpy.swapaxes(ao, -1, -2)
    return ao

#us = eval_gto()

us = eval_gto(mol, 'GTOval_cart_deriv1' if mol.cart else 'GTOval_sph_deriv1', grid.coords, 4)
us = eval_gto_python(mol, 'GTOval_cart_deriv1' if mol.cart else 'GTOval_sph_deriv1', grid.coords, target, 4)


# try to run with tilejax?

import matplotlib.pyplot as plt 
fig, ax = plt.subplots(1,1, figsize=(12, 4))

skip = 1024

us = us.reshape(-1)
target = target.reshape(-1)

diff = np.abs(us-target) # sort by error
indxs = np.flip(np.argsort(diff))

target = target[indxs]
us = us[indxs]
diff = diff[indxs]

print(diff)
print(diff[::1024])

#ax.plot(np.abs(target).reshape(-1)[::skip], 'o', label="target", ms=5, alpha=0.3)
#ax.plot(np.abs(us).reshape(-1)[::skip], 'kx', label="us", ms=1)
ax.plot(np.abs(diff).reshape(-1)[:skip], 'r^', label="|us-target|")
ax.set_ylim([1e-10, 1e3])
ax.legend()
ax.set_yscale("log")

plt.savefig("eval_gto.png")


print(np.max(np.abs(target-us)))
print(np.mean(np.abs(target-us)))
print(np.median(np.abs(target-us)))
print(np.min(np.abs(target-us)))
assert np.allclose(target, us )


print("PASS!")