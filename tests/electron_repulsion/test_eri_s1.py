import pytest
import numpy as np 
import pyscf 
import jax
from electron_repulsion.direct import single
from ctypes import * # gives cdll 
import os
import ctypes 
from pyscf import lib

from numpy.testing import assert_allclose
try: 
  # TODO: replace with cppimport + pybind as described in jax-ipu-addons
  ret = os.system("cd tests/electron_repulsion && ./direct.sh")
  assert ret == 0, f"Failed to compile libcgto. Return code {ret}"
  libcgto = cdll.LoadLibrary( "tests/electron_repulsion/gen.so" ) 
except:
  print("Couldn't load gen.so, likely because you didn't run compile.sh to compiel the C++ file ")


@pytest.mark.parametrize("basis", ["sto-3g", "6-31g"])
def test_int2e_sph(basis):
  # Fetch the specific integral we want to micro benchmark. 
  mol = pyscf.gto.mole.Mole()
  mol.build(atom="C 0 0 0;", unit="Bohr", basis=basis, spin=0, verbose=0)

  # Shapes/sizes.
  atm, bas, env   = mol._atm, mol._bas, mol._env
  n_atm, n_bas, N = atm.shape[0], bas.shape[0], mol.nao_nr()
  shape           = [1, N, N, N, N] 

  # Initialize buffer for CPU libcint computation. 
  buf     = np.zeros(np.prod(shape)*2)

  from electron_repulsion.direct import prepare_integrals_2_inputs
  input_floats, input_ints, tuple_ijkl = prepare_integrals_2_inputs(mol)[:3]

  input_ijkl = np.array(tuple_ijkl[0])
  micro = 5
  input_ijkl = input_ijkl[micro: micro+1]

  # Run on CPU. 
  _ = libcgto.int2e_sph(
          buf.ctypes.data_as(ctypes.c_void_p), 
          ctypes.c_int(np.prod(buf.shape)), 
          lib.c_null_ptr(),
          (ctypes.c_int*4)(*input_ijkl.tolist()[0]),
          atm.ctypes.data_as(ctypes.c_void_p),
          ctypes.c_int(n_atm), 
          bas.ctypes.data_as(ctypes.c_void_p), 
          ctypes.c_int(n_bas), 
          env.ctypes.data_as(ctypes.c_void_p), 
          ctypes.c_int(np.prod(env.shape))) 

  # Run on IPU. 
  tiles     = (1,)
  ipu_out, start_cycles, stop_cycles = single(input_floats, input_ints, input_ijkl, tiles)
  stop_cycles, start_cycles, ipu_out = np.asarray(stop_cycles), np.asarray(start_cycles), np.asarray(ipu_out)
  cycles                             = (stop_cycles[:, 0] - start_cycles[:, 0]).reshape(-1)
  print("[Cycles M]")
  print(cycles/10**6)
  print("> IPU")
  print(ipu_out[ipu_out!=0])
  print("> CPU")
  print(buf[buf !=0])
  print("> Diff")
  print(np.max(np.abs(ipu_out[ipu_out!=0]-buf[buf!=0])))

  assert_allclose(ipu_out[ipu_out!=0], buf[buf!=0], rtol=1e-6)


@pytest.mark.parametrize("basis", ["sto-3g", "6-31g"])
def test_cpu_dense_to_ipu_sparse(basis):
  mol = pyscf.gto.mole.Mole()
  mol.build(atom="C 0 0 0;", unit="Bohr", basis=basis, spin=0, verbose=0)

  dense_s1 = mol.intor("int2e_sph", aosym="s1")
  print(dense_s1.shape)

  N = dense_s1.shape[0]

  np.random.seed(42)
  dm  = np.random.normal(0,1, (N, N)) 
  dm  = dm + dm.T 
  dm  = np.linalg.qr(dm)[0] # this reduces errors quite a bit 

  # This only uses one thread ;; may want a more ellaborate test case 
  from electron_repulsion.direct import prepare_integrals_2_inputs , compute_integrals_2 
  _input_floats, _input_ints, _tuple_ijkl, _shapes, _sizes, _counts, indxs_inv, num_calls = prepare_integrals_2_inputs(mol)
  sparse_s8, cycles_start, cycles_stop = compute_integrals_2( _input_floats, _input_ints, _tuple_ijkl, _shapes, _sizes, _counts, tuple(indxs_inv))

  sparse_s8 = np.asarray(sparse_s8)

  from electron_repulsion.direct import prepare_integrals_2_inputs , compute_integrals_2 , ipu_direct_mult, prepare_ipu_direct_mult_inputs
  _tuple_indices, _tuple_do_lists, _N = prepare_ipu_direct_mult_inputs(num_calls, mol)

  vj_ipu, vk_ipu = jax.jit(ipu_direct_mult, backend="ipu", static_argnums=(2,3,4,5))(sparse_s8,
                                      dm, 
                                      _tuple_indices,
                                      _tuple_do_lists, _N, num_calls)
  vj_ipu = np.asarray(vj_ipu)
  vk_ipu = np.asarray(vk_ipu)
  vj_cpu = np.einsum('ijkl,ji->kl', dense_s1, dm)
  vk_cpu = np.einsum('ijkl,jk->il', dense_s1, dm) 

  assert_allclose(vj_cpu, vj_ipu, rtol=1e-6)
  assert_allclose(vk_cpu, vk_ipu, rtol=1e-5)
