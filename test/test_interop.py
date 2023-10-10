# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyscf_ipu.experimental.basis import basisset
from pyscf_ipu.experimental.interop import to_pyscf
from pyscf_ipu.experimental.mesh import electron_density, uniform_mesh
from pyscf_ipu.experimental.structure import molecule


@pytest.mark.parametrize("basis_name", ["sto-3g", "6-31g**"])
def test_to_pyscf(basis_name):
    mol = molecule("water")
    basis = basisset(mol, basis_name)
    pyscf_mol = to_pyscf(mol, basis_name)
    assert basis.num_orbitals == pyscf_mol.nao


def test_gto():
    from pyscf.dft.numint import eval_rho

    # Atomic orbitals
    basis_name = "6-31+g"
    structure = molecule("water")
    basis = basisset(structure, basis_name)
    mesh, _ = uniform_mesh()
    actual = basis(mesh)

    mol = to_pyscf(structure, basis_name)
    expect_ao = mol.eval_gto("GTOval_cart", np.asarray(mesh))
    assert_allclose(actual, expect_ao, atol=1e-6)

    # Molecular orbitals
    mf = mol.KS()
    mf.kernel()
    C = jnp.array(mf.mo_coeff, dtype=jnp.float32)
    actual = basis.occupancy * C @ C.T
    expect = jnp.array(mf.make_rdm1(), dtype=jnp.float32)
    assert_allclose(actual, expect, atol=1e-6)

    # Electron density
    actual = electron_density(basis, mesh, C)
    expect = eval_rho(mol, expect_ao, mf.make_rdm1(), "lda")
    assert_allclose(actual, expect, atol=1e-6)
