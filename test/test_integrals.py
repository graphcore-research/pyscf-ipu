# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import jax.numpy as jnp
import numpy as np
import pytest
from jax import tree_map, vmap
from numpy.testing import assert_allclose

from pyscf_ipu.experimental.basis import basisset
from pyscf_ipu.experimental.device import has_ipu, ipu_func
from pyscf_ipu.experimental.integrals import (
    eri_basis,
    eri_basis_sparse,
    eri_primitives,
    kinetic_basis,
    kinetic_primitives,
    nuclear_basis,
    nuclear_primitives,
    overlap_basis,
    overlap_primitives,
)
from pyscf_ipu.experimental.interop import to_pyscf
from pyscf_ipu.experimental.nuclear_gradients import (
    grad_kinetic_basis,
    grad_nuclear_basis,
    grad_overlap_basis,
)
from pyscf_ipu.experimental.primitive import Primitive
from pyscf_ipu.experimental.structure import molecule


@pytest.mark.parametrize("basis_name", ["sto-3g", "6-31g**"])
def test_to_pyscf(basis_name):
    mol = molecule("water")
    basis = basisset(mol, basis_name)
    pyscf_mol = to_pyscf(mol, basis_name)
    assert basis.num_orbitals == pyscf_mol.nao


@pytest.mark.parametrize("basis_name", ["sto-3g", "6-31+g"])
def test_gto(basis_name):
    from pyscf.dft.numint import eval_rho

    # Atomic orbitals
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


def test_overlap():
    # Exercise 3.21 of "Modern quantum chemistry: introduction to advanced
    # electronic structure theory."" by Szabo and Ostlund
    alpha = 0.270950 * 1.24 * 1.24
    a = Primitive(alpha=alpha)
    b = Primitive(alpha=alpha, center=jnp.array([1.4, 0.0, 0.0]))
    assert_allclose(overlap_primitives(a, a), 1.0, atol=1e-5)
    assert_allclose(overlap_primitives(b, b), 1.0, atol=1e-5)
    assert_allclose(overlap_primitives(b, a), 0.6648, atol=1e-5)


@pytest.mark.parametrize("basis_name", ["sto-3g", "6-31+g", "6-31+g*"])
def test_water_overlap(basis_name):
    basis = basisset(molecule("water"), basis_name)
    actual_overlap = overlap_basis(basis)

    # Note: PySCF doesn't appear to normalise d basis functions in cartesian basis
    scfmol = to_pyscf(molecule("water"), basis_name=basis_name)
    expect_overlap = scfmol.intor("int1e_ovlp_cart")
    n = 1 / np.sqrt(np.diagonal(expect_overlap))
    expect_overlap = n[:, None] * n[None, :] * expect_overlap
    assert_allclose(actual_overlap, expect_overlap, atol=1e-6)


def test_kinetic():
    # PyQuante test case for kinetic primitive integral
    p = Primitive()
    assert_allclose(kinetic_primitives(p, p), 1.5, atol=1e-6)

    # Reproduce the kinetic energy matrix for H2 using STO-3G basis set
    # See equation 3.230 of "Modern quantum chemistry: introduction to advanced
    # electronic structure theory."" by Szabo and Ostlund
    h2 = molecule("h2")
    basis = basisset(h2, "sto-3g")
    actual = kinetic_basis(basis)
    expect = np.array([[0.7600, 0.2365], [0.2365, 0.7600]])
    assert_allclose(actual, expect, atol=1e-4)


@pytest.mark.parametrize(
    "basis_name",
    [
        "sto-3g",
        "6-31+g",
        pytest.param(
            "6-31+g*", marks=pytest.mark.xfail(reason="Cartesian norm problem?")
        ),
    ],
)
def test_water_kinetic(basis_name):
    basis = basisset(molecule("water"), basis_name)
    actual = kinetic_basis(basis)

    expect = to_pyscf(molecule("water"), basis_name=basis_name).intor("int1e_kin_cart")
    assert_allclose(actual, expect, atol=1e-4)


def test_nuclear():
    # PyQuante test case for nuclear attraction integral
    p = Primitive()
    c = jnp.zeros(3)
    assert_allclose(nuclear_primitives(p, p, c), -1.595769, atol=1e-5)

    # Reproduce the nuclear attraction matrix for H2 using STO-3G basis set
    # See equation 3.231 and 3.232 of Szabo and Ostlund
    h2 = molecule("h2")
    basis = basisset(h2, "sto-3g")
    actual = nuclear_basis(basis, h2.position, h2.atomic_number)
    expect = np.array(
        [
            [[-1.2266, -0.5974], [-0.5974, -0.6538]],
            [[-0.6538, -0.5974], [-0.5974, -1.2266]],
        ]
    )

    assert_allclose(actual, expect, atol=1e-4)


def test_water_nuclear():
    basis_name = "sto-3g"
    h2o = molecule("water")
    basis = basisset(h2o, basis_name)
    actual = nuclear_basis(basis, h2o.position, h2o.atomic_number).sum(axis=0)
    expect = to_pyscf(h2o, basis_name=basis_name).intor("int1e_nuc_cart")
    assert_allclose(actual, expect, atol=1e-4)


def eri_orbitals(orbitals):
    def take(orbital, index):
        p = tree_map(lambda *xs: jnp.stack(xs), *orbital.primitives)
        p = tree_map(lambda x: jnp.take(x, index, axis=0), p)
        c = jnp.take(orbital.coefficients, index)
        return p, c

    indices = [jnp.arange(o.num_primitives) for o in orbitals]
    indices = [i.reshape(-1) for i in jnp.meshgrid(*indices)]
    prim, coef = zip(*[take(o, i) for o, i in zip(orbitals, indices)])
    return jnp.sum(jnp.prod(jnp.stack(coef), axis=0) * vmap(eri_primitives)(*prim))


def test_eri():
    # PyQuante test cases for ERI
    a, b, c, d = [Primitive()] * 4
    assert_allclose(eri_primitives(a, b, c, d), 1.128379, atol=1e-5)

    c, d = [Primitive(lmn=jnp.array([1, 0, 0]))] * 2
    assert_allclose(eri_primitives(a, b, c, d), 0.940316, atol=1e-5)

    # H2 molecule in sto-3g: See equation 3.235 of Szabo and Ostlund
    h2 = molecule("h2")
    basis = basisset(h2, "sto-3g")
    indices = [(0, 0, 0, 0), (0, 0, 1, 1), (1, 0, 0, 0), (1, 0, 1, 0)]
    expected = [0.7746, 0.5697, 0.4441, 0.2970]

    for ijkl, expect in zip(indices, expected):
        actual = eri_orbitals([basis.orbitals[aoid] for aoid in ijkl])
        assert_allclose(actual, expect, atol=1e-4)


def test_eri_basis():
    # H2 molecule in sto-3g: See equation 3.235 of Szabo and Ostlund
    h2 = molecule("h2")
    basis = basisset(h2, "sto-3g")

    actual = eri_basis(basis)
    expect = np.empty((2, 2, 2, 2), dtype=np.float32)
    expect[0, 0, 0, 0] = expect[1, 1, 1, 1] = 0.7746
    expect[0, 0, 1, 1] = expect[1, 1, 0, 0] = 0.5697
    expect[1, 0, 0, 0] = expect[0, 0, 0, 1] = 0.4441
    expect[0, 1, 0, 0] = expect[0, 0, 1, 0] = 0.4441
    expect[0, 1, 1, 1] = expect[1, 1, 1, 0] = 0.4441
    expect[1, 0, 1, 1] = expect[1, 1, 0, 1] = 0.4441
    expect[1, 0, 1, 0] = expect[0, 1, 1, 0] = 0.2970
    expect[0, 1, 0, 1] = expect[1, 0, 0, 1] = 0.2970
    assert_allclose(actual, expect, atol=1e-4)


def is_mem_limited():
    # Check if we are running on a limited memory host (e.g. github action)
    import psutil

    total_mem_gib = psutil.virtual_memory().total // 1024**3
    return total_mem_gib < 10


@pytest.mark.parametrize("sparse", [True, False])
@pytest.mark.skipif(is_mem_limited(), reason="Not enough host memory!")
def test_water_eri(sparse):
    basis_name = "sto-3g"
    h2o = molecule("water")
    basis = basisset(h2o, basis_name)
    actual = eri_basis_sparse(basis) if sparse else eri_basis(basis)
    aosym = "s8" if sparse else "s1"
    expect = to_pyscf(h2o, basis_name=basis_name).intor("int2e_cart", aosym=aosym)
    print("max |actual - expect|  ={}", np.max(np.abs(actual - expect)))
    assert_allclose(actual, expect, atol=1e-4)


@pytest.mark.skipif(not has_ipu(), reason="Skipping ipu test!")
def test_ipu_overlap():
    from pyscf_ipu.experimental.integrals import _overlap_primitives

    a, b = [Primitive()] * 2
    actual = ipu_func(_overlap_primitives)(a, b)
    assert_allclose(actual, overlap_primitives(a, b))


@pytest.mark.skipif(not has_ipu(), reason="Skipping ipu test!")
def test_ipu_kinetic():
    from pyscf_ipu.experimental.integrals import _kinetic_primitives

    a, b = [Primitive()] * 2
    actual = ipu_func(_kinetic_primitives)(a, b)
    assert_allclose(actual, kinetic_primitives(a, b))


@pytest.mark.skipif(not has_ipu(), reason="Skipping ipu test!")
def test_ipu_nuclear():
    from pyscf_ipu.experimental.integrals import _nuclear_primitives

    # PyQuante test case for nuclear attraction integral
    a, b = [Primitive()] * 2
    c = jnp.zeros(3)
    actual = ipu_func(_nuclear_primitives)(a, b, c)
    assert_allclose(actual, -1.595769, atol=1e-5)


@pytest.mark.skipif(not has_ipu(), reason="Skipping ipu test!")
def test_ipu_eri():
    from pyscf_ipu.experimental.integrals import _eri_primitives

    # PyQuante test cases for ERI
    a, b, c, d = [Primitive()] * 4
    actual = ipu_func(_eri_primitives)(a, b, c, d)
    assert_allclose(actual, 1.128379, atol=1e-5)


@pytest.mark.parametrize("basis_name", ["sto-3g", "6-31+g"])
def test_nuclear_gradients(basis_name):
    h2 = molecule("h2")
    scfmol = to_pyscf(h2, basis_name)
    basis = basisset(h2, basis_name)

    actual = grad_overlap_basis(basis)
    expect = scfmol.intor("int1e_ipovlp_cart", comp=3)
    assert_allclose(actual, expect, atol=1e-6)

    actual = grad_kinetic_basis(basis)
    expect = scfmol.intor("int1e_ipkin_cart", comp=3)
    assert_allclose(actual, expect, atol=1e-6)

    actual = grad_nuclear_basis(basis)
    expect = scfmol.intor("int1e_ipnuc_cart", comp=3)
    assert_allclose(actual, expect, atol=1e-6)
