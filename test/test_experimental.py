# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import jax.numpy as jnp
import numpy as np
import pytest
from jax import tree_map, vmap
from numpy.testing import assert_allclose

from pyscf_ipu.experimental.basis import basisset
from pyscf_ipu.experimental.integrals import (
    overlap_primitives,
    nuclear_basis,
    overlap_basis,
    nuclear_primitives,
    eri_basis,
    eri_basis_sparse,
    eri_primitives,
    kinetic_primitives,
    kinetic_basis,
)
from pyscf_ipu.experimental.mesh import electron_density, uniform_mesh
from pyscf_ipu.experimental.primitive import Primitive
from pyscf_ipu.experimental.structure import to_pyscf, water, Structure


@pytest.mark.parametrize("basis_name", ["sto-3g", "6-31g**"])
def test_to_pyscf(basis_name):
    mol = water()
    basis = basisset(mol, basis_name)
    pyscf_mol = to_pyscf(mol, basis_name)
    assert basis.num_orbitals == pyscf_mol.nao


@pytest.mark.parametrize("basis_name", ["sto-3g", "6-31+g"])
def test_gto(basis_name):
    from pyscf.dft.numint import eval_rho

    # Atomic orbitals
    structure = water()
    basis = basisset(structure, basis_name)
    mesh = uniform_mesh()
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
    basis = basisset(water(), basis_name)
    actual_overlap = overlap_basis(basis)

    # Note: PySCF doesn't appear to normalise d basis functions in cartesian basis
    expect_overlap = to_pyscf(water(), basis_name=basis_name).intor("int1e_ovlp_cart")
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
    h2 = Structure(
        atomic_number=np.array([1, 1]),
        position=np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]]),
    )
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
    basis = basisset(water(), basis_name)
    actual = kinetic_basis(basis)

    expect = to_pyscf(water(), basis_name=basis_name).intor("int1e_kin_cart")
    assert_allclose(actual, expect, atol=1e-4)


def test_nuclear():
    # PyQuante test case for nuclear attraction integral
    p = Primitive()
    c = jnp.zeros(3)
    assert_allclose(nuclear_primitives(p, p, c), -1.595769, atol=1e-5)

    # Reproduce the nuclear attraction matrix for H2 using STO-3G basis set
    # See equation 3.231 and 3.232 of Szabo and Ostlund
    h2 = Structure(
        atomic_number=np.array([1, 1]),
        position=np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]]),
    )
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
    h2o = water()
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
    h2 = Structure(
        atomic_number=np.array([1, 1]),
        position=np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]]),
    )
    basis = basisset(h2, "sto-3g")
    indices = [(0, 0, 0, 0), (0, 0, 1, 1), (1, 0, 0, 0), (1, 0, 1, 0)]
    expected = [0.7746, 0.5697, 0.4441, 0.2970]

    for ijkl, expect in zip(indices, expected):
        actual = eri_orbitals([basis.orbitals[aoid] for aoid in ijkl])
        assert_allclose(actual, expect, atol=1e-4)


def test_eri_basis():
    # H2 molecule in sto-3g: See equation 3.235 of Szabo and Ostlund
    h2 = Structure(
        atomic_number=np.array([1, 1]),
        position=np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]]),
    )
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


@pytest.mark.parametrize("sparse", [True, False])
def test_water_eri(sparse):
    basis_name = "sto-3g"
    h2o = water()
    basis = basisset(h2o, basis_name)
    actual = eri_basis_sparse(basis) if sparse else eri_basis(basis)
    aosym = "s8" if sparse else "s1"
    expect = to_pyscf(h2o, basis_name=basis_name).intor("int2e_cart", aosym=aosym)
    assert_allclose(actual, expect, atol=1e-4)
