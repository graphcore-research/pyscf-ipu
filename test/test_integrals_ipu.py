# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from pyscf_ipu.experimental.basis import basisset
from pyscf_ipu.experimental.device import has_ipu, ipu_func
from pyscf_ipu.experimental.integrals import (
    eri_basis_sparse,
    kinetic_primitives,
    overlap_primitives,
)
from pyscf_ipu.experimental.interop import to_pyscf
from pyscf_ipu.experimental.primitive import Primitive
from pyscf_ipu.experimental.structure import molecule


@pytest.mark.skipif(not has_ipu(), reason="Skipping ipu test!")
def test_overlap():
    from pyscf_ipu.experimental.integrals import _overlap_primitives

    a, b = [Primitive()] * 2
    actual = ipu_func(_overlap_primitives)(a, b)
    assert_allclose(actual, overlap_primitives(a, b))


@pytest.mark.skipif(not has_ipu(), reason="Skipping ipu test!")
def test_kinetic():
    from pyscf_ipu.experimental.integrals import _kinetic_primitives

    a, b = [Primitive()] * 2
    actual = ipu_func(_kinetic_primitives)(a, b)
    assert_allclose(actual, kinetic_primitives(a, b))


@pytest.mark.skipif(not has_ipu(), reason="Skipping ipu test!")
def test_nuclear():
    from pyscf_ipu.experimental.integrals import _nuclear_primitives

    # PyQuante test case for nuclear attraction integral
    a, b = [Primitive()] * 2
    c = jnp.zeros(3)
    actual = ipu_func(_nuclear_primitives)(a, b, c)
    assert_allclose(actual, -1.595769, atol=1e-5)


@pytest.mark.skipif(not has_ipu(), reason="Skipping ipu test!")
def test_eri():
    from pyscf_ipu.experimental.integrals import _eri_primitives

    # PyQuante test cases for ERI
    a, b, c, d = [Primitive()] * 4
    actual = ipu_func(_eri_primitives)(a, b, c, d)
    assert_allclose(actual, 1.128379, atol=1e-5)


@pytest.mark.skipif(not has_ipu(), reason="Skipping ipu test!")
def test_water_eri():
    basis_name = "sto-3g"
    h2o = molecule("water")
    basis = basisset(h2o, basis_name)
    actual = ipu_func(eri_basis_sparse)(basis)
    expect = to_pyscf(h2o, basis_name=basis_name).intor("int2e_cart", aosym="s8")
    assert_allclose(actual, expect, atol=1e-4)
