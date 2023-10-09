# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from pyscf_ipu.experimental.device import has_ipu, ipu_func
from pyscf_ipu.experimental.integrals import kinetic_primitives, overlap_primitives
from pyscf_ipu.experimental.primitive import Primitive


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
