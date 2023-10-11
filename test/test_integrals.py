# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

import pyscf_ipu.experimental as pyscf_experimental
from pyscf_ipu.experimental.basis import basisset
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
from pyscf_ipu.experimental.primitive import Primitive
from pyscf_ipu.experimental.structure import molecule


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


def check_recompile(recompile, function):
    # Force recompile
    if recompile == "recompile":
        # TBH, this is a bit of a red herring - it will force recompilation,
        # but the whole switch is only really useful if the False case
        # runs after the true case in the same process
        # i.e. timing from
        #    pytest -k test_nuclear[lookup-recompile] --durations=5
        # will be the same as
        #    pytest -k test_nuclear[lookup-cached] --durations=5
        # While
        #    pytest -k test_nuclear[lookup- --durations=5
        # will show both times, and cached will be lower
        function._clear_cache()


@pytest.mark.parametrize("recompile", ["recompile", "cached"])
@pytest.mark.parametrize("binom_factor_str", ["segment_sum", "lookup"])
def test_nuclear(binom_factor_str, recompile):
    # PyQuante test case for nuclear attraction integral
    p = Primitive()
    c = jnp.zeros(3)

    # Choose the implementation of binom_factor
    if binom_factor_str == "segment_sum":
        binom_factor = pyscf_experimental.special.binom_factor__via_segment_sum
    elif binom_factor_str == "lookup":
        binom_factor = pyscf_experimental.special.binom_factor__via_lookup
    else:
        assert False

    check_recompile(recompile, nuclear_primitives)
    assert_allclose(nuclear_primitives(p, p, c, binom_factor), -1.595769, atol=1e-5)

    # if recompile == 'recompile':
    #     from jaxutils.jaxpr_to_expr import show_jaxpr
    #     show_jaxpr(
    #         nuclear_primitives,
    #         (p, p, c, binom_factor),
    #         file=f"tmp/nuclear_primitives_jaxpr__binom_factor__via_{binom_factor_str}.py",
    #         optimize=False,
    #         static_argnums=3,
    #     )

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


def test_eri():
    # PyQuante test cases for ERI
    a, b, c, d = [Primitive()] * 4
    assert_allclose(eri_primitives(a, b, c, d), 1.128379, atol=1e-5)

    c, d = [Primitive(lmn=jnp.array([1, 0, 0]))] * 2
    assert_allclose(eri_primitives(a, b, c, d), 0.940316, atol=1e-5)

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


@pytest.mark.parametrize("recompile", ["recompile", "cached"])
@pytest.mark.parametrize("binom_factor_str", ["segment_sum", "lookup"])
@pytest.mark.parametrize("sparsity", ["sparse", "dense"])
@pytest.mark.skipif(is_mem_limited(), reason="Not enough host memory!")
def test_water_eri(recompile, binom_factor_str, sparsity):
    sparse = sparsity == "sparse"
    check_recompile(recompile, eri_primitives)
    binom_factor = eval(
        "pyscf_experimental.special.binom_factor__via_" + binom_factor_str
    )

    basis_name = "sto-3g"
    h2o = molecule("water")
    basis = basisset(h2o, basis_name)
    if sparse:
        actual = eri_basis_sparse(basis, binom_factor)
    else:
        actual = eri_basis(basis, binom_factor)
    aosym = "s8" if sparse else "s1"
    expect = to_pyscf(h2o, basis_name=basis_name).intor("int2e_cart", aosym=aosym)
    assert_allclose(actual, expect, atol=1e-4)
