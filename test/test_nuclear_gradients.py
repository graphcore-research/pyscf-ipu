# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import numpy as np
from numpy.testing import assert_allclose

from pyscf_ipu.experimental.basis import basisset
from pyscf_ipu.experimental.interop import to_pyscf
from pyscf_ipu.experimental.nuclear_gradients import (
    grad_kinetic_basis,
    grad_nuclear_basis,
    grad_overlap_basis,
)
from pyscf_ipu.experimental.structure import molecule


def test_nuclear_gradients():
    basis_name = "sto-3g"
    h2 = molecule("h2")
    scfmol = to_pyscf(h2, basis_name)
    basis = basisset(h2, basis_name)

    actual = grad_overlap_basis(basis)
    expect = scfmol.intor("int1e_ipovlp_cart", comp=3)
    assert_allclose(actual, expect, atol=1e-6)

    actual = grad_kinetic_basis(basis)
    expect = scfmol.intor("int1e_ipkin_cart", comp=3)
    assert_allclose(actual, expect, atol=1e-6)

    # TODO: investigate possible inconsistency in libcint outputs?
    actual = grad_nuclear_basis(basis)
    expect = scfmol.intor("int1e_ipnuc_cart", comp=3)
    expect = -np.moveaxis(expect, 1, 2)
    assert_allclose(actual, expect, atol=1e-6)
