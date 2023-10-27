# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from subprocess import Popen, call

import jax
import numpy as np
import pytest
from tessellate_ipu import (
    ipu_cycle_count,
    tile_map,
    tile_put_replicated,
    tile_put_sharded,
)

from pyscf_ipu.experimental.device import has_ipu, ipu_func
from pyscf_ipu.nanoDFT.nanoDFT import build_mol, nanoDFT, nanoDFT_options


@pytest.mark.skipif(not has_ipu(), reason="Skipping ipu test!")
@pytest.mark.ipu
def test_basic_demonstration():
    dummy = np.random.rand(2, 3).astype(np.float32)
    dummier = np.random.rand(2, 3).astype(np.float32)

    @jax.jit
    def jitted_inner_test(dummy, dummier):
        tiles = tuple(range(len(dummy)))
        dummy = tile_put_sharded(dummy, tiles)
        tiles = tuple(range(len(dummier)))
        dummier = tile_put_sharded(dummier, tiles)

        dummy, dummier, start = ipu_cycle_count(dummy, dummier)
        out = tile_map(jax.lax.add_p, dummy, dummier)
        out, end = ipu_cycle_count(out)

        return out, start, end

    _, start, end = jitted_inner_test(dummy, dummier)
    print("Start cycle count:", start, start.shape)
    print("End cycle count:", end, end.shape)
    print("Diff cycle count:", end.array - start.array)

    assert True


@pytest.mark.skipif(not has_ipu(), reason="Skipping ipu test!")
@pytest.mark.ipu
@pytest.mark.parametrize("molecule", ["methane", "benzene"])
def test_dense_eri(molecule):
    opts, mol_str = nanoDFT_options(float32=True, mol_str=molecule, backend="ipu")
    mol = build_mol(mol_str, opts.basis)

    _, _, ipu_cycles_stamps = nanoDFT(mol, opts, profile_performance=True)

    start, end = ipu_cycles_stamps
    start = np.asarray(start)
    end = np.asarray(end)

    diff = (end - start)[0][0][0]
    print(
        "----------------------------------------------------------------------------"
    )
    print("                                Diff cycle count:", diff)
    print("                            Diff cycle count [M]:", diff / 1e6)
    print("Estimated time of execution on Bow-IPU [seconds]:", diff / (1.85 * 1e9))
    print(
        "----------------------------------------------------------------------------"
    )

    assert True

@pytest.mark.skip(reason="No IPU in CI.")
@pytest.mark.ipu
@pytest.mark.parametrize("molecule", ["methane", "benzene", "c20"])
def test_sparse_eri(molecule):
    opts, mol_str = nanoDFT_options(
        float32=True,
        mol_str=molecule,
        backend="ipu",
        dense_ERI=False,
        eri_threshold=1e-9,
    )
    mol = build_mol(mol_str, opts.basis)

    _, _, ipu_cycles_stamps = nanoDFT(mol, opts, profile_performance=True)

    start, end = ipu_cycles_stamps
    start = np.asarray(start)
    end = np.asarray(end)

    diff = (end - start)[0][0][0]
    print(
        "----------------------------------------------------------------------------"
    )
    print("                                Diff cycle count:", diff)
    print("                            Diff cycle count [M]:", diff / 1e6)
    print("Estimated time of execution on Bow-IPU [seconds]:", diff / (1.85 * 1e9))
    print(
        "----------------------------------------------------------------------------"
    )

    assert True
