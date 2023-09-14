# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from typing import Tuple

import chex
import jax.numpy as jnp

from .orbital import Orbital
from .structure import Structure
from .types import FloatN, FloatNx3, FloatNxM


@chex.dataclass
class Basis:
    orbitals: Tuple[Orbital]
    structure: Structure

    @property
    def num_orbitals(self) -> int:
        return len(self.orbitals)

    @property
    def num_primitives(self) -> int:
        return sum(ao.num_primitives for ao in self.orbitals)

    @property
    def occupancy(self) -> FloatN:
        # Assumes uncharged systems in restricted Kohn-Sham
        occ = jnp.full(self.num_orbitals, 2.0)
        mask = occ.cumsum() > self.structure.num_electrons
        occ = occ.at[mask].set(0.0)
        return occ

    def __call__(self, pos: FloatNx3) -> FloatNxM:
        return jnp.hstack([o(pos) for o in self.orbitals])


def basisset(structure: Structure, basis_name: str = "sto-3g"):
    from basis_set_exchange import get_basis
    from basis_set_exchange.sort import sort_basis

    LMN_MAP = {
        0: [(0, 0, 0)],
        1: [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
        2: [(2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2)],
    }

    bse_basis = get_basis(
        basis_name,
        elements=structure.atomic_symbol,
        uncontract_spdf=True,
        uncontract_general=False,
    )
    bse_basis = sort_basis(bse_basis)["elements"]
    orbitals = []

    for a in range(structure.num_atoms):
        center = structure.position[a, :]
        shells = bse_basis[str(structure.atomic_number[a])]["electron_shells"]

        for s in shells:
            for lmn in LMN_MAP[s["angular_momentum"][0]]:
                ao = Orbital.from_bse(
                    center=center,
                    alphas=jnp.array(s["exponents"], dtype=float),
                    lmn=jnp.array(lmn, dtype=jnp.int32),
                    coefficients=jnp.array(s["coefficients"], dtype=float),
                )
                orbitals.append(ao)

    return Basis(
        orbitals=orbitals,
        structure=structure,
    )
