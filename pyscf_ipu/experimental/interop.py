# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from typing import Tuple

import numpy as np
from periodictable import elements
from pyscf import gto

from .basis import Basis, basisset
from .structure import Structure


def to_pyscf(
    structure: Structure, basis_name: str = "sto-3g", unit: str = "Bohr"
) -> "gto.Mole":
    mol = gto.Mole(unit=unit, spin=structure.num_electrons % 2, cart=True)
    mol.atom = [
        (symbol, pos)
        for symbol, pos in zip(structure.atomic_symbol, structure.position)
    ]
    mol.basis = basis_name
    mol.build(unit=unit)
    return mol


def from_pyscf(mol: "gto.Mole") -> Tuple[Structure, Basis]:
    atomic_number = []
    position = []

    for i in range(mol.natm):
        sym, pos = mol.atom[i]
        atomic_number.append(elements.symbol(sym).number)
        position.append(pos)

    structure = Structure(
        atomic_number=np.array(atomic_number),
        position=np.array(position),
        is_bohr=mol.unit != "Angstom",
    )

    basis = basisset(structure, basis_name=mol.basis)

    return structure, basis
