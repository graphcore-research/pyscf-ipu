# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from typing import List

import chex
import numpy as np
from periodictable import elements
from py3Dmol import view
from pyscf import gto

from .types import FloatNx3, IntN
from .units import to_angstrom, to_bohr


@chex.dataclass
class Structure:
    atomic_number: IntN
    position: FloatNx3
    is_bohr: bool = True

    def __post_init__(self):
        if not self.is_bohr:
            self.position = to_bohr(self.position)

    @property
    def num_atoms(self) -> int:
        return len(self.atomic_number)

    @property
    def atomic_symbol(self) -> List[str]:
        return [elements[z].symbol for z in self.atomic_number]

    @property
    def num_electrons(self) -> int:
        return np.sum(self.atomic_number)

    def to_xyz(self) -> str:
        xyz = f"{self.num_atoms}\n\n"
        sym = self.atomic_symbol
        pos = to_angstrom(self.position)

        for i in range(self.num_atoms):
            r = np.array2string(pos[i, :], separator="\t")[1:-1]
            xyz += f"{sym[i]}\t{r}\n"

        return xyz

    def view(self) -> "view":
        return view(data=self.to_xyz(), style={"stick": {"radius": 0.06}})


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


def water():
    r"""Single water molecule
    Structure of single water molecule calculated with DFT using B3LYP
    functional and 6-31+G** basis set <https://cccbdb.nist.gov/>"""
    return Structure(
        atomic_number=np.array([8, 1, 1]),
        position=np.array(
            [
                [0.0000, 0.0000, 0.1165],
                [0.0000, 0.7694, -0.4661],
                [0.0000, -0.7694, -0.4661],
            ]
        ),
        is_bohr=False,
    )


def benzene():
    r"""Benzene ring
    Structure of benzene ring calculated with DFT using B3LYP functional
    and 6-31+G** basis set <https://cccbdb.nist.gov/>"""
    return Structure(
        atomic_number=np.array([6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1]),
        position=np.array(
            [
                [0.0000000, 1.3983460, 0.0000000],
                [1.2110030, 0.6991730, 0.0000000],
                [1.2110030, -0.6991730, 0.0000000],
                [0.0000000, -1.3983460, 0.0000000],
                [-1.211003, -0.699173, 0.0000000],
                [-1.211003, 0.6991730, 0.0000000],
                [0.0000000, 2.4847510, 0.0000000],
                [2.1518570, 1.2423750, 0.0000000],
                [2.1518570, -1.2423750, 0.0000000],
                [0.0000000, -2.4847510, 0.0000000],
                [-2.151857, -1.242375, 0.0000000],
                [-2.151857, 1.2423750, 0.0000000],
            ]
        ),
        is_bohr=False,
    )
