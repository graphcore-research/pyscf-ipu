# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from typing import List

import chex
import numpy as np
from periodictable import elements
from py3Dmol import view

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

        # single atom case
        self.position = np.atleast_2d(self.position)

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


def molecule(name: str):
    name = name.lower()

    if name == "h2":
        return Structure(
            atomic_number=np.array([1, 1]),
            position=np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]]),
        )

    if name == "water":
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

    raise NotImplementedError(f"No structure registered for: {name}")
