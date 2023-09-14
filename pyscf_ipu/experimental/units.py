# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from jaxtyping import Array

BOHR_PER_ANGSTROM = 0.529177210903


def to_angstrom(bohr_value: Array) -> Array:
    return bohr_value / BOHR_PER_ANGSTROM


def to_bohr(angstrom_value: Array) -> Array:
    return angstrom_value * BOHR_PER_ANGSTROM
