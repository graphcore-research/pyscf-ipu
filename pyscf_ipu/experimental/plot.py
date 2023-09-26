# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import numpy as np
from numpy.typing import NDArray

from .structure import Structure
from .types import MeshAxes
from .units import to_angstrom


def plot_volume(structure: Structure, value: NDArray, axes: MeshAxes):
    """plots volumetric data value with molecular structure.

    Args:
        structure (Structure): molecular structure
        value (NDArray): the volume data to render
        axes (MeshAxes): the axes over which the data was sampled.

    Returns:
        py3DMol View object
    """
    v = structure.view()
    v.addVolumetricData(cube_data(value, axes), "cube", build_transferfn(value))
    return v


def cube_data(value: NDArray, axes: MeshAxes) -> str:
    """Generate the cube file format as a string.  See:

      https://paulbourke.net/dataformats/cube/

    Args:
        value (NDArray): the volume data to serialise in the cube format
        axes (MeshAxes): the axes over which the data was sampled

    Returns:
        str: cube format representation of the volumetric data.
    """
    axes = [to_angstrom(ax) for ax in axes]
    fmt = "cube format\n\n"
    x, y, z = axes
    nx, ny, nz = [ax.shape[0] for ax in axes]
    fmt += "0 " + " ".join([f"{v:12.6f}" for v in [x[0], y[0], z[0]]]) + "\n"
    fmt += f"{nx} " + " ".join([f"{v:12.6f}" for v in [x[1] - x[0], 0.0, 0.0]]) + "\n"
    fmt += f"{ny} " + " ".join([f"{v:12.6f}" for v in [0.0, y[1] - y[0], 0.0]]) + "\n"
    fmt += f"{nz} " + " ".join([f"{v:12.6f}" for v in [0.0, 0.0, z[1] - z[0]]]) + "\n"

    line = ""
    for i in range(len(value)):
        line += f"{value[i]:12.6f}"

        if i % 6 == 0:
            fmt += line + "\n"
            line = ""

    return fmt


def build_transferfn(value: NDArray) -> dict:
    """Generate the 3dmol.js transferfn argument for a particular value.

    Tries to set isovalues to capture main features of the volume data.

    Args:
        value (NDArray): the volume data.

    Returns:
        dict: containing transferfn
    """
    v = np.percentile(value, [99.9, 75])
    a = [0.02, 0.0005]
    return {
        "transferfn": [
            {"color": "blue", "opacity": a[0], "value": -v[0]},
            {"color": "blue", "opacity": a[1], "value": -v[1]},
            {"color": "white", "opacity": 0.0, "value": 0.0},
            {"color": "red", "opacity": a[1], "value": v[1]},
            {"color": "red", "opacity": a[0], "value": v[0]},
        ]
    }
