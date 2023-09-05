# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import os
from typing import Any, Dict, List, Tuple

import numpy as np
from jax import core

from .tile_array import tile_data_barrier
from .tile_interpreter import TileShardedArray, register_ipu_tile_primitive, tile_map
from .tile_interpreter_primitives import (
    IpuTileMapEquation,
    make_ipu_vertex_inout_info,
    make_ipu_vertex_name_templated,
    make_ipu_vertex_out_info,
)

# from jax._src.lax.control_flow.remat_impl import _optimization_barrier as optimization_barrier


hw_cycle_count_p = core.Primitive("hw_cycle_count")
hw_cycle_count_dtype = np.uint32


def hw_cycle_count(arg):
    return hw_cycle_count_p.bind(arg)


def hw_cycle_count_numpy_impl(arg):
    # Unsupported: zero cycle count.
    return arg, np.zeros((2,), dtype=hw_cycle_count_dtype)


def hw_cycle_count_abstract_eval(arg):
    assert isinstance(arg, core.ShapedArray)
    return arg, core.ShapedArray((2,), hw_cycle_count_dtype)


def hw_cycle_count_tile_translation_ipu(
    p: core.Primitive,
    tiles: Tuple[int, ...],
    inavals: List[core.ShapedArray],
    attributes: Dict[str, Any] = None,
) -> IpuTileMapEquation:
    """IPU tile translation for custom arange vertex.

    Args:
        p: JAX primitive.
        tiles: Collection of tiles.
        inavals: Input shaped arrays.
        attributes: Op attributes.
    Returns:
        IPU tile map primitive structure.
    """
    assert len(inavals) == 1
    inaval = inavals[0]
    _, outaval = hw_cycle_count_abstract_eval(inaval)

    vertex_name = make_ipu_vertex_name_templated("CycleCountBarrier", inaval.dtype)
    gp_filename = os.path.join(os.path.dirname(__file__), "vertex", "hw_vertex.cpp")

    # Translation rule to IPU vertex
    ipu_prim_info = IpuTileMapEquation(
        vname=vertex_name,
        pname=p.name,
        tiles=tiles,
        # IO vertex infos.
        inputs_info=[make_ipu_vertex_inout_info("data", inaval)],
        outputs_info=[make_ipu_vertex_inout_info("data", inaval), make_ipu_vertex_out_info("out", outaval)],
        # Perf. estimate from Poplar code.
        gp_filename=gp_filename,
        perf_estimate=50,
    )
    return ipu_prim_info


hw_cycle_count_p.map_primitive = False
hw_cycle_count_p.multiple_results = True
# Register the primal implementation with JAX
hw_cycle_count_p.def_impl(hw_cycle_count_numpy_impl)
# Register the abstract evaluation with JAX
hw_cycle_count_p.def_abstract_eval(hw_cycle_count_abstract_eval)
# Register tile IPU translation.
register_ipu_tile_primitive(hw_cycle_count_p, hw_cycle_count_tile_translation_ipu)


def ipu_cycle_count(*args: TileShardedArray, **kwargs: Any) -> Tuple[TileShardedArray, ...]:
    """Get IPU hardware cycle count, with arguments acting as barrier.

    Cycle counts are measured on the collection of tiles of the first argument.
    See XLA/MLIR optimization barrier for more information on the expected behaviour of a barrier.

    Args:
        *args: Tile arrays used as barrier before cycle count.
        sync: Should IPU tiles synced as well?
    Returns:
        (cycle count array, *args)
    """
    assert len(args) > 0
    assert all([isinstance(v, TileShardedArray) for v in args])
    sync = bool(kwargs.get("sync", False))
    # Tile barrier on input arrays, blocking until all are available/ready.
    if len(args) > 1:
        args = tile_data_barrier(*args)

    arg0, cycle_count = tile_map(hw_cycle_count_p, args[0], sync=sync)  # type:ignore
    # Re-pack the arguments + cycle count.
    return (arg0, *args[1:], cycle_count)


def ipu_cycle_count_overhead() -> int:
    """Overhead of measuring an IPU cycle count.

    Experimental value, measured using two successive calls to `ipu_cycle_count`
    """
    return 45


# Backward compatibility naming.
ipu_hw_cycle_count = ipu_cycle_count
