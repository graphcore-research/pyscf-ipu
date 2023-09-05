# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
# Register all additional libraries + paths.
import atexit

from . import external_libs
from .tile_array import (
    TileShardedArray,
    tile_barrier,
    tile_constant_replicated,
    tile_constant_sharded,
    tile_data_barrier,
    tile_gather,
    tile_put_replicated,
    tile_put_sharded,
)
from .tile_common_utils import from_ipu_type_to_numpy_dtype, make_ipu_shaped_array
from .tile_interpreter import (
    create_ipu_tile_primitive,
    create_ipu_tile_primitive_v2,
    declare_ipu_tile_primitive,
    get_ipu_tile_primitive_translation,
    register_ipu_tile_primitive,
    tile_map,
)
from .tile_interpreter_hw_primitives import (
    hw_cycle_count_p,
    ipu_cycle_count,
    ipu_cycle_count_overhead,
    ipu_hw_cycle_count,
)
from .tile_interpreter_primitives import (
    IpuTileMapEquation,
    IpuVertexIOType,
    from_numpy_dtype_to_ipu_type,
    get_ipu_type_name,
    make_ipu_vertex_attributes,
    make_ipu_vertex_constant_info,
    make_ipu_vertex_in_info,
    make_ipu_vertex_inout_info,
    make_ipu_vertex_inputs,
    make_ipu_vertex_io_info,
    make_ipu_vertex_name_templated,
    make_ipu_vertex_out_info,
    make_ipu_vertex_outputs,
    primitive_clone,
    primitive_num_inout_alias_args,
)


def tessellate_ipu_cleanup():
    """Nanobind is quite picky in terms of checking no reference gets leaked at shutdown.

    This method is called at Python exit, trying to deal with existing issue in typing LRU cache:
        https://github.com/wjakob/nanobind/issues/19
        https://github.com/python/cpython/issues/98253

    Additionally, it seems that JAX MLIR register is also not properly cleaning references, leading
    to additional reference leaks.
    """
    import typing

    from jax.interpreters.mlir import _lowerings, _platform_specific_lowerings

    # Clear JAX MLIR lowering register.
    _lowerings.clear()
    _platform_specific_lowerings.clear()
    # Typing extension manual cleanup.
    for cleanup in typing._cleanups:  # type: ignore
        cleanup()


atexit.register(tessellate_ipu_cleanup)
