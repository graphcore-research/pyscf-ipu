# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import base64
import json
import os
from typing import Any, Dict, Sequence, Tuple, Union

import jax.lax
import jax.numpy as jnp
import numpy as np
from jax import core
from jax.core import ShapedArray
from jax.interpreters import mlir
from jax.interpreters.mlir import LoweringRuleContext, ir, mhlo
from jax.ipu.primitive import ipu_mlir_lowering_custom_primitive

from tessellate_ipu.utils import DTypeLike, NDArray

tile_put_sharded_prim_p = core.Primitive("tile_put_sharded")
tile_put_replicated_prim_p = core.Primitive("tile_put_replicated")
tile_gather_prim_p = core.Primitive("tile_gather")
tile_data_barrier_prim_p = core.Primitive("tile_data_barrier")
tile_constant_replicated_prim_p = core.Primitive("tile_constant_replicated")
tile_constant_sharded_prim_p = core.Primitive("tile_constant_sharded")

default_backends = ["cpu", "cuda", "tpu", "rocm"]


def make_tiles_raw_attributes(tiles: Tuple[int, ...]) -> str:
    """Make raw JSON attributes corresponding to a collection of tiles."""
    tiles_json = json.dumps(tuple(tiles))
    return tiles_json


def tile_put_sharded_prim(x, tiles):
    return tile_put_sharded_prim_p.bind(x, tiles=tiles)


def tile_put_sharded_prim_impl(x, tiles):
    # No-op when not jitted.
    assert x.shape[0] == len(tiles)
    return x


def tile_put_sharded_prim_abstract_eval(xs, tiles) -> ShapedArray:
    assert xs.shape[0] == len(tiles)
    return xs


def tile_put_sharded_prim_mlir_lowering_default(ctx, xc, tiles):
    """`tile_put_sharded_prim` default MLIR lowering, for CPU/GPU backends: no-op"""
    return (xc,)


def tile_put_sharded_prim_mlir_lowering_ipu(ctx: LoweringRuleContext, xc: ir.Value, tiles: Any) -> Sequence[ir.Value]:
    """`tile_put_sharded_prim` IPU backend MLIR lowering, as a custom primitive."""
    from tessellate_ipu.lib.pytessellate_ipu_ops_jax import TilePutShardedPrimitive

    inputs = [xc]
    # Passing the tiles collections as a raw attributes to the C++ implementation.
    raw_attributes = make_tiles_raw_attributes(tiles)
    # TODO: Add Github permanent link to C++.
    outputs = ipu_mlir_lowering_custom_primitive(TilePutShardedPrimitive, ctx, inputs, opaque_attributes=raw_attributes)
    return outputs


# Register the primal implementation with JAX
tile_put_sharded_prim_p.def_impl(tile_put_sharded_prim_impl)
# Register the abstract evaluation with JAX
tile_put_sharded_prim_p.def_abstract_eval(tile_put_sharded_prim_abstract_eval)
# Register specific MLIR lowering for IPU.
mlir.register_lowering(tile_put_sharded_prim_p, tile_put_sharded_prim_mlir_lowering_ipu, platform="ipu")
# Register MLIR lowering, for other backends.
mlir.register_lowering(tile_put_sharded_prim_p, tile_put_sharded_prim_mlir_lowering_default)


def tile_put_replicated_prim(x, tiles):
    return tile_put_replicated_prim_p.bind(x, tiles=tiles)


def tile_put_replicated_prim_impl(x, tiles):
    return np.stack([x for _ in range(len(tiles))], axis=0)


def tile_put_replicated_prim_abstract_eval(xs, tiles) -> ShapedArray:
    outshape = (len(tiles), *xs.shape)
    return ShapedArray(outshape, xs.dtype, xs.weak_type)


def tile_put_replicated_prim_mlir_translation_default(
    ctx: LoweringRuleContext, *args: Union[ir.Value, Sequence[ir.Value]], **params
) -> Sequence[Union[ir.Value, Sequence[ir.Value]]]:
    """`tile_put_replicated_prim` default MLIR translation, for CPU/GPU backends: simple concat."""
    tiles = params["tiles"]

    # Not sure using a local function is a good idea?
    def tile_replicated_fn(input):
        N = len(tiles)
        input = jax.lax.expand_dims(input, dimensions=(0,))
        return jax.lax.concatenate([input] * N, dimension=0)

    # Lower to MLIR using JAX tooling. TODO: cache lowering?
    tile_replicated_lower_fn = mlir.lower_fun(tile_replicated_fn, multiple_results=False)
    return tile_replicated_lower_fn(ctx, *args)


def tile_put_replicated_prim_mlir_lowering_ipu(
    ctx: LoweringRuleContext, xc: ir.Value, tiles: Any
) -> Sequence[ir.Value]:
    """`tile_put_replicated_prim` IPU backend MLIR lowering, as a custom primitive."""
    from tessellate_ipu.lib.pytessellate_ipu_ops_jax import TilePutReplicatedPrimitive

    inputs = [xc]
    # Passing the tiles collections as a raw attributes to the C++ implementation.
    raw_attributes = make_tiles_raw_attributes(tiles)
    # outputs_aval = [tile_put_replicated_prim_abstract_eval(xla_shape_to_aval(ctx.get_shape(xc)), tiles)]
    outputs = ipu_mlir_lowering_custom_primitive(
        TilePutReplicatedPrimitive, ctx, inputs, opaque_attributes=raw_attributes
    )
    return outputs


# Register the primal implementation with JAX
tile_put_replicated_prim_p.def_impl(tile_put_replicated_prim_impl)
# Register the abstract evaluation with JAX
tile_put_replicated_prim_p.def_abstract_eval(tile_put_replicated_prim_abstract_eval)
# Register specific MLIR lowering for IPU.
mlir.register_lowering(tile_put_replicated_prim_p, tile_put_replicated_prim_mlir_lowering_ipu, platform="ipu")
# Register MLIR translation for other backends.
mlir.register_lowering(tile_put_replicated_prim_p, tile_put_replicated_prim_mlir_translation_default)


def tile_gather_prim(x, previous_tiles, indices, tiles):
    return tile_gather_prim_p.bind(x, previous_tiles=previous_tiles, indices=indices, tiles=tiles)


def tile_gather_prim_impl(x, previous_tiles, indices, tiles):
    # NumPy basic gather on axis=0
    return x[list(indices)]


def tile_gather_prim_abstract_eval(xs, previous_tiles, indices, tiles) -> ShapedArray:
    item_shape = xs.shape[1:]
    outshape = (len(tiles), *item_shape)
    return ShapedArray(outshape, xs.dtype, xs.weak_type)


def tile_gather_prim_mlir_lowering_default(ctx, xc, previous_tiles, indices, tiles):
    """`tile_gather_prim` default MLIR lowering, for CPU/GPU backends: simple JAX static gather"""
    # TODO: implementation from JAX?
    raise NotImplementedError()


def tile_gather_prim_mlir_lowering_ipu(
    ctx: LoweringRuleContext, xc: ir.Value, previous_tiles: Any, indices: Any, tiles: Any
) -> Sequence[ir.Value]:
    """`tile_gather_prim` IPU backend MLIR lowering, as a custom primitive."""
    # FIXME: Local imports to deal with JAX/typing(?) leaked references.
    from tessellate_ipu.lib.pytessellate_ipu_core import TileGatherParams
    from tessellate_ipu.lib.pytessellate_ipu_ops_jax import TileGatherPrimitive

    inputs = [xc]
    # Til gather parameters, to pass to the XLA/HLO op.
    gather_params = TileGatherParams(previous_tiles, indices, tiles)
    raw_attributes = gather_params.to_json_str()
    outputs = ipu_mlir_lowering_custom_primitive(TileGatherPrimitive, ctx, inputs, opaque_attributes=raw_attributes)
    return outputs


# Register the primal implementation with JAX
tile_gather_prim_p.def_impl(tile_gather_prim_impl)
# Register the abstract evaluation with JAX
tile_gather_prim_p.def_abstract_eval(tile_gather_prim_abstract_eval)
# Register specific MLIR lowering for IPU.
mlir.register_lowering(tile_gather_prim_p, tile_gather_prim_mlir_lowering_ipu, platform="ipu")
# Register MLIR lowering, for other backends.
mlir.register_lowering(tile_gather_prim_p, tile_gather_prim_mlir_lowering_default)


def tile_data_barrier_prim(inputs, inputs_tiles):
    return tile_data_barrier_prim_p.bind(*inputs, inputs_tiles=list(inputs_tiles))


def tile_data_barrier_prim_impl(*args, **params):
    return tuple(args)


def tile_data_barrier_prim_abstract_eval(*args: ShapedArray, **params) -> Tuple[ShapedArray, ...]:
    return args


def tile_data_barrier_prim_mlir_lowering_default(ctx, *args, **params):
    """`tile_data_barrier_prim` default MLIR lowering, for CPU/GPU backends."""
    # Translate into standard optimization barrier on CPU/GPU/TPU.
    return mhlo.OptimizationBarrierOp([v.type for v in args], args).results


_tile_barrier_dtype_mapping: Dict[DTypeLike, DTypeLike] = {
    np.dtype(np.int8): np.dtype(np.uint8),
    np.dtype(np.uint8): np.dtype(np.uint8),
    np.dtype(np.int16): np.dtype(np.uint16),
    np.dtype(np.uint16): np.dtype(np.uint16),
    np.dtype(np.float16): np.dtype(np.uint16),
    np.dtype(np.int32): np.dtype(np.uint32),
    np.dtype(np.uint32): np.dtype(np.uint32),
    np.dtype(np.float32): np.dtype(np.uint32),
}


def tile_data_barrier_refdtype(dtype: DTypeLike, is_half_accurate: bool) -> DTypeLike:
    """Find the reference dtype to use in IPU tile data barrier."""
    if not is_half_accurate and dtype == np.dtype(np.float16):
        # Half type specific case on IPU model => need to keep FP16.
        return dtype
    return _tile_barrier_dtype_mapping[dtype]


def tile_data_barrier_prim_mlir_lowering_ipu(
    ctx: LoweringRuleContext, *args: ir.Value, **params: Any
) -> Sequence[ir.Value]:
    """`tile_data_barrier_prim` IPU backend MLIR lowering, as a custom primitive."""
    # FIXME: Local imports to deal with JAX/typing(?) leaked references.
    from tessellate_ipu.lib.pytessellate_ipu_core import TileDataBarrierParams
    from tessellate_ipu.lib.pytessellate_ipu_ops_jax import TileDataBarrierPrimitive

    from .tile_interpreter_primitives import make_ipu_vertex_name_templated

    inputs = list(args)
    inputs_aval = ctx.avals_in
    dtypes = list({aval.dtype for aval in inputs_aval})
    dtypes_size = {dt.itemsize for dt in dtypes}
    if len(dtypes_size) > 1:
        raise TypeError(f"Only supporting dtypes of same size in Tile data barrier: {dtypes}.")

    inputs_tiles = params["inputs_tiles"]
    max_tile = max([max(s) for s in inputs_tiles])
    # Is half type accurate on IPU? IPU model is simulating float.
    # TODO: have specific property in IPU device.
    is_half_accurate = not jax.devices("ipu")[0].is_ipu_model
    # Passing the tiles collections as a raw attributes to the C++ implementation.
    refdtype = tile_data_barrier_refdtype(dtypes[0], is_half_accurate)
    vname = make_ipu_vertex_name_templated("TileDataBarrierVertex", refdtype)
    barrier_params = TileDataBarrierParams(vname, inputs_tiles, max_tile)
    raw_attributes = barrier_params.to_json_str()

    gp_filename = os.path.abspath(os.path.join(os.path.dirname(__file__), "vertex", "tile_prim_vertex.cpp"))
    outputs = ipu_mlir_lowering_custom_primitive(
        TileDataBarrierPrimitive,
        ctx,
        inputs,
        opaque_attributes=raw_attributes,
        ipu_gp_filename=gp_filename,
    )
    return outputs


tile_data_barrier_prim_p.multiple_results = True
# Register the primal implementation with JAX
tile_data_barrier_prim_p.def_impl(tile_data_barrier_prim_impl)
# Register the abstract evaluation with JAX
tile_data_barrier_prim_p.def_abstract_eval(tile_data_barrier_prim_abstract_eval)
# Register specific MLIR lowering for IPU.
mlir.register_lowering(tile_data_barrier_prim_p, tile_data_barrier_prim_mlir_lowering_ipu, platform="ipu")
# Register MLIR lowering, for other backends.
mlir.register_lowering(tile_data_barrier_prim_p, tile_data_barrier_prim_mlir_lowering_default)


def tile_constant_replicated_prim(data, tiles):
    # Dummy empty variable to circumvent a bug in XLA custom op (when zero inputs).
    dummy = jnp.empty((), np.float32)
    assert isinstance(data, np.ndarray)
    return tile_constant_replicated_prim_p.bind(dummy, data=data, tiles=tiles)


def tile_constant_replicated_prim_impl(dummy, data: NDArray[Any], tiles: Any) -> NDArray[Any]:
    return np.stack([data for _ in range(len(tiles))], axis=0)


def tile_constant_replicated_prim_abstract_eval(dummy, data, tiles) -> ShapedArray:
    outshape = (len(tiles), *data.shape)
    return ShapedArray(outshape, data.dtype)


def tile_constant_replicated_prim_mlir_translation_default(
    ctx: LoweringRuleContext, *args: Union[ir.Value, Sequence[ir.Value]], **params
) -> Sequence[Union[ir.Value, Sequence[ir.Value]]]:
    """`tile_constant_replicated_prim` default MLIR translation, for CPU/GPU backends: simple constant."""
    # TODO: fix dummy input requirement.
    assert len(args) == 1
    data = params["data"]
    tiles = params["tiles"]
    replicated_data = tile_constant_replicated_prim_impl(args[0], data=data, tiles=tiles)
    # MLIR constant from the replicated NumPy array.
    return mlir._ndarray_constant_handler(replicated_data, canonicalize_types=False)


def tile_constant_replicated_prim_mlir_lowering_ipu(
    ctx: LoweringRuleContext, dummy: ir.Value, data: NDArray[Any], tiles: Any
) -> Sequence[ir.Value]:
    """`tile_constant_replicated_prim` IPU backend MLIR lowering, as a custom primitive."""
    # FIXME: Local imports to deal with JAX/typing(?) leaked references.
    from tessellate_ipu.lib.pytessellate_ipu_core import Base64Data, TileConstantParams
    from tessellate_ipu.lib.pytessellate_ipu_ops_jax import TileConstantReplicatedPrimitive

    from .tile_common_utils import make_ipu_shaped_array

    params = TileConstantParams(
        aval=make_ipu_shaped_array(data.shape, data.dtype),
        tiles=tiles,
        data=Base64Data(base64.b64encode(data)),  # type:ignore
    )
    # TODO: remove `dummy` when bug with zero inputs fixed.
    outputs = ipu_mlir_lowering_custom_primitive(
        TileConstantReplicatedPrimitive, ctx, [dummy], opaque_attributes=params.to_json_str()
    )
    return outputs


# Primal + abstract evaluation same as `tile_put_replicated_prim`
tile_constant_replicated_prim_p.def_impl(tile_constant_replicated_prim_impl)
tile_constant_replicated_prim_p.def_abstract_eval(tile_constant_replicated_prim_abstract_eval)
# Register specific MLIR lowering for IPU.
mlir.register_lowering(tile_constant_replicated_prim_p, tile_constant_replicated_prim_mlir_lowering_ipu, platform="ipu")
# Register MLIR translation for other backends.
mlir.register_lowering(tile_constant_replicated_prim_p, tile_constant_replicated_prim_mlir_translation_default)


def tile_constant_sharded_prim(data, tiles):
    # Dummy empty variable to circumvent a bug in XLA custom op (when zero inputs).
    dummy = jnp.empty((), np.float32)
    assert isinstance(data, np.ndarray)
    return tile_constant_sharded_prim_p.bind(dummy, data=data, tiles=tiles)


def tile_constant_sharded_prim_impl(dummy, data: NDArray[Any], tiles: Any) -> NDArray[Any]:
    return data


def tile_constant_sharded_prim_abstract_eval(dummy, data, tiles) -> ShapedArray:
    return ShapedArray(data.shape, data.dtype)


def tile_constant_sharded_prim_mlir_translation_default(
    ctx: LoweringRuleContext, *args: Union[ir.Value, Sequence[ir.Value]], **params
) -> Sequence[Union[ir.Value, Sequence[ir.Value]]]:
    """`tile_constant_sharded_prim` default MLIR translation, for CPU/GPU backends: simple constant."""
    # TODO: fix dummy input requirement.
    assert len(args) == 1
    data = params["data"]
    return mlir._ndarray_constant_handler(data, canonicalize_types=False)


def tile_constant_sharded_prim_mlir_lowering_ipu(
    ctx: LoweringRuleContext, dummy: ir.Value, data: NDArray[Any], tiles: Any
) -> Sequence[ir.Value]:
    """`tile_constant_sharded_prim` IPU backend MLIR lowering, as a custom primitive."""
    # FIXME: Local imports to deal with JAX/typing(?) leaked references.
    from tessellate_ipu.lib.pytessellate_ipu_core import Base64Data, TileConstantParams
    from tessellate_ipu.lib.pytessellate_ipu_ops_jax import TileConstantShardedPrimitive

    from .tile_common_utils import make_ipu_shaped_array

    params = TileConstantParams(
        aval=make_ipu_shaped_array(data.shape, data.dtype),
        tiles=tiles,
        data=Base64Data(base64.b64encode(data)),  # type:ignore
    )
    # TODO: remove `dummy` when bug with zero inputs fixed.
    outputs = ipu_mlir_lowering_custom_primitive(
        TileConstantShardedPrimitive, ctx, [dummy], opaque_attributes=params.to_json_str()
    )
    return outputs


# Primal + abstract evaluation same as `tile_put_replicated_prim`
tile_constant_sharded_prim_p.def_impl(tile_constant_sharded_prim_impl)
tile_constant_sharded_prim_p.def_abstract_eval(tile_constant_sharded_prim_abstract_eval)
# Register specific MLIR lowering for IPU.
mlir.register_lowering(tile_constant_sharded_prim_p, tile_constant_sharded_prim_mlir_lowering_ipu, platform="ipu")
# Register MLIR translation for other backends.
mlir.register_lowering(tile_constant_sharded_prim_p, tile_constant_sharded_prim_mlir_translation_default)
