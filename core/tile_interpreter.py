# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
"""Building IPU tile MPMD programming as a custom JAX interpreter (https://github.com/google/jax/tree/main/jax/interpreters).

In particular, we need a registry mapping JAX primitives to IPU vertex (and additionally support custom IPU vertex).
"""
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from jax.core import Primitive, ShapedArray

from .tile_array import TileShardedArray
from .tile_common_utils import make_ipu_shaped_array
from .tile_interpreter_primitives import (
    IpuTileMapEquation,
    IpuVertexIOType,
    from_numpy_dtype_to_ipu_type,
    make_ipu_vertex_attributes,
    make_ipu_vertex_constant_info,
    make_ipu_vertex_io_info,
    tile_map_equation_call_multi_out,
    tile_map_equation_call_single_out,
)

IpuVertexTranslation = Callable[
    [Primitive, Tuple[int, ...], List[ShapedArray], Optional[Dict[str, Any]]], IpuTileMapEquation
]
"""Ipu vertex translation: callable translating a JAX primitive (with inputs/outputs) into a full
vertex info data structure.
"""

IpuVertexConstantFactory = Callable[
    [Sequence[ShapedArray], Sequence[ShapedArray], Optional[Dict[str, Any]]], np.ndarray
]
"""Ipu vertex constant factory method: can build the constant from inavals, outavals and attributes.
"""

_ipu_tile_primitive_registry: Dict[str, Tuple[Primitive, IpuVertexTranslation]] = {}
"""Global registry mapping JAX primitives to IPU vertex translation rules.

The registry is indexed by the primitive name.
"""


def check_tile_mapping_consistency(args: Sequence[TileShardedArray]):
    if len(args) == 0:
        return
    t0: TileShardedArray = args[0]
    for t1 in args[1:]:
        if t0.tiles != t1.tiles:
            raise ValueError(f"Inconsistent tile mapping between input arrays: {t0.tiles} vs {t1.tiles}.")


def check_in_out_arguments(inputs: Sequence[str], outputs: Sequence[str]):
    """Check in/out arguments validity, i.e. these should be first in the inputs and outputs list and
    1-to-1 equivalents.

    Args:
        inputs: Sequence of input names.
        outputs: Sequence of output names.
    """
    inout_names = {v for v in inputs if v in outputs}
    inout_map0 = {v: inputs.index(v) for v in inout_names}
    inout_map1 = {v: outputs.index(v) for v in inout_names}
    # Check 1-to-1 mapping of in/out arguments.
    for v in inout_names:
        if inout_map0[v] != inout_map1[v]:
            raise IndexError(f"Inconsistent TessellateIPU inplace in/out indices: {inout_map0[v]} vs {inout_map1[v]}.")
    # Check it corresponds to the first arguments.
    inout_idxes = tuple(sorted(inout_map0.values()))
    if inout_idxes != tuple(range(len(inout_idxes))):
        raise ValueError(f"In/out arguments should be at the first vertex arguments (here: {inout_idxes}).")


# This is moved up here temporarily to get a clearer diff between old and new below.
# TODO: move near create_ipu_tile_primitive_v2
def create_ipu_tile_primitive(
    pname: str,
    vname: str,
    inputs: List[str],
    outputs: Dict[str, int],
    constants: Optional[Dict[str, IpuVertexConstantFactory]] = None,
    tmp_space: Optional[Union[int, ShapedArray]] = None,
    gp_filename: Optional[str] = None,
    perf_estimate: int = 0,
) -> Primitive:
    """Create a simple IPU-JAX tile primitive directy mapping to an IPU vertex (from
    the official SDK, or custom implementation).

    Factory method helping creating tile primitives in the simple cases (i.e.
    output shape corresponding to an input, ...)

    Args:
        pname: Primitive name.
        vname: Vertex name. Supporting templated dtype from input(s).
        inputs: Set of input names.
        outputs: Set output names (with input index for aval).
        constants: Vertex constants factory function: (inavals, outavals, attrs) -> np.ndarray
        tmp_space: Optional tmp space. Either index refering an input array, or a static shaped array.
        gp_filename: Optional IPU gp filename.
        perf_estimate: Optional performance estimate.
    Returns:
        JAX primitive, suitable for IPU tile mapping.
    """
    p = Primitive(pname)
    p.map_primitive = False  # What for ???
    p.multiple_results = len(outputs) > 1
    num_inputs = len(inputs)
    constants_fn = constants or {}

    # InOut entries.
    inout_names = {v for v in inputs if v in outputs}

    def get_iotype(name: str, default: IpuVertexIOType) -> IpuVertexIOType:
        return IpuVertexIOType.InOut if name in inout_names else default

    # Build inputs/outputs vertex IO type.
    inputs_iotype = {v: get_iotype(v, IpuVertexIOType.In) for v in inputs}
    outputs_iotype = {v: get_iotype(v, IpuVertexIOType.Out) for v in outputs.keys()}

    def p_abstract_aval(*args, **kwargs):
        def _get_output_aval(outinfo):
            if isinstance(outinfo, int):
                return args[outinfo]
            elif isinstance(outinfo, ShapedArray):
                return outinfo
            raise ValueError(f"Unknown IPU vertex output descriptor: {outinfo}.")

        assert len(args) == num_inputs
        out_avals = [_get_output_aval(idx) for idx in outputs.values()]
        return tuple(out_avals) if p.multiple_results else out_avals[0]

    def p_tile_translation_ipu(
        p: Primitive,
        tiles: Tuple[int, ...],
        inavals: List[ShapedArray],
        attributes: Dict[str, Any] = None,
    ) -> IpuTileMapEquation:
        """IPU tile translation for custom vertex."""
        assert len(inavals) == len(inputs)
        outavals = p_abstract_aval(*inavals)
        if not p.multiple_results:
            outavals = (outavals,)

        inavals_dict = {inname: inaval for inname, inaval in zip(inputs, inavals)}
        # Generate vertex fullname, using templated dtypes.
        vname_used_inputs = [v for v in inputs if f"{{{v}}}" in vname]
        vname_dtype_inputs = {
            v: from_numpy_dtype_to_ipu_type(inavals_dict[v].dtype).name.lower() for v in vname_used_inputs
        }
        vertex_fullname = vname.format(**vname_dtype_inputs)

        # IO infos.
        inputs_info = [
            make_ipu_vertex_io_info(inname, inputs_iotype[inname], inaval) for inname, inaval in zip(inputs, inavals)
        ]
        outputs_info = [
            make_ipu_vertex_io_info(outname, outputs_iotype[outname], outaval)
            for outname, outaval in zip(outputs.keys(), outavals)
        ]
        constants_info = [
            make_ipu_vertex_constant_info(cname, fn(inavals, outavals, attributes), vertex_dim2=0)
            for cname, fn in constants_fn.items()
        ]
        # Pass attributes to the vertex.
        attributes = {} if attributes is None else attributes
        attrs_i32, attrs_f32 = make_ipu_vertex_attributes(**attributes)
        ipu_prim_info = IpuTileMapEquation(
            vname=vertex_fullname,
            pname=p.name,
            tiles=tiles,
            # IO vertex infos.
            inputs_info=inputs_info + constants_info,
            outputs_info=outputs_info,
            # Additional attributes to pass to the vertex
            attributes_i32=attrs_i32,
            attributes_f32=attrs_f32,
            # Optional GP filename and perf. estimate.
            gp_filename=gp_filename,
            perf_estimate=perf_estimate,
        )
        if tmp_space is not None:
            # Temporary scratch space to use by the vertex (zero=unused).
            tmp_inaval = None
            if isinstance(tmp_space, int):
                tmp_inaval = inavals[tmp_space]
            elif isinstance(tmp_space, ShapedArray):
                tmp_inaval = tmp_space
            else:
                raise ValueError(f"Unknown IPU vertex primitive tmp space: '{tmp_space}'.")
            ipu_prim_info.tmp_space_name = "tmp"
            ipu_prim_info.tmp_space_aval = make_ipu_shaped_array(tmp_inaval.shape, tmp_inaval.dtype)
        return ipu_prim_info

    # Register the abstract evaluation with JAX
    p.def_abstract_eval(p_abstract_aval)
    # Register tile IPU translation.
    register_ipu_tile_primitive(p, p_tile_translation_ipu)
    return p


def tile_map(
    primitive: Primitive, *args: TileShardedArray, **kwargs: Any
) -> Union[TileShardedArray, Sequence[TileShardedArray]]:
    """Map a JAX primitive over tiles.

    Args:
        primitive: JAX primitive to map.
        *args: List of input (tile) sharded arrays.
        **kwargs: Attributes to pass to the JAX primitive (and translation rule).
            tiles: Optional tile mapping, provided when there is no input.
            sync: Synchronize tiles before the Poplar compute set.
    Returns:
        List of output sharded arrays.
    """
    # Unpack arguments...
    inputs: List[TileShardedArray] = list(args)
    assert all([isinstance(v, TileShardedArray) for v in args])

    # Tiles & sync arguments.
    tiles: Optional[Tuple[int, ...]] = kwargs.get("tiles", None)
    sync: bool = kwargs.get("sync", False)
    # Remove IPU arguments.
    attributes = dict(kwargs)
    attributes.pop("tiles", None)
    attributes.pop("sync", None)

    if primitive is None:
        # No primitive: by default a no-op.
        return tuple(inputs)
    if primitive.name not in _ipu_tile_primitive_registry:
        raise KeyError(f"The JAX primitive `{primitive}` is not supported for tile mapping on the IPU.")
    if not all([isinstance(v, TileShardedArray) for v in inputs]):
        raise TypeError("Tile map inputs must be `TileShardedArray` instances.")

    # TODO: check tile mapping consistency.
    check_tile_mapping_consistency(inputs)
    tiles = tiles or inputs[0].tiles
    attributes = attributes or {}
    # Get the IPU tile map equation corresponding.
    _, ipu_prim_translation = _ipu_tile_primitive_registry[primitive.name]
    # TODO: pass outavals as well => no need to do it manually in every translation function.
    tile_map_eqn: IpuTileMapEquation = ipu_prim_translation(primitive, tiles, [v.tile_aval for v in inputs], attributes)
    tile_map_eqn.sync = sync
    tile_map_eqn_json: str = tile_map_eqn.to_json_str()

    # Call JAX tile custom primitive, dispatching properly the equation call.
    # And then convert to proper TileShardedArray
    if primitive.multiple_results:
        outputs = tile_map_equation_call_multi_out(
            [v.device_array for v in inputs],
            pname=primitive.name,
            tiles=tiles,
            tile_map_eqn_json=tile_map_eqn_json,
            **attributes,
        )
        return tuple([TileShardedArray(v, tiles) for v in outputs])
    else:
        output = tile_map_equation_call_single_out(
            [v.device_array for v in inputs],
            pname=primitive.name,
            tiles=tiles,
            tile_map_eqn_json=tile_map_eqn_json,
            **attributes,
        )
        return TileShardedArray(output, tiles)


def register_ipu_tile_primitive(primitive: Primitive, translation: IpuVertexTranslation):
    """Register an IPU tile vertex translation from JAX primitive.

    Args:
        primitive: JAX primitive.
        translation: IPU vertex translation rule.
    """
    global _ipu_tile_primitive_registry
    _ipu_tile_primitive_registry[primitive.name] = (primitive, translation)


def get_ipu_tile_primitive_translation(pname: str) -> Tuple[Primitive, IpuVertexTranslation]:
    """Get the primitive and IPU translation corresponding to a primitive name."""
    return _ipu_tile_primitive_registry[pname]


def create_ipu_tile_primitive_v2(
    pname: str, vname: str, initializer_fn, gp_filename: Optional[str] = None
) -> Primitive:
    """Create a simple IPU-JAX tile primitive directly mapping to an IPU vertex (from
    the official SDK, or custom implementation).

    Args:
        pname: Primitive name.
        vname: Vertex name. Supporting templated dtype from input(s).
        initializer_fn:
          Is called during JAX tracing,  its input parameters should
          have identical names to the Vertex inputs (in .cpp file)
          and should return dicts of ShapedArrays for:
           outputs, constants, and temporary storage
        gp_filename: Optional IPU cpp/gp filename
    Returns:
        JAX primitive, suitable for IPU tile mapping.
    """
    p = Primitive(pname)
    p.map_primitive = False  # What for ???
    p.multiple_results = None  # Set later
    #
    # TODO: Set inputs from initializer_fn.__code__.args
    num_inputs = initializer_fn.__code__.co_argcount
    inputs = initializer_fn.__code__.co_varnames[:num_inputs]

    def p_abstract_aval(*args, **kwargs):
        assert len(args) == num_inputs
        outputs, _, _, _ = initializer_fn(*args)
        # Always set primitive `multiple_results`, just in case!
        outputs = tuple(outputs.values())
        p.multiple_results = len(outputs) > 1
        return outputs if p.multiple_results else outputs[0]

    def p_tile_translation_ipu(
        p: Primitive,
        tiles: Tuple[int, ...],
        in_avals: List[ShapedArray],
        attributes: Dict[str, Any] = None,
    ) -> IpuTileMapEquation:
        """IPU tile translation for custom vertex."""
        outputs, constants, temps, perf_estimate = initializer_fn(*in_avals)

        # Always set primitive `multiple_results`, just in case!
        p.multiple_results = len(outputs) > 1

        # InOut entries (checking as well the validity).
        check_in_out_arguments(tuple(inputs), tuple(outputs.keys()))
        inout_names = {v for v in inputs if v in outputs}

        def get_iotype(name: str, default: IpuVertexIOType) -> IpuVertexIOType:
            return IpuVertexIOType.InOut if name in inout_names else default

        # Build inputs/outputs vertex IO type.
        inputs_iotype = {v: get_iotype(v, IpuVertexIOType.In) for v in inputs}
        outputs_iotype = {v: get_iotype(v, IpuVertexIOType.Out) for v in outputs.keys()}

        assert len(in_avals) == len(inputs)

        inavals_dict = {inname: inaval for inname, inaval in zip(inputs, in_avals)}
        # Generate vertex fullname, using templated dtypes.
        vname_used_inputs = [v for v in inputs if f"{{{v}}}" in vname]
        vname_dtype_inputs = {
            v: from_numpy_dtype_to_ipu_type(inavals_dict[v].dtype).name.lower() for v in vname_used_inputs
        }
        vertex_fullname = vname.format(**vname_dtype_inputs)

        # IO infos.
        inputs_info = [
            make_ipu_vertex_io_info(inname, inputs_iotype[inname], inaval) for inname, inaval in zip(inputs, in_avals)
        ]
        outputs_info = [
            make_ipu_vertex_io_info(outname, outputs_iotype[outname], outaval) for outname, outaval in outputs.items()
        ]
        constants = constants if constants else {}
        constants_info = [
            make_ipu_vertex_constant_info(cname, data, vertex_dim2=0) for cname, data in constants.items()
        ]
        # Pass attributes to the vertex.
        attributes = attributes if attributes else {}
        attrs_i32, attrs_f32 = make_ipu_vertex_attributes(**attributes)
        ipu_prim_info = IpuTileMapEquation(
            vname=vertex_fullname,
            pname=p.name,
            tiles=tiles,
            # IO vertex infos.
            inputs_info=inputs_info + constants_info,
            outputs_info=outputs_info,
            # Additional attributes to pass to the vertex
            attributes_i32=attrs_i32,
            attributes_f32=attrs_f32,
            # Optional GP filename and perf. estimate.
            gp_filename=gp_filename,
            perf_estimate=perf_estimate,
        )
        if temps is not None and len(temps) > 0:
            assert len(temps) == 1  # TODO: Decide whether or not to handle multiple tmps
            for name, aval in temps.items():
                # Temporary scratch space to use by the vertex (zero=unused).
                ipu_prim_info.tmp_space_name = name
                ipu_prim_info.tmp_space_aval = make_ipu_shaped_array(aval.shape, aval.dtype)

        return ipu_prim_info

    # Register the abstract evaluation with JAX
    p.def_abstract_eval(p_abstract_aval)
    # Register tile IPU translation.
    register_ipu_tile_primitive(p, p_tile_translation_ipu)
    return p


def declare_ipu_tile_primitive(vname, gp_filename=None):
    """
    Convenience wrapper for create_ipu_tile_primitive_v2.

    The decorator version looks like this:
    ```
    @declare_ipu_tile_primitive("MyVertex", "my_vertex.cpp")
    def my_vertex_p(x, y):
      ...
      return outputs,constants,temps,perf_estimate
    ```

    and translates to the equivalent of:
    ```
    def my_vertex_p_anonymous_fn(x, y):
      ...
      return outputs,constants,temps,perf_estimate

    my_vertex_p = create_ipu_tile_primitive_v2(
      "my_vertex", # name is "s/my_vertex_p/_p$/"
      "MyVertex",
      my_vertex_p_anonymous_fn,
      gp_filename=demo_vertex_filename
    )
    ```
    """

    def the_decorator(init_fn):
        # Get init_fn's name
        pname = init_fn.__name__
        primname, nsubs = re.subn(r"_p$", "", pname)
        if nsubs == 0:
            # TODO: Learn over time if this is helpful or annoying
            raise ValueError(f"We expect pname {pname} to end in _p.")

        return create_ipu_tile_primitive_v2(primname, vname, init_fn, gp_filename=gp_filename)

    return the_decorator
