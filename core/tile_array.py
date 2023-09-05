# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from dataclasses import dataclass
from typing import Any, Sequence, Tuple, Union

import chex
import numpy as np
from jax.core import ShapedArray
from jax.interpreters.xla import DeviceArray
from jax.tree_util import register_pytree_node_class

from tessellate_ipu.utils import ArrayLike, DTypeLike, Shape

from .tile_array_primitives import (
    tile_constant_replicated_prim,
    tile_constant_sharded_prim,
    tile_data_barrier_prim,
    tile_gather_prim,
    tile_put_replicated_prim,
    tile_put_sharded_prim,
)

TilesType = Tuple[int, ...]
SliceType = Union[int, slice]
MultiSliceType = Tuple[SliceType, ...]


def check_tile_array_multi_slice(slices: MultiSliceType, shape: Shape) -> bool:
    """Check if a tile array multi-slice is valid.

    This means it will keep memory contiguity of the underlying IPU array.

    Args:
        slices: A tuple of slices.
        shape: (full) The shape of the array to slice.
    """
    assert isinstance(slices, tuple)
    # TODO: support `newaxis`
    if len(slices) > len(shape):
        raise ValueError(f"Unsupported slicing `{slices}` on IPU tile array of shape `{shape}`.")
    if len(slices) < len(shape):
        # Complete with full slices.
        full_slices = [slice(None)] * (len(shape) - len(slices))
        slices = (*slices, *full_slices)

    # Check there is no strided slice.
    for s in slices[1:]:
        if isinstance(s, slice) and s.step not in {None, 1}:
            raise ValueError(f"Unsupported strided slicing `{slices}` on IPU tile array of shape `{shape}`.")

    # Last axis with non trivial stride
    non_trivial_slice_axes = [idx for idx in range(len(shape)) if (shape[idx] > 1 and slices[idx] != slice(None))]
    last_non_trivial_slice_axis = max(non_trivial_slice_axes) if len(non_trivial_slice_axes) > 0 else 0

    # Check only axis slicing in-between.
    for idx in range(1, last_non_trivial_slice_axis):
        s = slices[idx]
        valid_slice = isinstance(s, int)
        if not valid_slice:
            raise ValueError(f"Unsupported slicing `{slices}` on IPU tile array of shape `{shape}`.")

    # Should be good!
    return True


@register_pytree_node_class
@dataclass(frozen=True)
class TileShardedArray:
    """JAX array sharded over (IPU) tiles.

    An IPU tile sharded array should satisfy the following assumptions:
        - Must be sharded over the first axis on a given collection of tiles;
        - Each shard is contiguous in memory on every tile;

    On non-IPU hardware (for example, CPUs or GPUs), a tile sharded array will
    just be a normal array, with no particular assumption of memory layout.

    The constructor assumes a proper tile mapping already exists. If this is
    not the case, use `tile_put_sharded` and `tile_put_replicated` to build a
    proper sharded array.

    Args:
        array: The underlying JAX array.
        tiles: A tuple of tiles on which the array is sharded.
    """

    array: chex.ArrayDevice
    tiles: TilesType

    def __post_init__(self):
        # Check consistent array and collection of tiles.
        if len(self.tiles) != self.array.shape[0]:
            raise ValueError(
                f"Inconsistent IPU sharded array shape '{self.array.shape}' and number of tiles {len(self.tiles)}."
            )
        # Make sure we have a tuple of ints.
        tiles = tuple([int(v) for v in self.tiles])
        object.__setattr__(self, "tiles", tiles)

    def tree_flatten(self):
        # See official JAX documentation on extending PyTrees. Tile mapping is static, hence metadata.
        children = (self.array,)
        aux_data = self.tiles
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        # See official JAX documentation on extending PyTrees. Tile mapping is static, hence metadata.
        assert len(children) == 1
        return cls(children[0], aux_data)

    @property
    def dtype(self) -> Any:
        return self.array.dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.array.shape

    @property
    def size(self) -> int:
        return self.array.size

    @property
    def aval(self) -> ShapedArray:
        if isinstance(self.array, np.ndarray):
            return ShapedArray(self.array.shape, self.array.dtype)
        return self.array.aval

    @property
    def num_tiles(self) -> int:
        return len(self.tiles)

    @property
    def tile_aval(self) -> ShapedArray:
        """Abstract val, at the tile level."""
        aval = self.aval
        return ShapedArray(aval.shape[1:], aval.dtype)

    @property
    def tile_shape(self) -> Shape:
        return self.tile_aval.shape

    @property
    def device_array(self) -> DeviceArray:
        return self.array

    def reshape(self, shape: Shape) -> "TileShardedArray":
        d0 = shape[0]
        if d0 != -1 and d0 != self.num_tiles:
            raise ValueError(f"Can not reshape '{shape}' the tile sharding axis in a TileShardedArray.")
        shape = (self.num_tiles, *shape[1:])
        return TileShardedArray(array=self.array.reshape(shape), tiles=self.tiles)

    def tile_reshape(self, shape: Shape) -> "TileShardedArray":
        return self.reshape((self.num_tiles, *shape))

    def squeeze(self):
        squeezed_array = self.array.squeeze()
        has_single_tile = self.num_tiles == 1
        if has_single_tile:
            squeezed_array = squeezed_array.reshape((1, *squeezed_array.shape))
        return TileShardedArray(array=squeezed_array, tiles=self.tiles)

    def __len__(self) -> int:
        return len(self.array)

    def __array__(self, dtype: DTypeLike = None):
        # Force converting to NumPy array.
        return np.asarray(self.array, dtype=dtype)

    def __getitem__(self, key: Union[SliceType, MultiSliceType]) -> "TileShardedArray":
        """Slice over the tile axis."""
        # Make sure we have a tuple of slices.
        if isinstance(key, (int, slice)):
            return self.__getitem__((key,))
        if not isinstance(key, tuple):
            raise ValueError(f"Unsupported tile sharded array slicing key: {key}.")

        # First key => always a slice so we keep the tile axis.
        k0 = key[0]
        if isinstance(k0, int):
            key = (slice(k0, k0 + 1), *key[1:])

        # Check we have a valid slice (keep memory contiguity).
        check_tile_array_multi_slice(key, self.array.shape)
        return TileShardedArray(array=self.array[key], tiles=self.tiles[key[0]])  # type:ignore


def tile_put_sharded(array: DeviceArray, tiles: Sequence[int]) -> TileShardedArray:
    """Shard a JAX array over tiles on the first axis.

    Args:
        array: The array to shard on the first axis.
        tiles: A collection of tile IDs to shard the array on.
    Returns:
        The tile sharded array.
    """
    # TODO: support JAX pytrees.
    return TileShardedArray(array=tile_put_sharded_prim(array, tiles), tiles=tiles)  # type:ignore


def tile_put_replicated(array: DeviceArray, tiles: Sequence[int]) -> TileShardedArray:
    """Replicate a JAX array over tiles on the first axis.

    Args:
        array: The array to replicate on tiles
        tiles: A collection of tile IDs to shard the array on.
    Returns:
        The tile sharded array.
    """
    # TODO: support JAX pytrees.
    return TileShardedArray(array=tile_put_replicated_prim(array, tiles), tiles=tiles)  # type:ignore


def tile_data_barrier(*args: TileShardedArray) -> Tuple[TileShardedArray, ...]:
    """Tile sharded arrays data barrier: force aligning between tiles in the Poplar program.

    Args:
        *args: The input tile sharded arrays.
    Returns:
        The output tile arrays.
    """
    assert all([isinstance(v, TileShardedArray) for v in args])
    # No need for a barrier when it is a single array.
    if len(args) == 1:
        return args[0]  # type:ignore

    inputs_tiles = [v.tiles for v in args]
    raw_inputs = [v.array for v in args]
    raw_outputs = tile_data_barrier_prim(raw_inputs, inputs_tiles)
    return tuple([TileShardedArray(output, input.tiles) for output, input in zip(raw_outputs, args)])


# Short alias!
tile_barrier = tile_data_barrier


def tile_gather(
    arr: Union[DeviceArray, TileShardedArray], indices: Sequence[int], tiles: Sequence[int], copy: bool = False
) -> TileShardedArray:
    """Gather a JAX array over tiles on the first axis.

    By default, if a slice of an input sharded array is already located on the
    proper tile, data will not be copied (no `Memcpy` vertex inserted).

    Args:
        arr: An array. Can be generic, or already tile sharded.
        indices: Indices for the (static) Gather operation.
        tiles: The IPU tiles to shard over.
        copy: If True, data is always copied, even when already properly tile mapped.
    Returns:
        The array sharded over the IPU tiles.
    """
    assert len(indices) == len(tiles)
    assert min(indices) >= 0
    assert max(indices) <= len(arr) - 1
    # Existing tile mapping? -1 by default when none.
    previous_tiles = tuple([-1] * len(arr))
    if isinstance(arr, TileShardedArray):
        previous_tiles = arr.tiles
    # Force copy => act like there is no pre-existing tile mapping.
    if copy:
        previous_tiles = tuple([-1] * len(previous_tiles))

    data_arr = arr.array if isinstance(arr, TileShardedArray) else arr
    gather_arr = tile_gather_prim(data_arr, previous_tiles, indices, tiles)
    return TileShardedArray(array=gather_arr, tiles=tiles)  # type:ignore


def tile_constant_replicated(data: ArrayLike, tiles: Sequence[int]) -> TileShardedArray:
    """Replicate a (constant) NumPy array over tiles on the first axis.

    Args:
        data: The NumPy array with data to replicate.
        tiles: The tiles on which to replicate the data.
    Returns:
        `TileShardedArray` with constant data.
    """
    data = np.asarray(data)
    arr = tile_constant_replicated_prim(data, tiles)
    return TileShardedArray(array=arr, tiles=tiles)  # type:ignore


def tile_constant_sharded(data: ArrayLike, tiles: Sequence[int]) -> TileShardedArray:
    """Shard a (constant) NumPy array over tiles on the first axis.

    Args:
        data: The NumPy array with data to replicate.
        tiles: The tiles on which to replicate the data.
    Returns:
        `TileShardedArray` with constant data.
    """
    data = np.asarray(data)
    assert data.shape[0] == len(tiles)
    arr = tile_constant_sharded_prim(data, tiles)
    return TileShardedArray(array=arr, tiles=tiles)  # type:ignore
