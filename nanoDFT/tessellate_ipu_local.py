# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import os
from typing import Any, Tuple

import jax.lax
import jax.numpy as jnp
import numpy as np
from jax.core import ShapedArray

from tessellate_ipu import (
    TileShardedArray,
    create_ipu_tile_primitive,
    tile_constant_replicated,
    tile_constant_sharded,
    tile_data_barrier,
    tile_gather,
    tile_map,
    tile_put_replicated,
    tile_put_sharded,
)
from tessellate_ipu.core.tile_interpreter_vertex_utils import make_ipu_vector1d_worker_offsets
from tessellate_ipu.utils import NDArray

Array = Any


def get_jacobi_vertex_gp_filename() -> str:
    return os.path.join(os.path.dirname(__file__), "../core", "vertex", "tile_jacobi_vertex.cpp")


jacobi_update_first_step_p = create_ipu_tile_primitive(
    "jacobi_update_first_step",
    "JacobiUpdateFirstStep",
    inputs=["rotset", "pcol", "qcol"],
    outputs={"cs": ShapedArray((2,), dtype=np.float32), "pcol_updated": 1, "qcol_updated": 2},
    constants={
        "worker_offsets": lambda inavals, *_: make_ipu_vector1d_worker_offsets(
            inavals[1].size, vector_size=2, wdtype=np.uint16
        )
    },
    gp_filename=get_jacobi_vertex_gp_filename(),
    perf_estimate=200,
)


jacobi_update_second_step_p = create_ipu_tile_primitive(
    "jacobi_update_second_step",
    "JacobiUpdateSecondStep",
    inputs=["cs_arr", "rotset_arr", "rotset_idx_ignored", "pcol", "qcol"],
    outputs={"cs_arr": 0, "pcol_updated": 3, "qcol_updated": 4},
    constants={
        "worker_offsets": lambda inavals, *_: make_ipu_vector1d_worker_offsets(
            inavals[3].size, vector_size=2, wdtype=np.uint16
        )
    },
    gp_filename=get_jacobi_vertex_gp_filename(),
    perf_estimate=200,
)


jacobi_update_eigenvectors_p = create_ipu_tile_primitive(
    "jacobi_update_eigenvectors",
    "JacobiUpdateEigenvectors",
    inputs=["cs", "vpcol", "vqcol"],
    outputs={"vpcol_out": 1, "vqcol_out": 2},  # Bug when inplace update?
    constants={
        "worker_offsets": lambda inavals, *_: make_ipu_vector1d_worker_offsets(
            inavals[1].size, vector_size=2, wdtype=np.uint16
        )
    },
    gp_filename=get_jacobi_vertex_gp_filename(),
    perf_estimate=200,
)


def jacobi_initial_rotation_set(N: int) -> NDArray[np.uint32]:
    """Jacobi initial rotation array/set (N/2, 2)."""
    rot = np.arange(0, N).astype(np.uint32).reshape((-1, 2))
    return rot


def jacobi_next_rotation_set(rot: NDArray[np.uint32]) -> NDArray[np.uint32]:
    """Jacobi next rotation set (N/2, 2).

    In short: moving p columns to the right, q columns to the left, with
        p[0] not moving.
    """
    next_rot = np.copy(rot)
    # Translate columns.
    next_rot[2:, 0] = rot[1:-1, 0]
    next_rot[0:-1, 1] = rot[1:, 1]
    # Manage corners!
    next_rot[0, 1] = rot[1, 1]
    next_rot[1, 0] = rot[0, 1]
    next_rot[-1, 1] = rot[-1, 0]
    return next_rot


def jacobi_sort_rotation_set(rotset: NDArray[np.uint32]) -> NDArray[np.uint32]:
    """Sort the p, q indices in the Jacobi rotation set, such p < q."""
    pindices, qindices = rotset[:, 0], rotset[:, 1]
    pindices, qindices = np.minimum(pindices, qindices), np.maximum(pindices, qindices)
    return np.stack([pindices, qindices], axis=-1)


def ipu_jacobi_eigh_iteration(all_AV_cols: Tuple[Array, ...], Atiles: Any, Vtiles: Any) -> Tuple[Array, ...]:
    """IPU Eigen decomposition: single iteration of the Jacobi algorithm.

    NOTE: the goal is to have a function which can be easily combined with `fori_loop`.

    Args:
        all_AV_cols: A and V matrices p/q columns.
        Atiles: A matrix tiles.
        Vtiles: V matrix tiles.
    Returns:
        Tuple of updated A and V matrices p/q columns.
    """
    Apcols, Aqcols, Vpcols, Vqcols = all_AV_cols
    N = Apcols.shape[-1]
    halfN = N // 2
    # TODO: check compatibility of TileShardedArray with fori_loop
    # Shard arrays across tiles.
    Apcols = tile_put_sharded(Apcols, tiles=Atiles)
    Aqcols = tile_put_sharded(Aqcols, tiles=Atiles)
    # Initial eigenvectors (identity matrix).
    Vpcols = tile_put_sharded(Vpcols, tiles=Vtiles)
    Vqcols = tile_put_sharded(Vqcols, tiles=Vtiles)
    # Constant tensor of index to ignored at every iteration.
    rotset_index_ignored = tile_constant_sharded(np.arange(0, halfN, dtype=np.uint32), tiles=Atiles)
    rotset = jacobi_initial_rotation_set(N)
    # print("]]]]]]]]]]]]] HERE is ipu_jacobi_eigh_iteration called")

    # All different size 2 partitions on columns.
    for _ in range(1, N):
        # Sorted rotation set: p < q indices.
        rotset_sorted = jacobi_sort_rotation_set(rotset)
        # On tile constant rotation set tensor building.
        rotset_replicated = tile_constant_replicated(rotset_sorted, tiles=Atiles)
        rotset_sharded = tile_constant_sharded(rotset_sorted, tiles=Atiles)

        # Compute Schur decomposition + on-tile update of columns.
        cs_per_tile, Apcols, Aqcols = tile_map(  # type:ignore
            jacobi_update_first_step_p, rotset_sharded, Apcols, Aqcols, N=N
        )
        # Replicate Schur decomposition across all A tiles: (2*N//2) comms.
        cs_replicated = tile_put_replicated(cs_per_tile.array, tiles=Atiles)
        # Just copy Schur decomposition to associated V tiles.
        cs_Vtiles = tile_put_sharded(cs_per_tile.array, tiles=Vtiles)

        # Second Jacobi update step.
        cs_replicated, Apcols, Aqcols = tile_map(  # type:ignore
            jacobi_update_second_step_p,
            cs_replicated,
            rotset_replicated,
            rotset_index_ignored,
            Apcols,
            Aqcols,
            halfN=halfN,
        )
        # Jacobi eigenvectors update step.
        Vpcols, Vqcols = tile_map(  # type:ignore
            jacobi_update_eigenvectors_p,
            cs_Vtiles,
            Vpcols,
            Vqcols,
        )

        # Barrier, to make we sync. both set of tiles A and V
        Apcols, Aqcols, Vpcols, Vqcols = tile_data_barrier(Apcols, Aqcols, Vpcols, Vqcols)
        # Move columns between tiles. 2*N commns per tile.
        # NOTE: this inter-tile comm is keeping the p < q property on A and V columns.
        Apcols, Aqcols = tile_rotate_columns(Apcols, Aqcols, rotset)
        Vpcols, Vqcols = tile_rotate_columns(Vpcols, Vqcols, rotset)
        # Next rotation set.
        rotset = jacobi_next_rotation_set(rotset)
        
        # jax.debug.print("ipu_jacobi_eigh_iteration returns: {a}, {b}, {c}, {d}", a=Apcols.array, b=Aqcols.array, c=Vpcols.array, d=Vqcols.array)

    return (Apcols.array, Aqcols.array, Vpcols.array, Vqcols.array)


def ipu_jacobi_eigh(x: Array, num_iters: int = 1, initial_guess: Tuple[Array, Array] = None) -> Tuple[Array, Array]:
    """IPU Eigen decomposition, implemented using Jacobi algorithm.

    Args:
        x: Symmetric matrix.
    Returns:
        (eigenvectors (N, N), eigenvalues (N,))
    """
    assert x.ndim == 2
    assert x.shape[0] == x.shape[1]
    N = x.shape[0]
    assert N % 2 == 0
    assert N <= 1024
    halfN = N // 2

    Atiles = tuple(range(0, halfN))
    Vtiles = tuple(range(halfN, 2 * halfN))
    # Initial "eigenvalues" matrix.
    # Apcols = jax.lax.slice_in_dim(x, 0, N, stride=2)
    # Aqcols = jax.lax.slice_in_dim(x, 1, N, stride=2)
    # print("APQ:", Apcols, Apcols)
    # Initial eigenvectors (identity matrix).
    if initial_guess is None:
        Apcols = jax.lax.slice_in_dim(x, 0, N, stride=2)
        Aqcols = jax.lax.slice_in_dim(x, 1, N, stride=2)
        Vpcols = np.identity(N)[0::2]
        Vqcols = np.identity(N)[1::2]
        print("SHAPE DEFAULT:", x.shape, Apcols.shape, Aqcols.shape, Vpcols.shape, Vqcols.shape)
    else:
        # initial_a = jnp.diag(initial_guess[0]) # TODO: this jnp.diag shall be removed
        initial_a = initial_guess[0]
        # print("DIAG ELSE:", initial_a, initial_guess[0])
        Apcols = jax.lax.slice_in_dim(initial_a, 0, N, stride=2)
        Aqcols = jax.lax.slice_in_dim(initial_a, 1, N, stride=2)
        # Apcols = jax.lax.slice_in_dim(x, 0, N, stride=2)
        # Aqcols = jax.lax.slice_in_dim(x, 1, N, stride=2)
        # Apcols = initial_a[:halfN]
        # Aqcols = initial_a[halfN:]
        initial_v = initial_guess[1].T ## Need to transpose the eigvecs since it is transposed before being returned from ipu_eigh
        Vpcols = initial_v[0::2]
        Vqcols = initial_v[1::2]
        # jax.debug.print("initial_v\n {v}", v=initial_v)
        # print("SHAPE ELSE:", initial_a.shape, Apcols.shape, Aqcols.shape, Vpcols.shape, Vqcols.shape)
        print(":::::::::::", N, initial_guess[0].shape, initial_guess[1].shape)
        # jax.debug.print("APQ ELSE Apcols\n{a} \nAqcols\n {b}\nDiff\n {d}:", a=Apcols, b=Apcols_ref, d=Apcols-Apcols_ref)

    # print(">>>>>>>>>> HERE is ipu_jacobi_eigh called")

    # Set A and V tiling static.
    eigh_iteration_fn = lambda _, x: ipu_jacobi_eigh_iteration(x, Atiles, Vtiles)
    # JAX fori_loop => no Python unrolling and code bloating!
    tmpApcols = Apcols
    Apcols, Aqcols, Vpcols, Vqcols = jax.lax.fori_loop(
        0, num_iters, eigh_iteration_fn, (Apcols, Aqcols, Vpcols, Vqcols)
    )
    # jax.debug.print("AP BEFORE AFTER Apcols_before\n{a} \nAqcols_after\n {b}\n:", a=tmpApcols, b=Apcols)

    # Expect the output to follow the initial rotation set columns split.
    rotset = jacobi_initial_rotation_set(N)
    # Re-organize pcols and qcols into the result matrix.
    Aresult_rows = [None] * N
    Vresult_cols = [None] * N
    for idx, (p, q) in enumerate(rotset):
        Aresult_rows[p] = jax.lax.slice_in_dim(Apcols, start_index=idx, limit_index=idx + 1)
        Aresult_rows[q] = jax.lax.slice_in_dim(Aqcols, start_index=idx, limit_index=idx + 1)

        Vresult_cols[p] = jax.lax.slice_in_dim(Vpcols, start_index=idx, limit_index=idx + 1)
        Vresult_cols[q] = jax.lax.slice_in_dim(Vqcols, start_index=idx, limit_index=idx + 1)

    A = jax.lax.concatenate(Aresult_rows, dimension=0)
    VT = jax.lax.concatenate(Vresult_cols, dimension=0)
    return A, VT


def permute_pq_indices(
    pindices: NDArray[np.int32], qindices: NDArray[np.int32], rotset_permute_mask: NDArray[np.bool_]
) -> Tuple[NDArray[np.int32], NDArray[np.int32]]:
    """Permute p,q indices based on a mask.

    Args, Returns: (N//2,) shaped arrays.
    """
    return (np.where(rotset_permute_mask, pindices, qindices), np.where(rotset_permute_mask, qindices, pindices))


def tile_rotate_columns(
    pcols: TileShardedArray, qcols: TileShardedArray, rotset: NDArray[np.uint32]
) -> Tuple[TileShardedArray, TileShardedArray]:
    """Rotate columns between tiles using a static `tile_gather`.

    The tricky part of this function is to rotate the columns between tiles, but
    keep the property p < q, which means taking care of the present sorting permutation applied
    as well the next sorting permutation.
    """
    assert pcols.shape == qcols.shape
    assert pcols.tiles == qcols.tiles
    halfN = pcols.shape[0]
    N = halfN * 2
    # Concat all columns, in order to perform a single gather.
    all_cols = TileShardedArray(
        jax.lax.concatenate([pcols.array, qcols.array], dimension=0), (*pcols.tiles, *qcols.tiles)
    )

    # Start with current indices, in the concat representation of columns
    pcols_indices = np.arange(0, halfN, dtype=np.int32)
    qcols_indices = np.arange(halfN, N, dtype=np.int32)
    # First sorting permutation correction.
    rotset_permute_mask = rotset[:, 0] < rotset[:, 1]
    pcols_indices, qcols_indices = permute_pq_indices(pcols_indices, qcols_indices, rotset_permute_mask)

    # Rotation of columns between tiles (see Jacobi alg.)
    # Roughtly: pcols move to the right, qcols to the left.
    pcols_indices_new = np.concatenate([pcols_indices[0:1], qcols_indices[0:1], pcols_indices[1:-1]])
    qcols_indices_new = np.concatenate([qcols_indices[1:], pcols_indices[-1:]])
    pcols_indices, qcols_indices = pcols_indices_new, qcols_indices_new
    assert len(pcols_indices_new) == halfN
    assert len(qcols_indices_new) == halfN

    # Second sorting permutation correction, using the next rotation set.
    rotset = jacobi_next_rotation_set(rotset)
    rotset_permute_mask = rotset[:, 0] < rotset[:, 1]
    pcols_indices, qcols_indices = permute_pq_indices(pcols_indices, qcols_indices, rotset_permute_mask)

    # Move columns around + re-split between pcols and qcols.
    all_indices = np.concatenate([pcols_indices, qcols_indices])
    all_cols_updated = tile_gather(all_cols, all_indices.tolist(), all_cols.tiles)
    return all_cols_updated[:halfN], all_cols_updated[halfN:]


def ipu_eigh(
    x: Array, *, lower: bool = True, symmetrize_input: bool = False, sort_eigenvalues: bool = True, num_iters: int = 1,
    initial_guess = None
) -> Tuple[Array, Array]:
    """IPU (optimized) eigh implementation.

    Args:
        x: Input matrix (N,N) (Nd not supported).
        lower: Not supported.
        symmetrize_input: Not supported, must be false.
        sort_eigenvalues: Sort in ascending order.
    Returns:
        Tuple of eigenvectors (N, N), eigenvalues (N,)
    """
    assert x.ndim == 2
    assert x.shape[0] == x.shape[1]
    N = x.shape[0]
    assert N % 2 == 0
    assert N <= 1024
    assert not symmetrize_input

    A, VT = ipu_jacobi_eigh(x, num_iters=num_iters, initial_guess=initial_guess)
    # jax.debug.print("RETURN ipu_jacobi_eigh EIGVALS \n{va}\n and EIGVECTS\n{ve}\n", va=A, ve=VT)
    # jax.debug.print("<><><><> RETURN ipu_jacobi_eigh A matrix \n{va}\n", va=A)
    eigvalues = jnp.diag(A)
    eigvectors_tr = VT
    # Sorting eigen values.
    if sort_eigenvalues:
        indices = jax.lax.iota(np.int32, len(eigvalues))
        eigvalues, indices = jax.lax.sort_key_val(eigvalues, indices)
        eigvectors_tr = eigvectors_tr[indices]

    # TODO: understand memory layout bug when not forcing the data to be re-organized.
    # Is it related to host rearrangement?
    eigvectors = tile_put_sharded(eigvectors_tr.T, tiles=tuple(range(N)))
    jax.debug.print("actual eigcvects\n {v}", v=eigvectors.array)
    jax.debug.print("actual eigvals\n {v}", v=eigvalues)

    return eigvectors.array, eigvalues, (jnp.diag(A), VT)