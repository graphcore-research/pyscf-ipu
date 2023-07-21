# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import matplotlib.pyplot as plt
import numpy
import numpy as np
import math
import pyscf
import pyscf.dft
import time
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from functools import partial
import jax
import ctypes
import _ctypes
from icecream import ic

from ctypes import *
import os
import os.path as osp

# lazy load to allow using with CPU without JAX IPU experimental addons
try:
    from tessellate_ipu import create_ipu_tile_primitive, ipu_cycle_count, tile_map, tile_put_sharded, tile_put_replicated
except:
    pass

try: libcgto = cdll.LoadLibrary( os.getcwd() + "/gen.so" )
except: pass
from pyscf import lib

import re
def get_atom_string(atoms, locs):
    atom_string = atoms
    atoms = re.findall('[a-zA-Z][^A-Z]*', atoms)
    str = ""
    for atom, loc in zip(atoms, locs):
      str += "%s %4f %4f %4f; "%((atom,) + tuple(loc) )
    return atom_string, str

def num_tiles():
        return 1472

NUM_TILES = num_tiles()

@partial(jax.jit, backend="ipu", static_argnums=(3,))
def single(input_floats, input_ints, input_ijkl, tiles):

        vertex_filename = osp.join(osp.dirname(__file__), "int2e_sph.cpp")
        int2e_sph = create_ipu_tile_primitive(
                "poplar_int2e_sph",
                "poplar_int2e_sph",
                inputs=["ipu_floats", "ipu_ints", "ipu_ij", "ipu_output", "tile_g", "tile_idx", "tile_buf"],
                outputs={"ipu_output": 3, "tile_g": 4, "tile_idx": 5, "tile_buf": 6},
                gp_filename=vertex_filename,
                perf_estimate=100,
        )

        cpu_output = jnp.zeros((1, 625))

        tile_floats = tile_put_replicated(input_floats, tiles)
        tile_ints   = tile_put_replicated(input_ints,   tiles)
        cpu_output  = tile_put_replicated(cpu_output,   tiles)
        tile_ijkl   = tile_put_replicated(input_ijkl,   tiles)

        tile_g      = tile_put_replicated(jnp.zeros(3888+1),                      tiles)
        tile_idx    = tile_put_replicated(jnp.zeros(3888+1, dtype = jnp.int32) ,  tiles)
        tile_buf    = tile_put_replicated(jnp.zeros(1080*4+1) ,                   tiles)

        tile_floats, start = ipu_cycle_count(tile_floats)
        cpu_output, tile_g, _, _= tile_map(int2e_sph, tile_floats, tile_ints, tile_ijkl, cpu_output, tile_g, tile_idx, tile_buf)
        tile_floats, stop = ipu_cycle_count(tile_floats)

        return cpu_output, start, stop

def ipu_direct_mult_v0(__out3, dm, indices, do_lists, N, num_tiles, indxs_inv, indxs, threads=0):
        vertex_filename  = osp.join(osp.dirname(__file__), "int2e_sph.cpp")
        poplar_direct_s1 = create_ipu_tile_primitive(
                "poplar_direct_s1",
                "poplar_direct_s1",
                inputs=["integral", "indices", "do_list", "dm", "in_vj", "in_vk"],
                outputs={ "vj": 3, "vk": 3},
                gp_filename=vertex_filename,
                perf_estimate=100,
        )

        indxs = np.array(indxs)

        #print(threads)
        num_threads = threads
        num_tiles = NUM_TILES-1

        tiles = tuple((np.arange(num_tiles*num_threads)%num_tiles+1).tolist())
        dm    = tile_put_replicated(    dm,             tiles=tiles)


        print_sizes = False
        if print_sizes:
                print("input",[(a.shape, np.prod(a.shape)/10**6) for a in __out3])
                print("dm", np.prod(dm.shape)*4/10**6)

        _vj = tile_put_replicated(jnp.zeros((N, N)), tiles=tiles)
        _vk = tile_put_replicated(jnp.zeros((N, N)), tiles=tiles)

        start = 0
        for eri_s8 in __out3:
                stop = start + eri_s8.shape[0]

                _indices  = np.array(indices). reshape(-1, 8)[indxs][start:stop]
                _do_lists = np.array(do_lists).reshape(-1, 8)[indxs][start:stop]

                chunk_size = len(tiles)
                count      = eri_s8.shape[0]
#
                if print_sizes:
                        print("vj", np.prod(_vj.shape)*4/10**6)
                        print("vk", np.prod(_vk.shape)*4/10**6)

                # could jax.lax.foriloop this.
                for j in range( count // chunk_size + 1 ):
                        _start, _stop = j*chunk_size, (j+1)*chunk_size

                        __indices  = _indices [_start: _stop]
                        __do_lists = _do_lists[_start: _stop]
                        _eri_s8    = eri_s8   [_start: _stop]

                        n = __indices.shape[0]
                        if n != len(tiles):
                                # pad
                                __indices  = np.concatenate( (__indices,  np.zeros((len(tiles)-n, 8), dtype=__indices.dtype)))
                                __do_lists = np.concatenate( (__do_lists, np.zeros((len(tiles)-n, 8), dtype=__do_lists.dtype)))
                                _eri_s8    = jnp.concatenate((_eri_s8,    np.zeros((len(tiles)-n, _eri_s8.shape[1]))))


                                if print_sizes:
                                        print("indices", np.prod(__indices.shape)*4/10**6)
                                        print("do_lists", np.prod(__do_lists.shape)*4/10**6)
                                        print("eri", np.prod(_eri_s8.shape)*4/10**6)


                        _vj, _vk = tile_map(                    poplar_direct_s1,
                                        tile_put_sharded(       _eri_s8.  reshape(len(tiles), -1),             tiles=tiles),
                                        tile_put_sharded(       np.array(__indices). reshape(len(tiles), 8),   tiles=tiles),
                                        tile_put_sharded(       np.array(__do_lists).reshape(len(tiles), 8),   tiles=tiles),
                                                                dm,
                                                                _vj,
                                                                _vk
                        )

                start = stop

        vj = jnp.sum(_vj.array, axis=0)
        vk = jnp.sum(_vk.array, axis=0)

        return vj, vk

def ipu_direct_mult_v1(__out3, dm, indices, do_lists, N, num_tiles, indxs_inv, indxs, threads=1):
        vertex_filename  = osp.join(osp.dirname(__file__), "int2e_sph.cpp")
        poplar_direct_s1 = create_ipu_tile_primitive(
                "poplar_direct_s1",
                "poplar_direct_s1",
                inputs=["integral", "indices", "do_list", "dm", "in_vj", "in_vk"],
                outputs={ "vj": 3, "vk": 3},
                gp_filename=vertex_filename,
                perf_estimate=100,
        )
        poplar_direct_s1_forloop = create_ipu_tile_primitive(
                "poplar_direct_s1_forloop",
                "poplar_direct_s1_forloop",
                inputs=["integral", "indices", "do_list", "dm", "in_vj", "in_vk", "chunk_size"],
                outputs={ "vj": 3, "vk": 3},
                gp_filename=vertex_filename,
                perf_estimate=100,
        )

        indxs = np.array(indxs)

        #print(threads)
        num_threads = threads
        num_tiles = NUM_TILES-1

        tiles = tuple((np.arange(num_tiles*num_threads)%num_tiles+1).tolist())

        tiles = tuple([a for a in tiles if a != 847] ) # remove a specific tile; jnp.roll get's put there and makes it OOM.
        num_tiles = len(tiles)//num_threads

        dm    = tile_put_replicated(    dm,             tiles=tiles)

        print_sizes = False
        if print_sizes:
                print("input",[(a.shape, np.prod(a.shape)/10**6) for a in __out3])
                print("dm", np.prod(dm.shape)*4/10**6)

        _vj = tile_put_replicated(jnp.zeros((N, N)), tiles=tiles)
        _vk = tile_put_replicated(jnp.zeros((N, N)), tiles=tiles)

        start = 0
        for num, eri_s8 in enumerate(__out3):
                stop = start + eri_s8.shape[0]

                _indices  = np.array(indices). reshape(-1, 8)[indxs][start:stop]
                _do_lists = np.array(do_lists).reshape(-1, 8)[indxs][start:stop]
                print(num, eri_s8.shape, _indices.shape, _do_lists.shape)

                chunk_size = len(tiles)
                count      = eri_s8.shape[0]
                if print_sizes:
                        print("vj", np.prod(_vj.shape)*4/10**6)
                        print("vk", np.prod(_vk.shape)*4/10**6)

                chunks = count // chunk_size
                for __j in range(chunks):
                        _start, _stop = __j*chunk_size, (__j+1)*chunk_size


                if chunks > 0:

                        all_stop     = _stop
                        print(_indices.shape, len(tiles), chunks, 8, all_stop)

                        all_indices  = _indices [0:all_stop].reshape( len(tiles), chunks, 8 )
                        all_do_lists = _do_lists[0:all_stop].reshape( len(tiles), chunks, 8 )
                        all_eri_s8   = eri_s8   [0:all_stop].reshape( len(tiles), chunks, eri_s8.shape[1] )
                        _chunk_size   = tile_put_replicated( np.ones((1,chunks)), tiles=tiles)

                        _vj, _vk = tile_map(                            poplar_direct_s1_forloop,
                                                tile_put_sharded(       all_eri_s8,             tiles=tiles),
                                                tile_put_sharded(       all_indices,   tiles=tiles),
                                                tile_put_sharded(       all_do_lists,   tiles=tiles),
                                                                        dm,
                                                                        _vj,
                                                                        _vk,
                                                                        _chunk_size
                        )



                # do last iteration that might not fit size-wise
                for j in range( count//chunk_size, count // chunk_size + 1 ):
                        _start, _stop = j*chunk_size, (j+1)*chunk_size

                        __indices  = _indices [_start: _stop]
                        __do_lists = _do_lists[_start: _stop]
                        _eri_s8    = eri_s8   [_start: _stop]

                        n = __indices.shape[0]
                        if n != len(tiles):
                                # pad
                                __indices  = np.concatenate( (__indices,  np.zeros((len(tiles)-n, 8), dtype=__indices.dtype)))
                                __do_lists = np.concatenate( (__do_lists, np.zeros((len(tiles)-n, 8), dtype=__do_lists.dtype)))
                                _eri_s8    = jnp.concatenate((_eri_s8,    np.zeros((len(tiles)-n, _eri_s8.shape[1]))))


                                if print_sizes:
                                        print("indices", np.prod(__indices.shape)*4/10**6)
                                        print("do_lists", np.prod(__do_lists.shape)*4/10**6)
                                        print("eri", np.prod(_eri_s8.shape)*4/10**6)


                        _vj, _vk = tile_map(                    poplar_direct_s1,
                                        tile_put_sharded(       _eri_s8.  reshape(len(tiles), -1),             tiles=tiles),
                                        tile_put_sharded(       np.array(__indices). reshape(len(tiles), 8),   tiles=tiles),
                                        tile_put_sharded(       np.array(__do_lists).reshape(len(tiles), 8),   tiles=tiles),
                                                                dm,
                                                                _vj,
                                                                _vk
                        )

                start = stop

        vj = jnp.sum(_vj.array, axis=0)
        vk = jnp.sum(_vk.array, axis=0)

        return vj, vk

def ipu_direct_mult_v2(__out3, dm, indices, do_lists, N, num_tiles, indxs_inv, indxs, threads=1):
        vertex_filename  = osp.join(osp.dirname(__file__), "int2e_sph.cpp")
        poplar_direct_s1_vj = create_ipu_tile_primitive(
                "poplar_direct_s1_vj",
                "poplar_direct_s1_vj",
                inputs=["integral", "indices", "do_list", "dm", "in_vj"],
                outputs={ "vj": 3,},
                gp_filename=vertex_filename,
                perf_estimate=100,
        )
        poplar_direct_s1_vk = create_ipu_tile_primitive(
                "poplar_direct_s1_vk",
                "poplar_direct_s1_vk",
                inputs=["integral", "indices", "do_list", "dm",  "in_vk"],
                outputs={ "vk": 3},
                gp_filename=vertex_filename,
                perf_estimate=100,
        )
        poplar_direct_s1_forloop_vj = create_ipu_tile_primitive(
                "poplar_direct_s1_forloop_vj",
                "poplar_direct_s1_forloop_vj",
                inputs=["integral", "indices", "do_list", "dm", "in_vj",  "chunk_size"],
                outputs={ "vj": 3},
                gp_filename=vertex_filename,
                perf_estimate=100,
        )
        poplar_direct_s1_forloop_vk = create_ipu_tile_primitive(
                "poplar_direct_s1_forloop_vk",
                "poplar_direct_s1_forloop_vk",
                inputs=["integral", "indices", "do_list", "dm",  "in_vk", "chunk_size"],
                outputs={  "vk": 3},
                gp_filename=vertex_filename,
                perf_estimate=100,
        )

        indxs = np.array(indxs)

        #print(threads)
        num_threads = threads
        num_tiles = NUM_TILES-1

        tiles = tuple((np.arange(num_tiles*num_threads)%num_tiles+1).tolist())

        tiles = tuple([a for a in tiles if a != 847 and a%64 !=0] ) # remove a few tiles to help compiler
        num_tiles = len(tiles)//num_threads

        dm    = tile_put_replicated(    dm,             tiles=tiles)

        print_sizes = False
        if print_sizes:
                print("input",[(a.shape, np.prod(a.shape)/10**6) for a in __out3])
                print("dm", np.prod(dm.shape)*4/10**6)

        def get_vj():

                _vj = tile_put_replicated(jnp.zeros((N, N)), tiles=tiles)

                start = 0
                for num, eri_s8 in enumerate(__out3):
                        stop = start + eri_s8.shape[0]

                        _indices  = np.array(indices). reshape(-1, 8)[indxs][start:stop]
                        _do_lists = np.array(do_lists).reshape(-1, 8)[indxs][start:stop]
                        if print_sizes: print(num, eri_s8.shape, _indices.shape, _do_lists.shape)

                        chunk_size = len(tiles)
                        count      = eri_s8.shape[0]

                        if print_sizes:
                                print("vj", np.prod(_vj.shape)*4/10**6)
                                print("vk", np.prod(_vk.shape)*4/10**6)

                        chunks = count // chunk_size
                        for __j in range(chunks):
                                _start, _stop = __j*chunk_size, (__j+1)*chunk_size


                        if chunks > 0:

                                all_stop     = _stop
                                if print_sizes: print(_indices.shape, len(tiles), chunks, 8, all_stop)

                                all_indices  = _indices [0:all_stop].reshape( len(tiles), chunks, 8 )
                                all_do_lists = _do_lists[0:all_stop].reshape( len(tiles), chunks, 8 )
                                all_eri_s8   = eri_s8   [0:all_stop].reshape( len(tiles), chunks, eri_s8.shape[1] )
                                _chunk_size   = tile_put_replicated( np.ones((1,chunks)), tiles=tiles)


                                _vj= tile_map(          poplar_direct_s1_forloop_vj,
                                                        tile_put_sharded(       all_eri_s8,             tiles=tiles),
                                                        tile_put_sharded(       all_indices,   tiles=tiles),
                                                        tile_put_sharded(       all_do_lists,   tiles=tiles),
                                                                                dm,
                                                                                _vj,
                                                                                _chunk_size
                                )

                        # do last iteration that might not fit size-wise
                        for j in range( count//chunk_size, count // chunk_size + 1 ):
                                _start, _stop = j*chunk_size, (j+1)*chunk_size

                                __indices  = _indices [_start: _stop]
                                __do_lists = _do_lists[_start: _stop]
                                _eri_s8    = eri_s8   [_start: _stop]

                                n = __indices.shape[0]
                                if n != len(tiles):
                                        # pad
                                        __indices  = np.concatenate( (__indices,  np.zeros((len(tiles)-n, 8), dtype=__indices.dtype)))
                                        __do_lists = np.concatenate( (__do_lists, np.zeros((len(tiles)-n, 8), dtype=__do_lists.dtype)))
                                        _eri_s8    = jnp.concatenate((_eri_s8,    np.zeros((len(tiles)-n, _eri_s8.shape[1]))))


                                        if print_sizes:
                                                print("indices", np.prod(__indices.shape)*4/10**6)
                                                print("do_lists", np.prod(__do_lists.shape)*4/10**6)
                                                print("eri", np.prod(_eri_s8.shape)*4/10**6)


                                _vj= tile_map(                          poplar_direct_s1_vj,
                                                tile_put_sharded(       _eri_s8.  reshape(len(tiles), -1),             tiles=tiles),
                                                tile_put_sharded(       np.array(__indices). reshape(len(tiles), 8),   tiles=tiles),
                                                tile_put_sharded(       np.array(__do_lists).reshape(len(tiles), 8),   tiles=tiles),
                                                                        dm,
                                                                        _vj,
                                )

                        start = stop

                return jnp.sum(_vj.array, axis=0)

        vj = get_vj()

        def get_vk(a):

                _vk = tile_put_replicated(jnp.zeros((N, N)), tiles=tiles)

                start = 0
                for num, eri_s8 in enumerate(__out3):
                        stop = start + eri_s8.shape[0]

                        _indices  = np.array(indices). reshape(-1, 8)[indxs][start:stop]
                        _do_lists = np.array(do_lists).reshape(-1, 8)[indxs][start:stop]
                        if print_sizes: print(num, eri_s8.shape, _indices.shape, _do_lists.shape)

                        chunk_size = len(tiles)
                        count      = eri_s8.shape[0]

                        if print_sizes:
                                print("vj", np.prod(_vj.shape)*4/10**6)
                                print("vk", np.prod(_vk.shape)*4/10**6)

                        chunks = count // chunk_size
                        for __j in range(chunks):
                                _start, _stop = __j*chunk_size, (__j+1)*chunk_size


                        if chunks > 0:

                                all_stop     = _stop
                                if print_sizes: print(_indices.shape, len(tiles), chunks, 8, all_stop)

                                all_indices  = _indices [0:all_stop].reshape( len(tiles), chunks, 8 )
                                all_do_lists = _do_lists[0:all_stop].reshape( len(tiles), chunks, 8 )
                                all_eri_s8   = eri_s8   [0:all_stop].reshape( len(tiles), chunks, eri_s8.shape[1] )
                                _chunk_size   = tile_put_replicated( np.ones((1,chunks)), tiles=tiles)

                                _vk= tile_map(          poplar_direct_s1_forloop_vk,
                                                        tile_put_sharded(       all_eri_s8,             tiles=tiles),
                                                        tile_put_sharded(       all_indices,   tiles=tiles),
                                                        tile_put_sharded(       all_do_lists,   tiles=tiles),
                                                                                dm,
                                                                                _vk,
                                                                                _chunk_size
                                )

                        # do last iteration that might not fit size-wise
                        for j in range( count//chunk_size, count // chunk_size + 1 ):
                                _start, _stop = j*chunk_size, (j+1)*chunk_size

                                __indices  = _indices [_start: _stop]
                                __do_lists = _do_lists[_start: _stop]
                                _eri_s8    = eri_s8   [_start: _stop]

                                n = __indices.shape[0]
                                if n != len(tiles):
                                        # pad
                                        __indices  = np.concatenate( (__indices,  np.zeros((len(tiles)-n, 8), dtype=__indices.dtype)))
                                        __do_lists = np.concatenate( (__do_lists, np.zeros((len(tiles)-n, 8), dtype=__do_lists.dtype)))
                                        _eri_s8    = jnp.concatenate((_eri_s8,    np.zeros((len(tiles)-n, _eri_s8.shape[1]))))


                                        if print_sizes:
                                                print("indices", np.prod(__indices.shape)*4/10**6)
                                                print("do_lists", np.prod(__do_lists.shape)*4/10**6)
                                                print("eri", np.prod(_eri_s8.shape)*4/10**6)


                                _vk = tile_map(                         poplar_direct_s1_vk,
                                                tile_put_sharded(       _eri_s8.  reshape(len(tiles), -1),             tiles=tiles),
                                                tile_put_sharded(       np.array(__indices). reshape(len(tiles), 8),   tiles=tiles),
                                                tile_put_sharded(       np.array(__do_lists).reshape(len(tiles), 8),   tiles=tiles),
                                                                        dm,
                                                                        _vk
                                )

                        start = stop

                return jnp.sum(_vk.array, axis=0) # write custom binary tree summation?

        vk = get_vk(vj) # add argument to trick compiler to perform vj,vk sequentially to reduce memory consumption.

        return vj, vk

@partial(jax.jit, backend="ipu", static_argnums=(2,3,4,5,6,7,8,9))
def ipu_direct_mult(__out3, dm, indices, do_lists, N, num_tiles, indxs_inv, indxs, threads=1, v=2):
        #print("THREADS!", threads)
        if v == 0:
                return ipu_direct_mult_v0(__out3, dm, indices, do_lists, N, num_tiles, indxs_inv, indxs, threads=threads)
        elif v == 1:
                return ipu_direct_mult_v1(__out3, dm, indices, do_lists, N, num_tiles, indxs_inv, indxs, threads=threads)

        elif v == 2:
                return ipu_direct_mult_v2(__out3, dm, indices, do_lists, N, num_tiles, indxs_inv, indxs, threads=threads)

def ipu_einsum(__out3, dm, mol, threads=1, v=2):
        _tuple_indices, _tuple_do_lists, _N, num_calls = prepare_einsum_inputs(mol)
        N = mol.nao_nr()
        _, _, _tuple_ijkl, _shapes, _sizes, _counts, indxs, indxs_inv, _ = prepare_electron_repulsion_integrals(mol)
        return ipu_direct_mult( __out3, dm, _tuple_indices, _tuple_do_lists, N, num_calls, tuple(indxs_inv), tuple(indxs), threads, v)

@partial(jax.jit, backend="ipu", static_argnums=(2,3,4,5,6,7,8))
def compute_integrals_2(input_floats, input_ints, input_ijkl, shapes, sizes, counts, indxs_inv, num_threads=3, v=1):
        if v == 0:
                return integrals_v0(input_floats, input_ints, input_ijkl, shapes, sizes, counts, indxs_inv, num_threads=num_threads)
        if v == 1:
                return integrals_v1(input_floats, input_ints, input_ijkl, shapes, sizes, counts, indxs_inv, num_threads=num_threads)

def electron_repulsion_integrals(input_floats, input_ints, mol, num_threads=3, v=1):
        _, _, _tuple_ijkl, _shapes, _sizes, _counts, indxs, indxs_inv, _ = prepare_electron_repulsion_integrals(mol)
        return compute_integrals_2(input_floats, input_ints, _tuple_ijkl, _shapes, _sizes, _counts, tuple(indxs_inv), num_threads, v)[0]

def integrals_v0(input_floats, input_ints, input_ijkl, shapes, sizes, counts, indxs_inv, num_threads=3):

        vertex_filename = osp.join(osp.dirname(__file__), "int2e_sph.cpp")
        int2e_sph = create_ipu_tile_primitive(
                "poplar_int2e_sph",
                "poplar_int2e_sph",
                inputs=["ipu_floats", "ipu_ints", "ipu_ij", "ipu_output", "tile_g", "tile_idx", "tile_buf"],
                outputs={"ipu_output": 3, "tile_g": 4, "tile_idx": 5, "tile_buf": 6},
                gp_filename=vertex_filename,
                perf_estimate=100,
        )

        cpu_outputs = []
        start_index = 0
        start, stop = 0, 0
        np.random.seed(42) #


        num_tiles = NUM_TILES-1

        num_calls = len(indxs_inv)
        #print(num_calls)

        if num_calls < num_tiles*num_threads:
                tiles       = tuple((np.arange(num_calls)+1).tolist())
        else:
                tiles       = tuple((np.arange(num_tiles*num_threads)%(num_tiles)+1).tolist())

        tile_floats = tile_put_replicated(input_floats, tiles)
        tile_ints   = tile_put_replicated(input_ints,   tiles)

        for i, (size, count) in enumerate(zip(sizes, counts)):
                glen, nf, buflen = shapes[i]

                tiles       = tuple((np.arange(num_tiles*num_threads)%(num_tiles)+1).tolist())
                tile_g      = tile_put_replicated(jnp.empty(min(int(glen), 3888)+1),                     tiles)
                tile_idx    = tile_put_replicated(jnp.empty(max(256, min(int(nf*3), 3888)+1), dtype = jnp.int32) ,  tiles)
                tile_buf    = tile_put_replicated(jnp.empty(1080*4+1) ,                   tiles)

                list_cpu_output = []
                chunk_size = num_tiles*num_threads
                for j in range(count // (chunk_size) + 1):
                        start, stop = j*chunk_size, (j+1)*chunk_size
                        indices = np.array(input_ijkl[i][start:stop])
                        if indices.shape[0] != len(tiles):
                                tiles = tuple((np.arange(indices.shape[0])%(num_tiles)+1).tolist())

                        cpu_output = jnp.empty((len(tiles), size))

                        cpu_output  = tile_put_sharded(   cpu_output+j,   tiles)
                        tile_ijkl   = tile_put_sharded(   indices ,     tiles)

                        _cpu_output, _, _, _= tile_map( int2e_sph,
                                                        tile_floats[:len(tiles)],
                                                        tile_ints[:len(tiles)],
                                                        tile_ijkl,
                                                        cpu_output,
                                                        tile_g[:len(tiles)],
                                                        tile_idx[:len(tiles)],
                                                        tile_buf[:len(tiles)]
                                                        )
                        list_cpu_output.append(_cpu_output.array)

                cpu_outputs.append(jnp.concatenate(list_cpu_output)  )

        return cpu_outputs, start, stop

def integrals_v1(input_floats, input_ints, input_ijkl, shapes, sizes, counts, indxs_inv, num_threads=3):

        vertex_filename = osp.join(osp.dirname(__file__), "int2e_sph.cpp")

        int2e_sph = create_ipu_tile_primitive(
                "poplar_int2e_sph",
                "poplar_int2e_sph",
                inputs=["ipu_floats", "ipu_ints", "ipu_ij", "ipu_output", "tile_g", "tile_idx", "tile_buf"],
                outputs={"ipu_output": 3, "tile_g": 4, "tile_idx": 5, "tile_buf": 6},
                gp_filename=vertex_filename,
                perf_estimate=100,
        )
        int2e_sph_forloop = create_ipu_tile_primitive(
                "poplar_int2e_sph_forloop",
                "poplar_int2e_sph_forloop",
                inputs=["ipu_floats", "ipu_ints", "ipu_ij", "ipu_output", "tile_g", "tile_idx", "tile_buf", "chunks", "integral_size"],
                outputs={"ipu_output": 3, "tile_g": 4, "tile_idx": 5, "tile_buf": 6},
                gp_filename=vertex_filename,
                perf_estimate=100,
        )

        cpu_outputs = []
        start_index = 0
        start, stop = 0, 0
        np.random.seed(42)

        num_tiles = NUM_TILES-1

        num_calls = len(indxs_inv)
        #print(num_calls)

        if num_calls < num_tiles*num_threads:  tiles       = tuple((np.arange(num_calls)+1).tolist())
        else:                                  tiles       = tuple((np.arange(num_tiles*num_threads)%(num_tiles)+1).tolist())

        tile_floats = tile_put_replicated(input_floats, tiles)
        tile_ints   = tile_put_replicated(input_ints,   tiles)

        for i, (size, count) in enumerate(zip(sizes, counts)):
                glen, nf, buflen = shapes[i]

                tiles       = tuple((np.arange(num_tiles*num_threads)%(num_tiles)+1).tolist())
                tile_g      = tile_put_replicated(jnp.empty(min(int(glen), 3888)+1),                     tiles)
                tile_idx    = tile_put_replicated(jnp.empty(max(256, min(int(nf*3), 3888)+1), dtype = jnp.int32) ,  tiles)
                tile_buf    = tile_put_replicated(jnp.empty(1080*4+1) ,                   tiles)

                list_cpu_output = []
                chunk_size      = num_tiles * num_threads

                if count // chunk_size > 0:

                        cpu_output = jnp.empty((len(tiles), count//chunk_size, size))
                        cpu_output  = tile_put_sharded(   cpu_output,   tiles)

                        _indices = []
                        for j in range( count // (chunk_size) ):
                                start, stop = j*chunk_size, (j+1)*chunk_size
                                indices = np.array(input_ijkl[i][start:stop])
                                _indices.append(indices.reshape(indices.shape[0], 1, indices.shape[1]))
                        _indices = np.concatenate(_indices, axis=1)
                        _indices = tile_put_sharded(_indices, tiles)

                        chunks = tile_put_replicated( jnp.zeros(count//chunk_size), tiles)
                        integral_size = tile_put_replicated( jnp.zeros(size), tiles)

                        if False:
                                _cpu_output, _, _, _= tile_map(int2e_sph_forloop,
                                                                tile_floats,
                                                                tile_ints,
                                                                _indices[:, :, :],
                                                                cpu_output[:, :, :],
                                                                tile_g,
                                                                tile_idx,
                                                                tile_buf,
                                                                chunks[:,:],
                                                                integral_size
                                                                )

                                for j in range( count // chunk_size ):
                                        print("!!", _cpu_output.shape)
                                        list_cpu_output.append(_cpu_output.array[:, j])
                                batched_out = jnp.concatenate(list_cpu_output)
                                print(batched_out.shape)
                        else:
                                #print(">>", cpu_output.shape, _indices.shape , chunks.shape, integral_size.shape)

                                batched_out , _, _, _= tile_map(int2e_sph_forloop,
                                                                tile_floats,
                                                                tile_ints,
                                                                _indices,
                                                                cpu_output,
                                                                tile_g,
                                                                tile_idx,
                                                                tile_buf,
                                                                chunks,
                                                                integral_size
                                                                )
                                batched_out = jnp.transpose(batched_out.array, (1, 0, 2)).reshape(-1, size)


                # do last iteration normally.
                for j in range(count // chunk_size, count // (chunk_size) + 1):
                        start, stop = j*chunk_size, (j+1)*chunk_size
                        indices = np.array(input_ijkl[i][start:stop])
                        if indices.shape[0] != len(tiles):
                                tiles = tuple((np.arange(indices.shape[0])%(num_tiles)+1).tolist())

                        tile_ijkl   = tile_put_sharded(   indices ,     tiles)
                        cpu_output = jnp.empty((len(tiles), size))
                        cpu_output  = tile_put_sharded(   cpu_output+j,   tiles)

                        _cpu_output, _, _, _= tile_map( int2e_sph,
                                                        tile_floats[:len(tiles)],
                                                        tile_ints[:len(tiles)],
                                                        tile_ijkl,
                                                        cpu_output,
                                                        tile_g[:len(tiles)],
                                                        tile_idx[:len(tiles)],
                                                        tile_buf[:len(tiles)]
                                                )

                if count//chunk_size>0:cpu_outputs.append(jnp.concatenate([batched_out, _cpu_output.array])  )
                else: cpu_outputs.append(_cpu_output.array   )

        return cpu_outputs, start, stop


def prepare_int_floats(mol):
        # Shapes/sizes.
        atm, bas, env   = mol._atm, mol._bas, mol._env
        n_atm, n_bas, N = atm.shape[0], bas.shape[0], mol.nao_nr()
        ao_loc          = np.cumsum(np.concatenate([np.zeros(1), (bas[:,1]*2+1) * bas[:,3] ])).astype(np.int32)
        n_ao_loc        = np.prod(ao_loc.shape)
        shls_slice      = (0, n_bas, 0, n_bas, 0, n_bas, 0, n_bas)
        shape           = [1, N, N, N, N]

        # The padded shape used to store output from all tiles.
        n_buf, n_eri, n_env = 81, 81, np.prod(env.shape)
        if mol.basis == "6-31G*":
                print(">>>", mol.basis)
                n_buf = 5**4
                n_eri = 5**4

        # Prepare IPU inputs.
        # Merge all int/float inputs in seperate arrays.
        input_floats = env.reshape(1, -1)
        input_ints   = np.zeros((1, 6+n_ao_loc +n_atm*6+n_bas*8), dtype=np.int32)
        start, stop = 0, 6
        input_ints[:, start:stop] = np.array( [n_eri, n_buf, n_atm, n_bas, n_env, n_ao_loc] )
        start, stop = start+6, stop+n_ao_loc
        input_ints[:, start:stop] = ao_loc.reshape(-1)
        start, stop = start+n_ao_loc, stop + n_atm*6
        input_ints[:, start:stop] = atm.reshape(-1)
        start, stop = start+n_atm*6, stop + n_bas*8
        input_ints[:, start:stop] = bas.reshape(-1)

        return input_floats, input_ints

def prepare_integrals_2_inputs(mol):
        # Shapes/sizes.
        atm, bas, env   = mol._atm, mol._bas, mol._env
        n_atm, n_bas, N = atm.shape[0], bas.shape[0], mol.nao_nr()
        ao_loc          = np.cumsum(np.concatenate([np.zeros(1), (bas[:,1]*2+1) * bas[:,3] ])).astype(np.int32)
        n_ao_loc        = np.prod(ao_loc.shape)
        shls_slice      = (0, n_bas, 0, n_bas, 0, n_bas, 0, n_bas)
        shape           = [1, N, N, N, N]

        # Initialize tensors for CPU libcint computation.
        buf     = np.zeros(np.prod(shape)*2)
        out     = np.zeros(shape)
        eri     = np.zeros(shape).reshape(-1)
        ipu_eri = np.zeros(shape).reshape(-1)

        dtype = np.float32 #hardcoded
        buf, out, eri, ipu_eri, env = buf.astype(dtype), out.astype(dtype), eri.astype(dtype), ipu_eri.astype(dtype), env.astype(dtype)

        # The padded shape used to store output from all tiles.
        n_buf, n_eri, n_env = 81, 81, np.prod(env.shape)
        if mol.basis == "6-31G*": # has known error; please open github issue if you want to use 6-31G*
                n_buf = 5**4
                n_eri = 5**4

        # Compute how many distinct integrals after 8x symmetry.
        num_calls = 0
        for i in range(n_bas):
                for j in range(n_bas):
                        for k in range(n_bas):
                                for l in range(n_bas):
                                        # * 8-fold symmetry, k>=l, k>=i>=j,
                                        if not ( i >= j and k >= l and i*j >= k * l): continue
                                        num_calls+=1

        # Input/outputs for calling the IPU vertex.
        input_ijkl   = np.zeros((num_calls, 4),     dtype=np.int32)
        cpu_output   = np.zeros((num_calls, n_eri), dtype=np.float32)
        output_sizes = np.zeros((num_calls, 5))

        # Fill input_ijkl and output_sizes with the necessary indices.
        c = 0
        for i in range(n_bas):
                for j in range(n_bas):
                        for k in range(n_bas):
                                for l in range(n_bas):
                                        # * 8-fold symmetry, k>=l, k>=i>=j,
                                        if not ( i >= j and k >= l and i*j >= k * l): continue

                                        input_ijkl[c] = [i, j, k, l]

                                        di = ao_loc[i+1] - ao_loc[i]
                                        dj = ao_loc[j+1] - ao_loc[j]
                                        dk = ao_loc[k+1] - ao_loc[k]
                                        dl = ao_loc[l+1] - ao_loc[l]

                                        output_sizes[c] = [di, dj, dk, dl, di*dj*dk*dl]

                                        c += 1

        # Prepare IPU inputs.
        # Merge all int/float inputs in seperate arrays.
        input_floats = env.reshape(1, -1)
        input_ints   = np.zeros((1, 6+n_ao_loc +n_atm*6+n_bas*8), dtype=np.int32)
        start, stop = 0, 6
        input_ints[:, start:stop] = np.array( [n_eri, n_buf, n_atm, n_bas, n_env, n_ao_loc] )
        start, stop = start+6, stop+n_ao_loc
        input_ints[:, start:stop] = ao_loc.reshape(-1)
        start, stop = start+n_ao_loc, stop + n_atm*6
        input_ints[:, start:stop] = atm.reshape(-1)
        start, stop = start+n_atm*6, stop + n_bas*8
        input_ints[:, start:stop] = bas.reshape(-1)

        def get_shapes(input_ijkl):
                i_sh, j_sh, k_sh, l_sh  = input_ijkl[0]
                BAS_SLOTS = 8
                NPRIM_OF = 2
                NCTR_OF = 3
                ANG_OF = 1
                GSHIFT = 4

                i_prim  = bas.reshape(-1)[BAS_SLOTS*i_sh + NPRIM_OF]
                j_prim  = bas.reshape(-1)[BAS_SLOTS*j_sh + NPRIM_OF]
                k_prim  = bas.reshape(-1)[BAS_SLOTS*k_sh + NPRIM_OF]
                l_prim  = bas.reshape(-1)[BAS_SLOTS*l_sh + NPRIM_OF]

                i_ctr   = bas.reshape(-1)[BAS_SLOTS * i_sh + NCTR_OF]
                j_ctr   = bas.reshape(-1)[BAS_SLOTS * j_sh + NCTR_OF]
                k_ctr   = bas.reshape(-1)[BAS_SLOTS * k_sh + NCTR_OF]
                l_ctr   = bas.reshape(-1)[BAS_SLOTS * l_sh + NCTR_OF]

                i_l     = bas.reshape(-1)[BAS_SLOTS * i_sh + ANG_OF]
                j_l     = bas.reshape(-1)[BAS_SLOTS * j_sh + ANG_OF]
                k_l     = bas.reshape(-1)[BAS_SLOTS * k_sh + ANG_OF]
                l_l     = bas.reshape(-1)[BAS_SLOTS * l_sh + ANG_OF]

                nfi  = (i_l+1)*(i_l+2)/2
                nfj  = (j_l+1)*(j_l+2)/2
                nfk  = (k_l+1)*(k_l+2)/2
                nfl  = (l_l+1)*(l_l+2)/2
                nf = nfi * nfk * nfl * nfj;
                n_comp = 1

                nc = i_ctr * j_ctr * k_ctr * l_ctr;
                lenl = nf * nc * n_comp;
                lenk = nf * i_ctr * j_ctr * k_ctr * n_comp;
                lenj = nf * i_ctr * j_ctr * n_comp;
                leni = nf * i_ctr * n_comp;
                len0 = nf * n_comp;

                ng = [0, 0, 0, 0, 0, 1, 1, 1];

                IINC=0
                JINC=1
                KINC=2
                LINC=3

                li_ceil = i_l + ng[IINC]
                lj_ceil = j_l + ng[JINC]
                lk_ceil = k_l + ng[KINC]
                ll_ceil = l_l + ng[LINC]
                nrys_roots = (li_ceil + lj_ceil + lk_ceil + ll_ceil)/2 + 1


                ibase = li_ceil > lj_ceil;
                kbase = lk_ceil > ll_ceil;
                if (nrys_roots <= 2):
                        ibase = 0;
                        kbase = 0;
                if (kbase) :
                        dlk = lk_ceil + ll_ceil + 1;
                        dll = ll_ceil + 1;
                else:
                        dlk = lk_ceil + 1;
                        dll = lk_ceil + ll_ceil + 1;

                if (ibase) :
                        dli = li_ceil + lj_ceil + 1;
                        dlj = lj_ceil + 1;
                else :
                        dli = li_ceil + 1;
                        dlj = li_ceil + lj_ceil + 1;

                g_stride_i = nrys_roots;
                g_stride_k = nrys_roots * dli;
                g_stride_l = nrys_roots * dli * dlk;
                g_stride_j = nrys_roots * dli * dlk * dll;
                g_size     = nrys_roots * dli * dlk * dll * dlj;
                gbits        = ng[GSHIFT];
                leng = g_size*3*((1<<gbits)+1);

                len = leng + lenl + lenk + lenj + leni + len0;

                di = i_l * 2 + 1;
                dj = j_l * 2 + 1;
                dk = k_l * 2 + 1;
                dl = l_l * 2 + 1;

                ni = (i_l*2+1) * i_ctr;
                nj = (j_l*2+1) * j_ctr;
                nk = (k_l*2+1) * k_ctr;
                nl = (l_l*2+1) * l_ctr;
                nfik = nfi * nfk;
                nfikl = nfik * nfl;
                dlj = dl * dj;
                ofj = ni * dj;

                ofk = ni * nj * dk;
                ofl = ni * nj * nk * dl;
                buflen = nfikl*dj;

                return len, nf, buflen


        sizes, counts  = np.unique(output_sizes[:, -1], return_counts=True)
        sizes, counts = sizes.astype(np.int32), counts.astype(np.int32)

        indxs               = np.argsort(output_sizes[:, -1])
        sorted_output_sizes = output_sizes[indxs]
        input_ijkl          = input_ijkl[indxs]

        sizes, counts  = np.unique(output_sizes[:, -1], return_counts=True)
        sizes, counts = sizes.astype(np.int32), counts.astype(np.int32)
        start_index = 0
        inputs = []
        shapes = []
        for i, (size, count) in enumerate(zip(sizes, counts)):
                a = input_ijkl[start_index: start_index+count]
                tuples = tuple(map(tuple, a))
                inputs.append(tuples)
                start_index += count

        tuple_ijkl = tuple(inputs)
        input_ijkl = inputs

        for i in range(len(sizes)):
                shapes.append( get_shapes(input_ijkl[i]) )

        def inverse_permutation(a):
                b = np.arange(a.shape[0])
                b[a] = b.copy()
                return b
        indxs_inv = inverse_permutation(indxs)

        return input_floats, input_ints, tuple_ijkl, tuple(shapes), tuple(sizes.tolist()), tuple(counts.tolist()), indxs, indxs_inv, num_calls

prepare_electron_repulsion_integrals = prepare_integrals_2_inputs

def prepare_ipu_direct_mult_inputs(num_calls, mol):
        atm, bas, env   = mol._atm, mol._bas, mol._env
        n_atm, n_bas, N = atm.shape[0], bas.shape[0], mol.nao_nr()
        ao_loc          = np.cumsum(np.concatenate([np.zeros(1), (bas[:,1]*2+1) * bas[:,3] ])).astype(np.int32)
        n = n_bas
        c = 0
        dct = {}
        lst = []

        do_lists = np.zeros((num_calls, 8), dtype=np.int32)
        indices  = np.zeros((num_calls, 8))

        #for i in tqdm(range(n)):
        for i in range(n):
                for j in range(n):
                        for k in range(n):
                                for l in range(n):
                                        if not (i >= j and k >= l and i*j >= k * l): continue

                                        # the output of integral (i,j,k,l) has shape (di, dj, dk, dl)
                                        # where di,dj,dk,dl are in {1,2,3,4,5} because assuming
                                        #  - only use {c,h,o,n}
                                        #  - represent electrons as pcq dataset 6-31G* (or sto3g/6-31g)
                                        di = ao_loc[i+1] - ao_loc[i]
                                        dj = ao_loc[j+1] - ao_loc[j]
                                        dk = ao_loc[k+1] - ao_loc[k]
                                        dl = ao_loc[l+1] - ao_loc[l]

                                        # knowing the shape of integral (i,j,k,l) we can fetch it.
                                        #integral = np.transpose( eri_s1[c][ :di*dj*dk*dl].reshape(dl,dk,dj,di), (3,2,1,0) )

                                        # we now need to compute where the current integral 'block' goes in
                                        # the final output matrix.
                                        i0 = ao_loc[i] - ao_loc[0]
                                        j0 = ao_loc[j] - ao_loc[0]
                                        k0 = ao_loc[k] - ao_loc[0]
                                        l0 = ao_loc[l] - ao_loc[0]

                                        indices[c] = np.array([ di, dj, dk, dl, i0, j0, k0, l0 ])

                                        do_list = [False]*8

                                        if not ( i0,j0,k0,l0 ) in dct:
                                                dct [ i0, j0, k0, l0 ] = 8
                                                do_list[0] = True

                                        if not ( i0, j0, l0, k0) in dct:
                                                dct [ i0, j0, l0, k0] = 9
                                                do_list[1] = True

                                        if not ( j0, i0, k0, l0 ) in dct:
                                                dct[ j0, i0, k0, l0 ] = 10
                                                do_list[2] = True

                                        if not ( j0, i0, l0, k0 ) in dct:
                                                dct[ j0, i0, l0, k0 ] = 11
                                                do_list[3] = True

                                        if not ( k0, l0, i0, j0 ) in dct:
                                                dct[ k0, l0, i0, j0 ] = 12
                                                do_list[4] = True

                                        if not ( k0, l0, j0, i0  ) in dct:
                                                dct[ k0, l0, j0, i0 ]  = 13
                                                do_list[5] = True

                                        if not ( l0, k0, i0, j0  ) in dct:
                                                dct[ l0, k0, i0, j0 ]  = 14
                                                do_list[6] = True
                                        if not ( l0, k0, j0, i0 ) in dct:
                                                dct [ l0, k0, j0, i0 ]  = 15
                                                do_list[7] = True

                                        do_lists[c] = np.array(do_list)
                                        c += 1


        tuple_indices  = tuple(map(tuple, indices.astype(np.int32).tolist()))
        tuple_do_lists = tuple(map(tuple, do_lists.astype(np.int32).tolist()))
        return tuple_indices, tuple_do_lists, N

def prepare_einsum_inputs(mol):
        atm, bas, env   = mol._atm, mol._bas, mol._env
        n_atm, n_bas, N = atm.shape[0], bas.shape[0], mol.nao_nr()
        ao_loc          = np.cumsum(np.concatenate([np.zeros(1), (bas[:,1]*2+1) * bas[:,3] ])).astype(np.int32)
        n = n_bas
        c = 0
        dct = {}
        lst = []

        # Compute how many distinct integrals after 8x symmetry.
        num_calls = 0
        for i in range(n_bas):
                for j in range(n_bas):
                        for k in range(n_bas):
                                for l in range(n_bas):
                                        # * 8-fold symmetry, k>=l, k>=i>=j,
                                        if not ( i >= j and k >= l and i*j >= k * l): continue
                                        num_calls+=1

        do_lists = np.zeros((num_calls, 8), dtype=np.int32)
        indices  = np.zeros((num_calls, 8))

        #for i in tqdm(range(n)):
        for i in range(n):
                for j in range(n):
                        for k in range(n):
                                for l in range(n):
                                        if not (i >= j and k >= l and i*j >= k * l): continue

                                        # the output of integral (i,j,k,l) has shape (di, dj, dk, dl)
                                        # where di,dj,dk,dl are in {1,2,3,4,5} because assuming
                                        #  - only use {c,h,o,n}
                                        #  - represent electrons as pcq dataset 6-31G* (or sto3g/6-31g)
                                        di = ao_loc[i+1] - ao_loc[i]
                                        dj = ao_loc[j+1] - ao_loc[j]
                                        dk = ao_loc[k+1] - ao_loc[k]
                                        dl = ao_loc[l+1] - ao_loc[l]

                                        # knowing the shape of integral (i,j,k,l) we can fetch it.
                                        #integral = np.transpose( eri_s1[c][ :di*dj*dk*dl].reshape(dl,dk,dj,di), (3,2,1,0) )

                                        # we now need to compute where the current integral 'block' goes in
                                        # the final output matrix.
                                        i0 = ao_loc[i] - ao_loc[0]
                                        j0 = ao_loc[j] - ao_loc[0]
                                        k0 = ao_loc[k] - ao_loc[0]
                                        l0 = ao_loc[l] - ao_loc[0]

                                        indices[c] = np.array([ di, dj, dk, dl, i0, j0, k0, l0 ])

                                        do_list = [False]*8

                                        if not ( i0,j0,k0,l0 ) in dct:
                                                dct [ i0, j0, k0, l0 ] = 8
                                                do_list[0] = True

                                        if not ( i0, j0, l0, k0) in dct:
                                                dct [ i0, j0, l0, k0] = 9
                                                do_list[1] = True

                                        if not ( j0, i0, k0, l0 ) in dct:
                                                dct[ j0, i0, k0, l0 ] = 10
                                                do_list[2] = True

                                        if not ( j0, i0, l0, k0 ) in dct:
                                                dct[ j0, i0, l0, k0 ] = 11
                                                do_list[3] = True

                                        if not ( k0, l0, i0, j0 ) in dct:
                                                dct[ k0, l0, i0, j0 ] = 12
                                                do_list[4] = True

                                        if not ( k0, l0, j0, i0  ) in dct:
                                                dct[ k0, l0, j0, i0 ]  = 13
                                                do_list[5] = True

                                        if not ( l0, k0, i0, j0  ) in dct:
                                                dct[ l0, k0, i0, j0 ]  = 14
                                                do_list[6] = True
                                        if not ( l0, k0, j0, i0 ) in dct:
                                                dct [ l0, k0, j0, i0 ]  = 15
                                                do_list[7] = True

                                        do_lists[c] = np.array(do_list)
                                        c += 1


        tuple_indices  = tuple(map(tuple, indices.astype(np.int32).tolist()))
        tuple_do_lists = tuple(map(tuple, do_lists.astype(np.int32).tolist()))
        return tuple_indices, tuple_do_lists, N, num_calls

def compute_eri(mol, atom_str, eri_them, eri_them_s8):
        print("")
        print("###################")
        print("### compute eri ###")
        print("###################")

        # Shapes/sizes.
        atm, bas, env   = mol._atm, mol._bas, mol._env
        n_atm, n_bas, N = atm.shape[0], bas.shape[0], mol.nao_nr()
        ao_loc          = np.cumsum(np.concatenate([np.zeros(1), (bas[:,1]*2+1) * bas[:,3] ])).astype(np.int32)
        n_ao_loc        = np.prod(ao_loc.shape)
        shls_slice      = (0, n_bas, 0, n_bas, 0, n_bas, 0, n_bas)
        shape           = [1, N, N, N, N]

        # Initialize tensors for CPU libcint computation.
        buf     = np.zeros(np.prod(shape)*2)
        out     = np.zeros(shape)
        eri     = np.zeros(shape).reshape(-1)
        ipu_eri = np.zeros(shape).reshape(-1)

        # Fetch dtype from cpp file and cast tensors correspondingly.
        _type = open("cpu_int2e_sph.cpp", "r").read().split("#define input_type")[1].split("\n")[0]
        if "double" in _type: dtype = np.float64
        else:                 dtype = np.float32
        print("[dtype] %s"%(str(dtype)))
        buf, out, eri, ipu_eri, env = buf.astype(dtype), out.astype(dtype), eri.astype(dtype), ipu_eri.astype(dtype), env.astype(dtype)

        # The padded shape used to store output from all tiles.
        n_buf, n_eri, n_env = 81, 81, np.prod(env.shape)
        if args.basis == "6-31G*" or args.basis=="631gs": # 6-31G* has known bug, raise github issue if you want to use it.
                n_buf = 5**4
                n_eri = 5**4

        # Compute how many distinct integrals after 8x symmetry.
        num_calls = 0
        for i in range(n_bas):
                for j in range(n_bas):
                        for k in range(n_bas):
                                for l in range(n_bas):
                                        # * 8-fold symmetry, k>=l, k>=i>=j,
                                        if not ( i >= j and k >= l and i*j >= k * l): continue
                                        num_calls+=1

        print("[num calls] %i -> %i (with/without symmetry)"%(n_bas**4, num_calls))

        # Input/outputs for calling the IPU vertex.
        input_ijkl   = np.zeros((num_calls, 4),     dtype=np.int32)
        cpu_output   = np.zeros((num_calls, n_eri), dtype=np.float32)
        output_sizes = np.zeros((num_calls, 5))

        # Fill input_ijkl and output_sizes with the necessary indices.
        c = 0
        for i in range(n_bas):
                for j in range(n_bas):
                        for k in range(n_bas):
                                for l in range(n_bas):
                                        # * 8-fold symmetry, k>=l, k>=i>=j,
                                        if not ( i >= j and k >= l and i*j >= k * l): continue

                                        input_ijkl[c] = [i, j, k, l]

                                        di = ao_loc[i+1] - ao_loc[i]
                                        dj = ao_loc[j+1] - ao_loc[j]
                                        dk = ao_loc[k+1] - ao_loc[k]
                                        dl = ao_loc[l+1] - ao_loc[l]

                                        output_sizes[c] = [di, dj, dk, dl, di*dj*dk*dl]


                                        c += 1

        # Prepare IPU inputs.
        # Merge all int/float inputs in seperate arrays.
        input_floats = env.reshape(1, -1)
        input_ints   = np.zeros((1, 6+n_ao_loc +n_atm*6+n_bas*8), dtype=np.int32)
        start, stop = 0, 6
        input_ints[:, start:stop] = np.array( [n_eri, n_buf, n_atm, n_bas, n_env, n_ao_loc] )
        start, stop = start+6, stop+n_ao_loc
        input_ints[:, start:stop] = ao_loc.reshape(-1)
        start, stop = start+n_ao_loc, stop + n_atm*6
        input_ints[:, start:stop] = atm.reshape(-1)
        start, stop = start+n_atm*6, stop + n_bas*8
        input_ints[:, start:stop] = bas.reshape(-1)

        # Load vertex using TileJax.
        vertex_filename = osp.join(osp.dirname(__file__), "int2e_sph.cpp")
        int2e_sph = create_ipu_tile_primitive(
                "poplar_int2e_sph",
                "poplar_int2e_sph",
                inputs=["ipu_floats", "ipu_ints", "ipu_ij", "ipu_output", "tile_g", "tile_idx", "tile_buf"],
                outputs={"ipu_output": 3, "tile_g": 4, "tile_idx": 5, "tile_buf": 6},
                gp_filename=vertex_filename,
                perf_estimate=100,
        )

        # When num_calls < num_tiles*num_threads we can call the vertex once.
        # When num_calls > num_tiles*num_threads we need an additional for loop.
        # The two functions below handle this.

        def get_shapes(input_ijkl):
                i_sh, j_sh, k_sh, l_sh  = input_ijkl[0]
                BAS_SLOTS = 8
                NPRIM_OF = 2
                NCTR_OF = 3
                ANG_OF = 1
                GSHIFT = 4

                i_prim  = bas.reshape(-1)[BAS_SLOTS*i_sh + NPRIM_OF]
                j_prim  = bas.reshape(-1)[BAS_SLOTS*j_sh + NPRIM_OF]
                k_prim  = bas.reshape(-1)[BAS_SLOTS*k_sh + NPRIM_OF]
                l_prim  = bas.reshape(-1)[BAS_SLOTS*l_sh + NPRIM_OF]

                i_ctr   = bas.reshape(-1)[BAS_SLOTS * i_sh + NCTR_OF]
                j_ctr   = bas.reshape(-1)[BAS_SLOTS * j_sh + NCTR_OF]
                k_ctr   = bas.reshape(-1)[BAS_SLOTS * k_sh + NCTR_OF]
                l_ctr   = bas.reshape(-1)[BAS_SLOTS * l_sh + NCTR_OF]

                i_l     = bas.reshape(-1)[BAS_SLOTS * i_sh + ANG_OF]
                j_l     = bas.reshape(-1)[BAS_SLOTS * j_sh + ANG_OF]
                k_l     = bas.reshape(-1)[BAS_SLOTS * k_sh + ANG_OF]
                l_l     = bas.reshape(-1)[BAS_SLOTS * l_sh + ANG_OF]

                nfi  = (i_l+1)*(i_l+2)/2
                nfj  = (j_l+1)*(j_l+2)/2
                nfk  = (k_l+1)*(k_l+2)/2
                nfl  = (l_l+1)*(l_l+2)/2
                nf = nfi * nfk * nfl * nfj;
                n_comp = 1

                nc = i_ctr * j_ctr * k_ctr * l_ctr;
                lenl = nf * nc * n_comp;
                lenk = nf * i_ctr * j_ctr * k_ctr * n_comp;
                lenj = nf * i_ctr * j_ctr * n_comp;
                leni = nf * i_ctr * n_comp;
                len0 = nf * n_comp;

                ng = [0, 0, 0, 0, 0, 1, 1, 1];

                IINC=0
                JINC=1
                KINC=2
                LINC=3

                li_ceil = i_l + ng[IINC]
                lj_ceil = j_l + ng[JINC]
                lk_ceil = k_l + ng[KINC]
                ll_ceil = l_l + ng[LINC]
                nrys_roots = (li_ceil + lj_ceil + lk_ceil + ll_ceil)/2 + 1


                ibase = li_ceil > lj_ceil;
                kbase = lk_ceil > ll_ceil;
                if (nrys_roots <= 2):
                        ibase = 0;
                        kbase = 0;
                if (kbase) :
                        dlk = lk_ceil + ll_ceil + 1;
                        dll = ll_ceil + 1;
                else:
                        dlk = lk_ceil + 1;
                        dll = lk_ceil + ll_ceil + 1;

                if (ibase) :
                        dli = li_ceil + lj_ceil + 1;
                        dlj = lj_ceil + 1;
                else :
                        dli = li_ceil + 1;
                        dlj = li_ceil + lj_ceil + 1;

                g_stride_i = nrys_roots;
                g_stride_k = nrys_roots * dli;
                g_stride_l = nrys_roots * dli * dlk;
                g_stride_j = nrys_roots * dli * dlk * dll;
                g_size     = nrys_roots * dli * dlk * dll * dlj;
                gbits        = ng[GSHIFT];
                leng = g_size*3*((1<<gbits)+1);

                len = leng + lenl + lenk + lenj + leni + len0;

                di = i_l * 2 + 1;
                dj = j_l * 2 + 1;
                dk = k_l * 2 + 1;
                dl = l_l * 2 + 1;

                ni = (i_l*2+1) * i_ctr;
                nj = (j_l*2+1) * j_ctr;
                nk = (k_l*2+1) * k_ctr;
                nl = (l_l*2+1) * l_ctr;
                nfik = nfi * nfk;
                nfikl = nfik * nfl;
                dlj = dl * dj;
                ofj = ni * dj;

                ofk = ni * nj * dk;
                ofl = ni * nj * nk * dl;
                buflen = nfikl*dj;

                return len, nf, buflen


        @partial(jax.jit, backend="ipu")
        def compute_fn(input_floats, input_ints, input_ijkl):
                sizes, counts  = np.unique(output_sizes[:, -1], return_counts=True)
                sizes, counts = sizes.astype(np.int32), counts.astype(np.int32)

                cpu_outputs = []
                start_index  = 0
                for i, (size, count) in enumerate(zip(sizes, counts)):

                        cpu_output = jnp.zeros((count, size))
                        tiles = tuple(range(count))

                        tile_floats = tile_put_replicated(input_floats, tiles)
                        tile_ints   = tile_put_replicated(input_ints,   tiles)
                        cpu_output  = tile_put_sharded(   cpu_output,   tiles)
                        tile_ijkl   = tile_put_sharded(   input_ijkl[start_index: start_index+count],   tiles)

                        tile_g      = tile_put_replicated(jnp.zeros(3888+1),                      tiles)
                        tile_idx    = tile_put_replicated(jnp.zeros(3888+1, dtype = jnp.int32) ,  tiles)
                        tile_buf    = tile_put_replicated(jnp.zeros(1080*4+1) ,                   tiles)

                        tile_floats, start = ipu_cycle_count(tile_floats)
                        cpu_output, tile_g, _, _= tile_map(int2e_sph, tile_floats, tile_ints, tile_ijkl, cpu_output, tile_g, tile_idx, tile_buf)
                        tile_floats, stop = ipu_cycle_count(tile_floats)

                        start_index += count
                        cpu_outputs.append(cpu_output)

                return cpu_outputs, start, stop






        def wrap_cycles(func):
                def g( int2e_sph, tile_floats, tile_ints, input_ijkl, output, tile_g, tile_idx, tile_buf ):
                        ipu_output, tile_g, tile_idx, tile_buf = func(int2e_sph, tile_floats, tile_ints, input_ijkl, output, tile_g, tile_idx, tile_buf)
                        return ipu_output, tile_g, [0], [0]
                return g

        @partial(jax.jit, backend="ipu", static_argnums=(3,4))
        def compute_integrals(input_floats, input_ints, input_ijkl, num_calls, num_tiles):
                outs, cycles_start, cycles_stop, output = [], [], [], jnp.zeros((num_tiles, n_buf))


                tile_floats = tile_put_replicated(input_floats, tiles)
                tile_ints   = tile_put_replicated(input_ints,   tiles)
                # in 6-31G* we get a 5! this means 5**4 = 625

                # this takes ~12K = 50KiB
                tile_g      = tile_put_replicated(jnp.zeros(3888+1),                     tiles)
                tile_idx    = tile_put_replicated(jnp.zeros(3888+1, dtype = jnp.int32) ,  tiles)
                tile_buf    = tile_put_replicated(jnp.zeros(1080*4+1) ,                   tiles)

                for i in range( num_calls // num_tiles ):
                        start, stop = num_tiles*i, num_tiles*(i+1)

                        out, tile_g, cycle_start, cycle_stop = wrap_cycles(tile_map)(
                                                int2e_sph,
                                                tile_floats,
                                                tile_ints,
                                                tile_put_sharded(input_ijkl[start:stop], tiles),
                                                tile_put_sharded(output+i, tiles),
                                                tile_g,
                                                tile_idx,
                                                tile_buf
                        )
                        outs.append(out.array)
                        cycles_start.append(cycle_start)
                        cycles_stop.append(cycle_stop)

                elements_left = input_ijkl[num_tiles*(i+1):].shape[0]
                out, tile_g, cycle_start, cycle_stop = wrap_cycles(tile_map)(
                                int2e_sph,
                                tile_floats,
                                tile_ints,
                                tile_put_sharded(input_ijkl[-num_tiles:], tiles),
                                tile_put_sharded(output+i+1,              tiles),
                                tile_g,
                                tile_idx,
                                tile_buf
                )
                outs.append(out.array[-elements_left:])
                cycles_start.append(cycle_start)
                cycles_stop.append(cycle_stop)

                return outs, tile_g.array[:10, :10], cycles_start, cycles_stop


        sizes, counts  = np.unique(output_sizes[:, -1], return_counts=True)
        sizes, counts = sizes.astype(np.int32), counts.astype(np.int32)
        print("[integral size ] ", sizes )
        print("[integral count] ", counts)

        print("[matrix list] %i MB -> %i MB"%(num_calls*sizes[-1]/10**6, np.sum(sizes*counts/10**6)))

        if args.cpu:
                out3 = np.zeros(out.shape)
        else:
                num_threads = int(args.threads)
                ipu_num_tiles = 1472
                if num_calls >= ipu_num_tiles*num_threads:
                        # If enough calls allocate all threads and all tiles.
                        tiles = [i for i in range(1, ipu_num_tiles) for _ in range(num_threads)]
                else:
                        # If too few calls deal with remainder.
                        tiles = [i for i in range(num_calls//num_threads) for _ in range(num_threads)]
                        remainder = num_calls - int(num_calls//num_threads)*num_threads
                        if remainder != 0: tiles = tiles + [int(num_calls//num_threads)]*remainder

                num_tiles = len(tiles)
                print("[input]\t input_floats=%s=%i\tinput_ints=%s=%i\tcpu_output=%s=%i\tinput_ijkl=%s=%i (shape/kB)"%
                               (str(input_floats.shape), input_floats.nbytes/1000,
                                str(input_ints.shape),   input_ints.nbytes/1000,
                                str(cpu_output.shape),   cpu_output.nbytes/1000,
                                str(input_ijkl.shape),   input_ijkl.nbytes/1000))

                # Benchmark a single integral.
                # Allows us to iterate quickly on improving the C++ code on CPU/MK2.
                # This also runs the C++ variant in cpu_int2e_sph.cpp (this allows printing)
                if args.micro != -1:
                        # Fetch the specific integral we want to micro benchmark.
                        args.micro = int(args.micro)
                        input_ijkl = input_ijkl[args.micro: args.micro+1]
                        cpu_output = cpu_output[args.micro: args.micro+1]

                        # Run on CPU.
                        ret = libcgto.int2e_sph(
                                buf.ctypes.data_as(ctypes.c_void_p),
                                ctypes.c_int(np.prod(buf.shape)),
                                lib.c_null_ptr(),
                                (ctypes.c_int*4)(*input_ijkl.tolist()[0]),
                                atm.ctypes.data_as(ctypes.c_void_p),
                                ctypes.c_int(n_atm),
                                bas.ctypes.data_as(ctypes.c_void_p),
                                ctypes.c_int(n_bas),
                                env.ctypes.data_as(ctypes.c_void_p),
                                ctypes.c_int(np.prod(env.shape)))

                        # Run on IPU.
                        num_calls = 1
                        tiles     = (1,)

                        out3, start_cycles, stop_cycles = compute_fn( input_floats, input_ints, input_ijkl)

                        stop_cycles, start_cycles, out3 = np.asarray(stop_cycles), np.asarray(start_cycles), np.asarray(out3)
                        cycles                          = (stop_cycles[:, 0] - start_cycles[:, 0]).reshape(-1)
                        print("[Cycles M]")
                        print(cycles/10**6)
                        print("> IPU")
                        print(out3[out3!=0])
                        print("> CPU")
                        print(buf [buf !=0])
                        print("> Diff")
                        print(np.max(np.abs(out3[out3!=0]-buf[buf!=0])))
                        assert np.allclose(out3[out3!=0].reshape(-1), buf[buf!=0].reshape(-1), atol=1e-5)
                        print("PASSED!")

                # The number of integrals smaller than all tiles/threads.
                # One call is sufficient.
                elif num_calls < 1471*num_threads and False:
                        print("[For Loop] False")

                        indxs               = np.argsort(output_sizes[:, -1])
                        sorted_output_sizes = output_sizes[indxs]
                        input_ijkl          = input_ijkl[indxs]

                        _out3, start_cycles, stop_cycles = compute_fn( input_floats, input_ints, input_ijkl)
                        #out3, start_cycles, stop_cycles = np.asarray(out3), np.asarray(start_cycles), np.asarray(stop_cycles)
                        start_cycles, stop_cycles = np.asarray(start_cycles), np.asarray(stop_cycles)
                        __out3 = [np.asarray(a) for a in _out3]

                        outs = []
                        for out in __out3:
                                a = np.concatenate([out.astype(np.float32), np.zeros((out.shape[0], 625-out.shape[1]), dtype=np.float32)], axis=1)
                                outs.append( a)

                        ___out3 = np.concatenate(outs, axis=0)
                        def inverse_permutation(a):
                                b = np.arange(a.shape[0])
                                b[a] = b.copy()
                                return b
                        indxs_inv = inverse_permutation(indxs)

                        out3 = ___out3[indxs_inv]

                else:
                        print("[For Loop] True ")

                        indxs               = np.argsort(output_sizes[:, -1])
                        sorted_output_sizes = output_sizes[indxs]
                        input_ijkl          = input_ijkl[indxs]

                        sizes, counts  = np.unique(output_sizes[:, -1], return_counts=True)
                        sizes, counts = sizes.astype(np.int32), counts.astype(np.int32)
                        start_index = 0
                        inputs = []
                        shapes = []
                        for i, (size, count) in enumerate(zip(sizes, counts)):
                                a = input_ijkl[start_index: start_index+count]
                                tuples = tuple(map(tuple, a))
                                inputs.append(tuples)
                                start_index += count

                        tuple_ijkl = tuple(inputs)
                        input_ijkl = inputs

                        for i in range(len(sizes)):
                                shapes.append( get_shapes(input_ijkl[i]) )

                        def inverse_permutation(a):
                                b = np.arange(a.shape[0])
                                b[a] = b.copy()
                                return b
                        indxs_inv = inverse_permutation(indxs)

                        _input_floats, _input_ints, _tuple_ijkl, _shapes, _sizes, _counts, indxs, indxs_inv, _ = prepare_integrals_2_inputs(mol)
                        assert np.allclose(input_floats, _input_floats)
                        assert np.allclose(input_ints, _input_ints)
                        assert all([ np.allclose(a, b) for a,b in zip(tuple_ijkl, _tuple_ijkl)])
                        assert np.allclose(shapes, _shapes)
                        assert np.allclose(sizes, _sizes)
                        assert np.allclose(counts, _counts)

                        if not args.precompute:

                                #_out3, cycles_start, cycles_stop, _, ___out3 = compute_integrals_2( _input_floats, _input_ints, _tuple_ijkl, _shapes, _sizes, _counts, tuple(indxs_inv.tolist()))
                                ___out3, cycles_start, cycles_stop = compute_integrals_2( _input_floats, _input_ints, _tuple_ijkl, _shapes, _sizes, _counts, tuple(indxs_inv.tolist()), num_threads=int(args.threads_int), v=args.intv)
                                cycles_start, cycles_stop          = np.asarray(cycles_start), np.asarray(cycles_stop)

                                out3 = np.array((5,5))

                                # Save so if we do -precompute we don't have to wait for compute! => 30s to 10s iteration!
                                nps = [np.asarray(a) for a in ___out3]
                                print([a.shape for a in nps])
                                for _num, a in enumerate(___out3):
                                        a = np.asarray(a)
                                        #print(a)
                                        os.makedirs("tmp/", exist_ok=True)
                                        np.savez("tmp/%i_%i.npz"%(_num, np.sum(_shapes)), a=a) # to allow saving different runs!

                                if args.skipmult:
                                        exit()


                        else:
                                out3 = np.zeros(((247, 81)))
                                cycles_start, cycles_stop = np.array(5), np.array(5)

                                ___out3 = []
                                for i in range(5):
                                        ___out3.append( np.load("tmp/%i_%i.npz"%(i, np.sum(_shapes)))["a"] )

                        try:

                                cycles = np.concatenate(cycles)
                                fig, ax = plt.subplots(1,3)
                                ax[0].plot(np.sort(cycles))
                                ax[1].plot(np.sort(cycles/output_sizes[:, -1]))
                                ax[2].plot(np.sort(output_sizes[:, -1]))
                                ax[0].set_ylabel("cycles")
                                ax[0].set_xlabel("integral")
                                ax[1].set_ylabel("cycles/integral_output_size")
                                ax[1].set_xlabel("integral")
                                plt.tight_layout()
                                plt.savefig("plots/sorted_cycles.jpg")

                        except:
                                cycles = []
                        if not args.load: np.savez("cycles.npz", cycles=cycles)


                        print("[list of vects]", cpu_output.nbytes, np.sum([a.nbytes for a in out3]) , cpu_output.nbytes / np.sum([a.nbytes for a in out3]))

        tensor = np.zeros((N, N, N, N))

        from tqdm import tqdm
        outs = []
        _max = -1
        c = 0
        skip = 0
        do = 0


        for i in tqdm(range(shls_slice[1])):
                for j in range(shls_slice[3]):
                        for k in range(shls_slice[5]):
                                for l in range(shls_slice[7]):
                                        if not ( i >= j and k >= l and i*j >= k * l):
                                                skip+=1
                                                continue
                                        else: do += 1

                                        ret = libcgto.int2e_sph(
                                                buf.ctypes.data_as(ctypes.c_void_p),
                                                ctypes.c_int(np.prod(buf.shape)),
                                                lib.c_null_ptr(),
                                                (ctypes.c_int*4)(*[i,j,k,l]),
                                                atm.ctypes.data_as(ctypes.c_void_p),
                                                ctypes.c_int(n_atm),
                                                bas.ctypes.data_as(ctypes.c_void_p),
                                                ctypes.c_int(n_bas),
                                                env.ctypes.data_as(ctypes.c_void_p),
                                                ctypes.c_int(np.prod(env.shape)))

                                        assert ret == 1 # this is case where C++ returns !empty in the end.

                                        # so always writing here should be okay!
                                        if ret:
                                                ni = ao_loc[shls_slice[1]] - ao_loc[0]
                                                nj = ao_loc[shls_slice[3]] - ao_loc[0]
                                                nk = ao_loc[shls_slice[5]] - ao_loc[0]
                                                nl = ao_loc[shls_slice[7]] - ao_loc[0]

                                                nij  = ni * nj
                                                nkl  = nk * nl
                                                neri = nij * nkl

                                                i0 = ao_loc[i] - ao_loc[0]
                                                j0 = ao_loc[j] - ao_loc[0]
                                                k0 = ao_loc[k] - ao_loc[0]
                                                l0 = ao_loc[l] - ao_loc[0]

                                                di = ao_loc[i+1] - ao_loc[i]
                                                dj = ao_loc[j+1] - ao_loc[j]
                                                dk = ao_loc[k+1] - ao_loc[k]
                                                dl = ao_loc[l+1] - ao_loc[l]

                                                dij   = di * dj
                                                dijk  = di * dj * dk
                                                dijkl = di * dj * dk * dl

                                                if _max > dijkl: _max = dijkl

                                                cpu_output[c, :di*dj*dk*dl] = buf[:di*dj*dk*dl]
                                                c += 1


        indices = np.zeros((cpu_output.shape[0], 8))
        def transform(eri_s8):
                print(eri_s8.shape)
                # transforms sparse_eri to dense_eri
                # accepts sparse input as list of vectors or matrix.
                # output:  (n, n, n, n) matrix without symmetries
                if type(eri_s8) != type([]):
                        print(eri_s8.shape)
                        if eri_s8.shape[0] == 1: eri_s8 = eri_s8[0]
                        print(eri_s8.shape)
                out = np.zeros((N, N, N, N)) # 14
                c = 0

                for i in tqdm(range(shls_slice[1])): # 10
                        for j in range(shls_slice[3]):
                                for k in range(shls_slice[5]):
                                        for l in range(shls_slice[7]):
                                                if not (i >= j and k >= l and i*j >= k * l): continue

                                                # the output of integral (i,j,k,l) has shape (di, dj, dk, dl)
                                                # where di,dj,dk,dl are in {1,2,3,4,5} assuming
                                                #  - only use {c,h,o,n}
                                                #  - represent electrons as pcq dataset
                                                di = ao_loc[i+1] - ao_loc[i]
                                                dj = ao_loc[j+1] - ao_loc[j]
                                                dk = ao_loc[k+1] - ao_loc[k]
                                                dl = ao_loc[l+1] - ao_loc[l]

                                                # knowing the shape of integral (i,j,k,l) we can fetch it.
                                                #integral = np.transpose( cpu_output[c, :di*dj*dk*dl].reshape(dl,dk,dj,di), (3,2,1,0) )
                                                integral = np.transpose( eri_s8[c][ :di*dj*dk*dl].reshape(dl,dk,dj,di), (3,2,1,0) )

                                                # we now need to compute where the current integral 'block' goes in
                                                # the final output matrix.
                                                i0 = ao_loc[i]   - ao_loc[0]
                                                j0 = ao_loc[j]   - ao_loc[0]
                                                k0 = ao_loc[k]   - ao_loc[0]
                                                l0 = ao_loc[l]   - ao_loc[0]

                                                indices[c] = np.array([di, dj, dk, dl, i0, j0, k0, l0])
                                                c += 1

                                                # we then store the block of integrals in whatever places.
                                                out[ i0:i0+di, j0:j0+dj, k0:k0+dk, l0:l0+dl ] = integral #ijkl
                                                out[ i0:i0+di, j0:j0+dj, l0:l0+dl, k0:k0+dk ] = np.transpose(integral, (0,1,3,2)) #ijlk
                                                out[ j0:j0+dj, i0:i0+di, k0:k0+dk, l0:l0+dl ] = np.transpose(integral, (1,0,2,3)) #jikl
                                                out[ j0:j0+dj, i0:i0+di, l0:l0+dl, k0:k0+dk ] = np.transpose(integral, (1,0,3,2)) #jilk
                                                out[ k0:k0+dk, l0:l0+dl, i0:i0+di, j0:j0+dj ] = np.transpose(integral, (2,3,0,1)) #klij
                                                out[ k0:k0+dk, l0:l0+dl, j0:j0+dj, i0:i0+di ] = np.transpose(integral, (2,3,1,0)) #klji
                                                out[ l0:l0+dl, k0:k0+dk, i0:i0+di, j0:j0+dj ] = np.transpose(integral, (3,2,0,1)) #lkij
                                                out[ l0:l0+dl, k0:k0+dk, j0:j0+dj, i0:i0+di ] = np.transpose(integral, (3,2,1,0)) #lkji

                return out

        if cpu_output.shape == out3.shape: print("[error cpu/ipu s1]", np.max(np.abs(cpu_output-out3)))
        else: print("[error cpu/ipu s1] SHAPES NOT EQUAL")

        print("[error cpu_output s8] ", np.max(np.abs(transform(cpu_output).reshape(-1) - eri_them.reshape(-1))))

        if args.cpu: exit()

        def direct_mult(sparse_s8, dm):
                print(sparse_s8.shape)
                n = shls_slice[1]
                c = 0
                _vj, _vk = np.zeros(dm.shape), np.zeros(dm.shape)
                vj, vk   = np.zeros((sparse_s8.shape[0],)+dm.shape), np.zeros((sparse_s8.shape[0],)+dm.shape)
                dct = {}
                lst = []

                do_lists = np.zeros((sparse_s8.shape[0], 8), dtype=np.int32)

                for i in tqdm(range(n)):
                        for j in range(n):
                                for k in range(n):
                                        for l in range(n):

                                                if not (i >= j and k >= l and i*j >= k * l): continue

                                                # the output of integral (i,j,k,l) has shape (di, dj, dk, dl)
                                                # where di,dj,dk,dl are in {1,2,3,4,5} assuming
                                                #  - only use {c,h,o,n}
                                                #  - represent electrons as pcq dataset 6-31G*
                                                di = ao_loc[i+1] - ao_loc[i]
                                                dj = ao_loc[j+1] - ao_loc[j]
                                                dk = ao_loc[k+1] - ao_loc[k]
                                                dl = ao_loc[l+1] - ao_loc[l]

                                                # knowing the shape of integral (i,j,k,l) we can fetch it.
                                                integral = np.transpose( sparse_s8[c][ :di*dj*dk*dl].reshape(dl,dk,dj,di), (3,2,1,0) )

                                                # we now need to compute where the current integral 'block' goes in
                                                # the final output matrix.
                                                i0 = ao_loc[i] - ao_loc[0]
                                                j0 = ao_loc[j] - ao_loc[0]
                                                k0 = ao_loc[k] - ao_loc[0]
                                                l0 = ao_loc[l] - ao_loc[0]
                                                #print(i0,j0,k0,l0) # this is different!

                                                indices[c] = np.array([ di, dj, dk, dl, i0, j0, k0, l0 ])

                                                do_list = [False]*8

                                                if not ( i0,j0,k0,l0 ) in dct:
                                                        dct [ i0, j0, k0, l0 ] = 8
                                                        do_list[0] = True

                                                if not ( i0, j0, l0, k0) in dct:
                                                        dct [ i0, j0, l0, k0] = 9
                                                        do_list[1] = True

                                                if not ( j0, i0, k0, l0 ) in dct:
                                                        dct[ j0, i0, k0, l0 ] = 10
                                                        do_list[2] = True

                                                if not ( j0, i0, l0, k0 ) in dct:
                                                        dct[ j0, i0, l0, k0 ] = 11
                                                        do_list[3] = True

                                                if not ( k0, l0, i0, j0 ) in dct:
                                                        dct[ k0, l0, i0, j0 ] = 12
                                                        do_list[4] = True

                                                if not ( k0, l0, j0, i0  ) in dct:
                                                        dct[ k0, l0, j0, i0 ]  = 13
                                                        do_list[5] = True

                                                if not ( l0, k0, i0, j0  ) in dct:
                                                        dct[ l0, k0, i0, j0 ]  = 14
                                                        do_list[6] = True
                                                if not ( l0, k0, j0, i0 ) in dct:
                                                        dct [ l0, k0, j0, i0 ]  = 15
                                                        do_list[7] = True

                                                do_lists[c] = np.array(do_list)
                                                c += 1

                c= 0
                sums = np.zeros((sparse_s8.shape[0]))
                _sums = np.zeros((sparse_s8.shape[0], 2))
                _debug = np.zeros((sparse_s8.shape[0], 8))

                for i in tqdm(range(n)):
                        if args.fast: break
                        for j in range(n):
                                for k in range(n):
                                        for l in range(n):

                                                if not (i >= j and k >= l and i*j >= k * l): continue
                                                # could we change this to for c in range num_calls?
                                                # every tile can get exactly this and just deal with the c'th part of eri_s8!

                                                di, dj, dk, dl, i0, j0, k0, l0  = [int(a) for a in indices[c]]
                                                #integral        = np.transpose( eri_s1[c][ :di*dj*dk*dl].copy().reshape(dl,dk,dj,di), (3,2,1,0) )
                                                integral        =  sparse_s8[c][ :di*dj*dk*dl].copy().reshape(dl,dk,dj,di)
                                                #integral        =  eri_s1[c][ :di*dj*dk*dl]#.copy().reshape(dl,dk,dj,di)
                                                #eri_s1[c] = 0
                                                #eri_s1[c, :di*dj*dk*dl] = integral.reshape(-1)

                                                do_list = do_lists[c]

                                                for ci, _i in enumerate(range(i0, i0+di)):
                                                        for cj, _j in enumerate(range(j0, j0+dj)):
                                                                for ck, _k in enumerate(range(k0, k0+dk)):
                                                                        for cl, _l in enumerate(range(l0, l0+dl)):
                                                                                val = integral.reshape(-1)[ci+cj*di+ck*dj*di+cl*dj*dk*di]

                                                                                if do_list[0]:
                                                                                        #_vj[_k, _l] += dm[_j,_i] * integral[ci, cj, ck, cl]
                                                                                        #_vk[_i, _l] += dm[_j,_k] * integral[ci, cj, ck, cl]
                                                                                        _vj.reshape(-1)[_k*N+ _l] += dm.reshape(-1)[_j*N+_i] * val
                                                                                        _vk.reshape(-1)[_i*N+ _l] += dm.reshape(-1)[_j*N+_k] * val

                                                                                        vj[c].reshape(-1)[_k*N+ _l] += dm.reshape(-1)[_j*N+_i] * val
                                                                                        vk[c].reshape(-1)[_i*N+ _l] += dm.reshape(-1)[_j*N+_k] * val
                                                                                        sums[c] += val

                                                                                        _sums[c, 0] += dm.reshape(-1)[_j*N+_i] #* val
                                                                                        _sums[c, 1] += dm.reshape(-1)[_j*N+_k] #* val

                                                                                        _debug[c, 4] += val
                                                                                        _debug[c, 5] += dm.reshape(-1)[_j*N+_i]
                                                                                        _debug[c, 6] += dm.reshape(-1)[_j*N+_k] *val

                                                                                if do_list[1]:
                                                                                        #_vj[_l, _k] += dm[_j,_i] * integral[ci, cj, ck, cl]
                                                                                        #_vk[_i, _k] += dm[_j,_l] * integral[ci, cj, ck, cl]

                                                                                        _vj.reshape(-1)[_l*N+ _k] += dm.reshape(-1)[_j*N+_i] * val
                                                                                        _vk.reshape(-1)[_i*N+ _k] += dm.reshape(-1)[_j*N+_l] * val

                                                                                        vj[c].reshape(-1)[_l*N+ _k] += dm.reshape(-1)[_j*N+_i] * val
                                                                                        vk[c].reshape(-1)[_i*N+ _k] += dm.reshape(-1)[_j*N+_l] * val
                                                                                        sums[c] += val
                                                                                if do_list[2]:
                                                                                        #_vj[_k, _l] += dm[_i,_j] * integral[ci, cj, ck, cl]
                                                                                        #_vk[_j, _l] += dm[_i,_k] * integral[ci, cj, ck, cl]
                                                                                        _vj.reshape(-1)[_k*N+ _l] += dm.reshape(-1)[_i*N+_j] * val
                                                                                        _vk.reshape(-1)[_j*N+ _l] += dm.reshape(-1)[_i*N+_k] * val

                                                                                        vj[c].reshape(-1)[_k*N+ _l] += dm.reshape(-1)[_i*N+_j] * val
                                                                                        vk[c].reshape(-1)[_j*N+ _l] += dm.reshape(-1)[_i*N+_k] * val
                                                                                        sums[c] += val
                                                                                if do_list[3]:
                                                                                        #_vj[_l, _k] += dm[_i,_j] * integral[ci, cj, ck, cl]
                                                                                        #_vk[_j, _k] += dm[_i,_l] * integral[ci, cj, ck, cl]

                                                                                        _vj.reshape(-1)[_l*N+ _k] += dm.reshape(-1)[_i*N+_j] * val
                                                                                        _vk.reshape(-1)[_j*N+ _k] += dm.reshape(-1)[_i*N+_l] * val

                                                                                        vj[c].reshape(-1)[_l*N+ _k] += dm.reshape(-1)[_i*N+_j] * val
                                                                                        vk[c].reshape(-1)[_j*N+ _k] += dm.reshape(-1)[_i*N+_l] * val
                                                                                        sums[c] += val
                                                                                if do_list[4]:
                                                                                        #_vj[_i, _j] += dm[_l,_k] * integral[ci, cj, ck, cl]
                                                                                        #_vk[_k, _j] += dm[_l,_i] * integral[ci, cj, ck, cl]

                                                                                        _vj.reshape(-1)[_i*N+ _j] += dm.reshape(-1)[_l*N+_k] * val
                                                                                        _vk.reshape(-1)[_k*N+ _j] += dm.reshape(-1)[_l*N+_i] * val

                                                                                        vj[c].reshape(-1)[_i*N+ _j] += dm.reshape(-1)[_l*N+_k] * val
                                                                                        vk[c].reshape(-1)[_k*N+ _j] += dm.reshape(-1)[_l*N+_i] * val
                                                                                        sums[c] += val
                                                                                if do_list[5]:
                                                                                        #_vj[_j, _i] += dm[_l,_k] * integral[ci, cj, ck, cl]
                                                                                        #_vk[_k, _i] += dm[_l,_j] * integral[ci, cj, ck, cl]

                                                                                        _vj.reshape(-1)[_j*N+ _i] += dm.reshape(-1)[_l*N+_k] * val
                                                                                        _vk.reshape(-1)[_k*N+ _i] += dm.reshape(-1)[_l*N+_j] * val

                                                                                        vj[c].reshape(-1)[_j*N+ _i] += dm.reshape(-1)[_l*N+_k] * val
                                                                                        sums[c] += val
                                                                                        vk[c].reshape(-1)[_k*N+ _i] += dm.reshape(-1)[_l*N+_j] * val
                                                                                if do_list[6]:
                                                                                        #_vj[_i, _j] += dm[_k,_l] * integral[ci, cj, ck, cl]
                                                                                        #_vk[_l, _j] += dm[_k,_i] * integral[ci, cj, ck, cl]

                                                                                        _vj.reshape(-1)[_i*N+ _j] += dm.reshape(-1)[_k*N+_l] * val
                                                                                        _vk.reshape(-1)[_l*N+ _j] += dm.reshape(-1)[_k*N+_i] * val

                                                                                        vj[c].reshape(-1)[_i*N+ _j] += dm.reshape(-1)[_k*N+_l] * val
                                                                                        vk[c].reshape(-1)[_l*N+ _j] += dm.reshape(-1)[_k*N+_i] * val
                                                                                        sums[c] += val
                                                                                if do_list[7]:
                                                                                        #_vj[_j, _i] += dm[_k,_l] * integral[ci, cj, ck, cl]
                                                                                        #_vk[_l, _i] += dm[_k,_j] * integral[ci, cj, ck, cl]

                                                                                        _vj.reshape(-1)[_j*N+ _i] += dm.reshape(-1)[_k*N+_l] * val
                                                                                        _vk.reshape(-1)[_l*N+ _i] += dm.reshape(-1)[_k*N+_j] * val

                                                                                        vj[c].reshape(-1)[_j*N+ _i] += dm.reshape(-1)[_k*N+_l] * val
                                                                                        vk[c].reshape(-1)[_l*N+ _i] += dm.reshape(-1)[_k*N+_j] * val
                                                                                        sums[c] += val




                                                c+= 1



                return _vj, _vk, do_lists

        N = mol.nao
        np.random.seed(42)
        dm  = np.random.normal(0,1, (N, N))
        dm  = dm + dm.T
        dm  = np.linalg.qr(dm )[0]

        vj, vk, do_lists = direct_mult(cpu_output, dm)
        vj, vk = np.asarray(vj), np.asarray(vk)

        if args.fast:
                vj = np.einsum('ijkl,ji->kl', eri_them, dm)
                vk = np.einsum('ijkl,jk->il', eri_them, dm)



        vertex_filename = osp.join(osp.dirname(__file__), "int2e_sph.cpp")
        poplar_direct_s1 = create_ipu_tile_primitive(
                "poplar_direct_s1",
                "poplar_direct_s1",
                inputs=["integral", "indices", "do_list", "dm"],
                outputs={ "vj": 3, "vk": 3},
                gp_filename=vertex_filename,
                perf_estimate=100,
        )

        eri_s1 = cpu_output
        num_tiles = eri_s1.shape[0]
        ipu_num_tiles = jax.devices()[0].num_tiles
        tiles = tuple(np.array(np.arange(num_tiles)%ipu_num_tiles).tolist())

        tuple_indices  = tuple(map(tuple, indices.astype(np.int32).tolist()))
        tuple_do_lists = tuple(map(tuple, do_lists.astype(np.int32).tolist()))

        _tuple_indices, _tuple_do_lists, _N = prepare_ipu_direct_mult_inputs(eri_s1.shape[0], mol)

        assert np.allclose(_tuple_indices, tuple_indices)
        assert np.allclose(_tuple_do_lists, tuple_do_lists)
        assert np.allclose(_N, N)

        print(indxs_inv)

        ipu_vj, ipu_vk = jax.jit(ipu_direct_mult, backend="ipu", static_argnums=(2,3,4,5,6,7,8))(
                                                ___out3,
                                                dm.astype(np.float32),
                                                tuple_indices,
                                                tuple_do_lists, N, eri_s1.shape[0],
                                                tuple(indxs_inv.tolist()),
                                                tuple(indxs.tolist()),
                                                threads=int(args.threads)
                                                )
        ipu_vj, ipu_vk = np.asarray(ipu_vj),np.asarray(ipu_vk )

        eri = eri_them
        vj0 = np.einsum('ijkl,ji->kl', eri, dm)
        vk0 = np.einsum('ijkl,jk->il', eri, dm)

        print("[full v dot] np.max(np.abs( diff ))", np.max(np.abs(vj0-vj)), np.max(np.abs(vk0 - vk)))
        print("[full v dot] np.max(np.abs( diff ))", np.max(np.abs(vj0-ipu_vj)), np.max(np.abs(vk0 - ipu_vk)), "[ipu]")

        vj, vk = ipu_vj, ipu_vk

        fig ,ax = plt.subplots(1,2, figsize=(12, 4))
        ax[0].plot(np.abs(vj0.reshape(-1)), "bo", ms=5, label="reference")
        ax[0].plot(np.abs(vj.reshape(-1)), "wx", ms=2, label="us/ipu")
        ax[0].plot(np.abs((vj-vj0).reshape(-1)), "r^", ms=2, label="error")
        ax[0].set_yscale("log")
        ax[0].legend()

        ax[1].plot(np.abs(vk0.reshape(-1)), "bo", ms=5, label="referenc")
        ax[1].plot(np.abs(vk.reshape(-1)), "wx", ms=2, label="us/ipu")
        ax[1].plot(np.abs((vk-vk0)).reshape(-1), "r^", ms=2, label="error")
        ax[1].set_yscale("log")
        ax[1].legend()

        plt.tight_layout()
        plt.savefig("direct2.jpg")

        exit()

def runjk(dm1, mol, nao, *namejk):
    type = open("cpu_int2e_sph.cpp", "r").read().split("#define ddtype")[1].split("\n")[0]

    if "double" in type:
        dtype = np.float64
    else:
        dtype = np.float32

    #print(dtype)

    dm1    = dm1.astype(dtype)
    c_atm  = numpy.array(mol._atm, dtype=numpy.int32)
    c_bas  = numpy.array(mol._bas, dtype=numpy.int32)
    c_env  = numpy.array(mol._env, dtype=dtype)
    ao_loc = numpy.asarray(mol.ao_loc_nr(), dtype=numpy.int32)
    vjk = numpy.zeros((2, nao, nao)).astype(dtype)

    # I think this pointer stuff is the one breaking in float32?
    fjk    = (ctypes.c_void_p*(2))()
    vjkptr = (ctypes.c_void_p*(2))()

    vjkptr[0] = vjk[0].ctypes.data_as(ctypes.c_void_p)
    fjk[0]    = ctypes.c_void_p(_ctypes.dlsym(libcgto._handle, "CVHFnrs8_ji_s1kl"))

    vjkptr[1] = vjk[1].ctypes.data_as(ctypes.c_void_p)
    fjk[1]    = ctypes.c_void_p(_ctypes.dlsym(libcgto._handle, "CVHFnrs8_jk_s1il"))

    libcgto.CVHFnr_direct_drv(
         fjk,
         dm1.ctypes.data_as(ctypes.c_void_p) ,
         vjkptr,
         ctypes.c_int(2),
         ctypes.c_int(1),
         (ctypes.c_int*8)(*([0, mol.n_bas]*4)) ,
         ao_loc.ctypes.data_as(ctypes.c_void_p),
         c_atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(c_atm.shape[0]) ,
         c_bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(c_bas.shape[0]),
         c_env.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(np.prod(c_env.shape))
    )

    return vjk.reshape(2,nao,nao)

from pyscf import lib
libcvhf2 = lib.load_library('libcvhf')

def test(str):
        print("[Molecule]")
        print(str.replace(";", "\n"))
        mol = pyscf.gto.mole.Mole()
        print("")

        try:
                mol.build(atom=str, unit="Bohr", basis=args.basis, spin=0, verbose=0)
        except:
                return


        them, us = [], []
        print("[PySCF full] ", end="")
        for _ in range(5):
                t0 = time.time()
                eri_them = mol.intor("int2e_sph")
                t1 = time.time()
                them.append(t1-t0)
                print("%5f \t"%(them[-1]), end="")
                if them[-1] > 1: break

        print(eri_them.astype(np.float32).nbytes/10**6, "MB  ", np.prod(eri_them.shape), eri_them.shape, end ="")
        print("")

        N = mol.nao
        np.random.seed(42)
        dm  = np.random.normal(0,1, (N, N))
        dm  = dm + dm.T
        vj0 = np.einsum('ijkl,ji->kl', eri_them, dm)
        vk0 = np.einsum('ijkl,jk->il', eri_them, dm)

        print("[PySCF s8  ] ", end="")
        for _ in range(5):
                t0 = time.time()
                eri_them_s8 = mol.intor("int2e_sph", aosym="s8")
                t1 = time.time()
                them.append(t1-t0)
                print("%5f\t"%them[-1], end="")
                if them[-1] > 1: break
        print(eri_them_s8.astype(np.float32).nbytes/10**6, "MB  ", np.prod(eri_them_s8.shape), eri_them_s8.shape, end ="")


        print("")
        print("[basis set]", args.basis)

        repeat = 1
        for _ in range(repeat):
                t0 = time.time()
                eri_us = compute_eri(mol, str, eri_them, eri_them_s8)
                t1 = time.time()
                us.append(t1-t0)
                print("%10f\t"%us[-1], end="")





if __name__ == "__main__":

        import numpy as np
        import pyscf
        import time
        import ctypes
        from pyscf import lib

        import argparse
        parser = argparse.ArgumentParser(description='Arguments for Density Functional Theory. ')
        parser.add_argument('-cpu',       action="store_true", help="Only run C++ code on cpu. ")
        parser.add_argument('-precompute',    action="store_true", help='Whether to load precomputed integrals for ipu_direct_mult; allows faster iteration. ')
        parser.add_argument('-num',       default=10,          type=int,   help='Run the first "num" test molecules. ')
        parser.add_argument('-id',        default=-1,          type=int,   help='Run only test molecule "id". ')
        parser.add_argument('-str',       default="",          help='Molecule string, e.g., "H 0 0 0; H 0 0 1; O 1 0 0; "')
        parser.add_argument('-basis',     default="STO-3G",    help='Basis set')
        parser.add_argument('-graphene',  action="store_true",  help='Graphene')
        parser.add_argument('-methane',   action="store_true",  help='simulate methane . ')
        parser.add_argument('-H',         action="store_true",  help='simple hydrogen system. ')
        parser.add_argument('-O',         action="store_true",  help='Just one oxygen')
        parser.add_argument('-C',         default=-1, type=int,  help='Number of carbons from C20. ')
        parser.add_argument('-gdb',       default=-1, type=int,  help='GDB variant to use. ')
        parser.add_argument('-Be',        action="store_true",  help='Just one Be atom.')
        parser.add_argument('-skipmult',        action="store_true",  help='only do integrals, useful for profiling. ')
        parser.add_argument('-Ne',        action="store_true",  help='Just one Be atom.')
        parser.add_argument('-he',        action="store_true",  help="Just do a single He atom, fastest possible case. ")
        parser.add_argument('-micro',     default=-1, help="Do micro benchmark; takes an integer and does as many integrals as integers. ")
        parser.add_argument('-threads',  default=1, help="number of threads to parallelize over in each tile (for mult/einsum); ")
        parser.add_argument('-threads_int',  default=1, help="number of threads to parallelize over in each tile (for integrals); ")
        parser.add_argument('-intv',  default=1, type=int, help="integral code version ")
        parser.add_argument('-load',      action="store_true", help="Load schedule. ")
        parser.add_argument('-compare',   action="store_true", help="Compare iteration by iteration. ")
        parser.add_argument('-benzene',   action="store_true", help='Compute benzene.')
        parser.add_argument('-maldehyde',       action="store_true", help="Compute ethanol . ")
        parser.add_argument('-fast',       action="store_true", help="Skip some part of the test cases. ")

        args = parser.parse_args()

        import sys
        sys.argv = sys.argv[:1]

        if args.str != "": test(args.str)
        elif args.he:      test(str="He 0 0 0; ")
        elif args.Ne:      test(str="Ne 0 0 0; ")
        elif args.H:       test(str="H 0 0 0; H 1 1 1; ")
        elif args.methane: test(str="C 0 0 0; H 0 0 1; H 1 0 0; H 0 1 0; H 1 1 0;")

        elif args.O:       test(str="F 0 0 0; F 0 0 1; ")

        elif args.maldehyde:
                test("O 5.4641   -0.5600    0.0000; \
                      O 2.0000   -0.5600    0.0000; \
                      C 3.7320   -0.5600    0.0000; \
                      C 4.5981   -0.0600    0.0000; \
                      C 2.8660   -0.0600    0.0000; \
                      H 3.3335   -1.0350    0.0000; \
                      H 4.1306   -1.0350    0.0000; \
                      H 4.5981    0.5600    0.0000; \
                      H 2.8660    0.5600    0.0000;")

        elif args.gdb > 0:

            if args.gdb == 9:  args.smiles = [a for a in open("../gdb/gdb11_size09_sorted.csv", "r").read().split("\n")][:-1]

            print("Length GDB: ", len(args.smiles))

            smile = args.smiles[int(args.id)]

            angstrom_to_bohr = 1.88973

            from rdkit import Chem
            from rdkit.Chem import AllChem
            b = Chem.MolFromSmiles(smile)
            b = Chem.AddHs(b)
            atoms = [atom.GetSymbol() for atom in b.GetAtoms()]

            AllChem.EmbedMolecule(b)

            locs = b.GetConformer().GetPositions() * angstrom_to_bohr
            atom_string, string = get_atom_string(" ".join(atoms), locs)

            print(string)
            test(string)

        elif args.C > 0:

                _str = ";".join("C     1.56910  -0.65660  -0.93640;\
        C     1.76690   0.64310  -0.47200;\
        C     0.47050  -0.66520  -1.79270;\
        C     0.01160   0.64780  -1.82550;\
        C     0.79300   1.46730  -1.02840;\
        C    -0.48740  -1.48180  -1.21570;\
        C    -1.56350  -0.65720  -0.89520;\
        C    -1.26940   0.64900  -1.27670;\
        C    -0.00230  -1.96180  -0.00720;\
        C    -0.76980  -1.45320   1.03590;\
        C    -1.75760  -0.63800   0.47420;\
        C     1.28780  -1.45030   0.16290;\
        C     1.28960  -0.65950   1.30470;\
        C     0.01150  -0.64600   1.85330;\
        C     1.58300   0.64540   0.89840;\
        C     0.48480   1.43830   1.19370;\
        C    -0.50320   0.64690   1.77530;\
        C    -1.60620   0.67150   0.92310;\
        C    -1.29590   1.48910  -0.16550;\
        C    -0.01020   1.97270  -0.00630;".split(";")[:args.C])

                test(str=_str)

        elif args.graphene:
                str = "C	-4.928	-4.26777318984971	0; C	-3.696	-4.97906872149133	0; C	-2.464	-4.26777318984971	0; C	-4.928	-2.84518212656648	0; C	-3.696	-2.13388659492486	0; C	-4.928	0	0;"
                test(str)

        elif args.co2:
                test("C 0 0 0; O 0 0 1; O 1 0 0;")
