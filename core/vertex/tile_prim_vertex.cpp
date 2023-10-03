// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

#include "intrinsics_utils.hpp"

using namespace poplar;

/**
 * @brief Tile barrier vertex: not doing anything, but setting a barrier
 * to put constraints on Poplar program workflow.
 *
 * TODO: support multiple tensor datatypes. Issue: Poplar general reinterpret
 * cast.
 * Compilation for all supported targets:
 *      popc -O2 tessellate/tile/vertex/tile_prim_vertex.cpp
 * -I./tessellate/tile/vertex -o
 * tessellate/tile/vertex/tile_prim_vertex.gp
 */
template <typename T>
class TileDataBarrierVertex : public SupervisorVertex {
  static const bool needsAlignWorkers = false;

 public:
  // data gated by the barrier.
  Vector<InOut<Vector<T, poplar::VectorLayout::ONE_PTR, 1>>,
         poplar::VectorLayout::ONE_PTR, 1>
      data;

  SUPERVISOR_TARGET bool compute() {
    // Hihihi, not doing anything!
    return true;
  }
};

// explicit instantiations
template class TileDataBarrierVertex<bool>;
template class TileDataBarrierVertex<unsigned char>;
template class TileDataBarrierVertex<signed char>;
template class TileDataBarrierVertex<unsigned short>;
template class TileDataBarrierVertex<short>;
template class TileDataBarrierVertex<unsigned>;
template class TileDataBarrierVertex<int>;
template class TileDataBarrierVertex<float>;
template class TileDataBarrierVertex<half>;

/**
 * @brief On-tile memcpy vertex. Useful for explicit copies
 * on tile (but not as performant as Poplar optimized ones).
 */
template <typename T>
class TileMemcpyVertex : public MultiVertex {
 public:
  // Conservative alignment assumption?
  // static constexpr int AlignSize = sizeof(T) * 2;
  static constexpr int AlignSize = 8;
  static constexpr int NumWorkers = 6;

  Input<Vector<T, poplar::VectorLayout::SPAN, AlignSize>> in;  // (N,) in vector
  Output<Vector<T, poplar::VectorLayout::SPAN, AlignSize>>
      out;  // (N,) out vector

  bool compute(unsigned wid) {
    if (wid != 0) {
      return true;
    }
    // Start with the most basic loop!
    for (int i = 0; i < in.size(); ++i) {
      out[i] = in[i];
    }
    return true;
  }
};

// explicit instantiations
template class TileMemcpyVertex<bool>;
template class TileMemcpyVertex<char>;
template class TileMemcpyVertex<unsigned char>;
template class TileMemcpyVertex<signed char>;
template class TileMemcpyVertex<unsigned short>;
template class TileMemcpyVertex<short>;
template class TileMemcpyVertex<unsigned>;
template class TileMemcpyVertex<int>;
template class TileMemcpyVertex<float>;
template class TileMemcpyVertex<half>;
