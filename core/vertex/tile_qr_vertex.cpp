// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

#include "intrinsics_utils.hpp"

using namespace poplar;

/* popc -O2 -I tessellate/tile/vertex\
     tessellate/tile/vertex/tile_qr_vertex.cpp \
     -o tessellate/tile/vertex/tile_qr_vertex.gp
*/

class [[poplar::constraint("elem(*x) != elem(*y)")]] DotProduct1dVertex
    : public MultiVertex {
 public:
  using T = float;
  using T2 = float2;
  // Using `uint16` seems to be generating more efficient loops?
  using IndexType = unsigned short;

  static constexpr size_t MIN_ALIGN = 8;

  Input<Vector<T, poplar::VectorLayout::ONE_PTR, MIN_ALIGN>>
      x;  // (N,) x vector
  Input<Vector<T, poplar::VectorLayout::ONE_PTR, MIN_ALIGN>>
      y;  // (N,) y vector

  Input<Vector<IndexType, poplar::VectorLayout::ONE_PTR>>
      worker_offsets;  // (7,) number threads + 1.
  Output<Vector<T, poplar::VectorLayout::ONE_PTR>> partials;  // float result.

  bool compute(unsigned wid) {
    // Always assuming size % 2 == 0
    const IndexType wstart = worker_offsets[wid];
    const IndexType wend = worker_offsets[wid + 1];
    const IndexType wsize = wend - wstart;

    T2* ptr_tmp_partials_f2 = reinterpret_cast<T2*>(partials.data()) + wid;
    // Nothing to do in this worker thread.
    if (wstart == wend) {
      ipu::store_postinc(&ptr_tmp_partials_f2, T2{0, 0}, 1);
      return true;
    }
    // X and Y input pointers.
    const T2* ptr_inxdata_f2 = reinterpret_cast<const T2*>(x.data()) + wstart;
    const T2* ptr_inydata_f2 = reinterpret_cast<const T2*>(y.data()) + wstart;
    T2 partial = T2{0, 0};

    for (IndexType idx = 0; idx != wsize; ++idx) {
      // TODO: use ld2x64pace + tapack instructions?
      const T2 xin = ipu::load_postinc(&ptr_inxdata_f2, 1);
      const T2 yin = ipu::load_postinc(&ptr_inydata_f2, 1);
      // popc seems to recognize this pattern and optimize it.
      // Using directly ipu::fma intrinsics leads to poor performance!?
      partial += xin * yin;
    }
    ipu::store_postinc(&ptr_tmp_partials_f2, partial, 1);
    return true;
  }
};

/**
 * @brief Vertex computing the correction vector in the QR algorithm.
 */
class QRCorrectionVectorVertex : public MultiVertex {
 public:
  using T = float;
  Input<Vector<T, poplar::VectorLayout::SPAN>> Rcol;      // (N,) R column.
  Input<Vector<T, poplar::VectorLayout::ONE_PTR>> sdiag;  // (N,) R diag. sign.

  Output<Vector<T, poplar::VectorLayout::ONE_PTR>>
      v;  // (N,) QR correction vector (not normalized)
  Output<Vector<T, poplar::VectorLayout::ONE_PTR>>
      vrescale;  // (1,) QR correction vector rescaling (2 / norm)

  const unsigned col_idx;  // R column index.

  // Use static variables as easy sync. mechanisms between worker threads.
  // see: https://graphcore.slack.com/archives/C013LPHPX61/p1647443852989259
  static T shared_partial_sqnorms[6];

  QRCorrectionVectorVertex();

  bool compute(unsigned wid) {
    const unsigned num_workers = 6;
    const unsigned step = num_workers * 2;
    const unsigned size = Rcol.size();

    const unsigned col_idx_rem = col_idx % 2;
    const unsigned col_idx_mrem = col_idx - col_idx_rem;
    const unsigned col_idx_prem = col_idx + col_idx_rem;

    const T initial_rcol_val = Rcol[col_idx];
    const T initial_rcol_val_sq = initial_rcol_val * initial_rcol_val;

    // FIRST: SET SHARED STATE between workers.
    shared_partial_sqnorms[wid] = -1;

    const float2 zeros_f2{0, 0};
    float2* ptr_outdata_f2 = reinterpret_cast<float2*>(v.data()) + wid;
    // Push to col_idx_prem, may write one zero too much, but does not matter!
    float2* ptr_outdata_end_f2 = reinterpret_cast<float2*>(&v[col_idx_prem]);
    // First chunk of v initialized with zeros.
    while (ptr_outdata_f2 < ptr_outdata_end_f2) {
      ipu::store_postinc(&ptr_outdata_f2, zeros_f2, num_workers);
    }

    float2 partials_f2{0, 0};
    const float2* ptr_indata_f2 =
        reinterpret_cast<const float2*>(&Rcol[col_idx_prem]) + wid;
    ptr_outdata_f2 = reinterpret_cast<float2*>(&v[col_idx_prem]) + wid;
    ptr_outdata_end_f2 = reinterpret_cast<float2*>(&v[size]);
    // Copy Rcol data and accumulate squared norm.
    while (ptr_outdata_f2 < ptr_outdata_end_f2) {
      const float2 v = ipu::load_postinc(&ptr_indata_f2, num_workers);
      partials_f2 += v * v;
      ipu::store_postinc(&ptr_outdata_f2, v, num_workers);
    }
    T partial = partials_f2[0] + partials_f2[1];

    // GLOBAL STATE shared by all workers.
    shared_partial_sqnorms[wid] = partial;

    if (wid == 0) {
      // Special case of odd R column index.
      // On thread 0: correction to squared normed depending on `col_idx_rem`
      T norm_squared = partial + col_idx_rem * initial_rcol_val_sq;
      // Accumulate & wait.
      for (unsigned w = 1; w < num_workers; ++w) {
        // Avoid compiler optimizer with volatile pointer.
        volatile T* ptr_partial = &shared_partial_sqnorms[w];
        while (*ptr_partial < 0) {
        }
        norm_squared += shared_partial_sqnorms[w];
      }

      // Compute the norm.
      const T norm = std::sqrt(norm_squared);
      // Change the entry of v that corresponds to the diagonal element of R.
      const auto update_vidx_val = initial_rcol_val - norm * sdiag[col_idx];
      // Re-writing the full new value is faster than updating.
      v[col_idx] = update_vidx_val;

      // Update the squared norm of v.
      norm_squared -= initial_rcol_val_sq;
      norm_squared += update_vidx_val * update_vidx_val;

      // Vector rescaling for QR householder update.
      vrescale[0] = T(2) / norm_squared;
    }
    return true;
  }
};

float QRCorrectionVectorVertex::shared_partial_sqnorms[6] = {-1};

/**
 * @brief Vertex implementing the inplace householder (row) update in the QR
 * algorithm. NOTE: the vertex is only updating the sub-slice of x corresponding
 * to v.
 *
 * More specifically: x[end-len(v)+i] -= scale1[0] * scale2[0] * v[i]
 *
 * NOTE: poplar::constraint here to make sure x and v are not part of the same
 * memory bank, allowing simultaneous loads (see `ld2x64pace` instruction).
 */
class [[poplar::constraint(
    "elem(*x) != elem(*v)")]] QRHouseholderRowUpdateVertex
    : public MultiVertex {
 public:
  using T = float;
  // Using `uint16` seems to be generating more efficient loops?
  using IndexType = unsigned short;

  InOut<Vector<T, poplar::VectorLayout::ONE_PTR, 8>> x;  // (N,) row of Q or R
  Input<Vector<T, poplar::VectorLayout::SPAN, 8>>
      v;  // (M,) v correction vector

  // Passing 2 scaling factors is more efficient for the QR implementation.
  // Avoids another full pass on the v vector in the vertex it is constructed.
  Input<Vector<T, poplar::VectorLayout::ONE_PTR>>
      scale1;  // (1,) first scaling factor.
  Input<Vector<T, poplar::VectorLayout::ONE_PTR>>
      scale2;  // (1,) 2nd scaling factor.

  Input<Vector<IndexType, poplar::VectorLayout::ONE_PTR>>
      worker_offsets;  // (7,) threads work size + 1.

  IndexType start_idx;  // X start idx. Must be a multiple of 4 (for bank
                        // alignment aspects).

  bool compute(unsigned wid) {
    // Always assuming size % 2 == 0
    constexpr unsigned ptr_step = 1;
    const IndexType wstart = worker_offsets[wid];
    const IndexType wend = worker_offsets[wid + 1];
    const IndexType wsize = wend - wstart;

    // Set the $TAS register with the proper scale.
    const T s = -scale1[0] * scale2[0];
    // __builtin_ipu_put_tas(s);
    __ipu_and_ipumodel_tas tas;
    tas.put(s);

    // Nothing to do in this worker thread.
    if (wstart == wend) {
      return true;
    }
    // X and v IO pointers.
    const float2* ptr_inxdata_f2 =
        reinterpret_cast<const float2*>(&x[start_idx]) + wstart;
    float2* ptr_outxdata_f2 = reinterpret_cast<float2*>(&x[start_idx]) + wstart;
    const float2* ptr_vdata_f2 =
        reinterpret_cast<const float2*>(&v[0]) + wstart;

    float2 xin, vin, rtmp, rout;
    // First vectors loading.
    xin = ipu::load_postinc(&ptr_inxdata_f2, ptr_step);
    vin = ipu::load_postinc(&ptr_vdata_f2, ptr_step);
    // TODO: use ld2x64pace + tapack instructions.
    for (IndexType idx = 1; idx != wsize; ++idx) {
      rtmp = tas.f32v2axpy(xin, vin);
      // rtmp = __builtin_ipu_f32v2axpy(xin, vin);
      // Grouping here seems to help the compiler optimising loads?
      xin = ipu::load_postinc(&ptr_inxdata_f2, ptr_step);
      vin = ipu::load_postinc(&ptr_vdata_f2, ptr_step);
      rout = tas.f32v2axpy(rtmp, rtmp);
      // rout = __builtin_ipu_f32v2axpy(rtmp, rtmp);
      ipu::store_postinc(&ptr_outxdata_f2, rout, ptr_step);
    }
    // Finish the loop, getting the last computation.
    // rtmp = __builtin_ipu_f32v2axpy(xin, vin);
    // rout = __builtin_ipu_f32v2axpy(rtmp, rtmp);
    rtmp = tas.f32v2axpy(xin, vin);
    rout = tas.f32v2axpy(rtmp, rtmp);
    ipu::store_postinc(&ptr_outxdata_f2, rout, ptr_step);

    return true;
  }
};
