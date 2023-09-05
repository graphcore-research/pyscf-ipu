// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#pragma once

#ifdef __IPU__
// Use the IPU intrinsics
#include <ipu_memory_intrinsics>
#include <ipu_vector_math>
#define NAMESPACE ipu
#else
#include "ipu_model_types.hpp"
// Use the std functions
#include <cmath>

#include "ipu_model_types.hpp"
#define NAMESPACE std
#endif

#ifdef __IPU__
#define SUPERVISOR_TARGET __attribute__((target("supervisor")))
#else
#define SUPERVISOR_TARGET
#endif

// #define ALWAYS_INLINE __attribute__((always_inline))
#define ALWAYS_INLINE inline

#ifdef __IPU__

/**
 * @brief Efficient division by 6, on IPU hardware. Up to 98,304.
 */
template <typename T>
ALWAYS_INLINE T ipu_div_by_6(T n) noexcept {
  return (n * 0xaaab) >> 18;
}

/**
 * @brief IPU intrinsics, for setting up the $TAS register.
 */
ALWAYS_INLINE void __builtin_ipu_put_tas(float v) noexcept {
  // TAS register, used for __builtin_ipu_f32v2axpy.
  asm volatile(
      R"l( uput $TAS, %[sv]
        )l"
      :
      : [sv] "r"(v)
      :);
}

/**
 * @brief IPU cmac f32 instruction.
 */
ALWAYS_INLINE void __builtin_ipu_f32v2cmac(float2 x, float2 y) noexcept {
  asm volatile(
      R"l( f32v2mac %[x], %[y]
        )l"
      :
      : [x] "r"(x), [y] "r"(y)
      :);
}

template <typename T>
ALWAYS_INLINE float ld32(const T* address, unsigned offset) {
  float result;
  // TODO - Use intrinsic/builtin for this when one becomes available
  asm volatile(
      R"l(  ld32 %[result], %[address], %[offset]
      )l"
      : [result] "=r"(result)
      : [address] "r"(address), [offset] "r"(offset)
      :);
  return result;
}

struct __ipu_and_ipumodel_tas {
  void put(float v) { __builtin_ipu_put_tas(v); }
  float2 f32v2axpy(float2 const& x, float2 const& y) {
    return __builtin_ipu_f32v2axpy(x, y);
  }
};

#else

#include <limits>

namespace ipu {
// Implementations of IPU intrinsics for IPUModel

// https://docs.graphcore.ai/projects/poplar-api/en/latest/doxygen/namespaceipu.html#aa1a33d2be82a6b73549badf896cfd88e
template <class T>
void store_postinc(T** a, T const& v, int i) {
  **a = v;
  (*a) += i;
}

// https://docs.graphcore.ai/projects/poplar-api/en/latest/doxygen/namespaceipu.html#acb144a365e4027998954ee1e9d98e0d3
template <class T>
T load_postinc(T const** a, int i) {
  T const* p = *a;
  (*a) += i;
  return *p;
}

// https://docs.graphcore.ai/projects/poplar-api/en/latest/doxygen/namespaceipu.html#a2a81ec4b6956ea14fe230a137178ff48
template <class T, size_t N>
IpuVector<T, N> fma(IpuVector<T, N> const& x, IpuVector<T, N> const& y,
                    IpuVector<T, N> const& z) {
  IpuVector<T, N> ret = z;
  for (size_t i = 0; i < N; ++i) ret[i] += x[i] * y[i];
  return ret;
}

}  // namespace ipu

// Reflect IPU's AXPY semantics in a way that is IPUModel compatible
// IPU-only usage:
//   __builtin_ipu_put_tas(v);
//   z_prev = __builtin_ipu_f32v2axpy(x, y)
//
// IPUModel-compatible usage:
//   __ipu_and_ipumodel_tas tas;
//   tas.put(v);
//   z_prev = tas.f32v2axpy(x, y)
//
// https://docs.graphcore.ai/projects/poplar-api/en/latest/ipu_intrinsics/ipu_builtins.html#_CPPv423__builtin_ipu_f32v2axpy6float26float2
struct __ipu_and_ipumodel_tas {
  float tas;
  float2 prev;

  __ipu_and_ipumodel_tas() : tas{0}, prev{0, 0} {}

  void put(float v) { tas = v; }

  float2 f32v2axpy(float2 const& x, float2 const& y) {
    const auto res = prev;
    prev = float2{
        // TODO: understand ordering!?
        // tas * x[0] + y[0],
        // tas * x[1] + y[1],
        tas * y[0] + x[0],
        tas * y[1] + x[1],
    };
    return res;
  }
};

// And give useful error messages when people port from IPU to IPUModel, e.g.
/* clang-format off */ // need these error messages on one line
/*
/workspaces/tessellate-ipu/tessellate/tile/vertex/intrinsics_utils.hpp:166:3: error: static_assert failed due to requirement '__ipu_false<IpuVector<float, 2>>()': *** Replace __builtin_ipu_f32v2axpy with __ipu_and_ipumodel_tas for TAS handling on IPUModel.
  static_assert(__ipu_false<T>(), "*** Replace __builtin_ipu_f32v2axpy with __ipu_and_ipumodel_tas for TAS handling on IPUModel.");
  ^             ~~~~~~~~~~~~~~~~
/workspaces/tessellate-ipu/tessellate/tile/vertex/tile_qr_vertex.cpp:231:12: note: in instantiation of function template specialization '__builtin_ipu_f32v2axpy<IpuVector<float, 2>>' requested here
    rout = __builtin_ipu_f32v2axpy(rtmp, rtmp);
*/
template <typename T>
constexpr bool __ipu_false() {
  return !std::is_same<T, T>::value;
}

template <typename T>
void __builtin_ipu_put_tas(T v) {
  static_assert(__ipu_false<T>(), "*** Replace __builtin_ipu_put_tas with __ipu_and_ipumodel_tas for TAS handling on IPUModel.");
}

template <typename T>
T __builtin_ipu_f32v2axpy(T const& x, T const& y) {
  static_assert(__ipu_false<T>(), "*** Replace __builtin_ipu_f32v2axpy with __ipu_and_ipumodel_tas for TAS handling on IPUModel.");
  return T{};
}
// clang-format on

#endif
