// Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#pragma once

#ifndef __IPU__
#include <array>
#include <cstddef>

template <class T, size_t N>
struct IpuVector : public std::array<T, N> {
 public:
  using BaseArrayType = std::array<T, N>;
  // No explicit construct/copy/destroy for aggregate type.

  // Basic unary/binary operations on vectors.
  IpuVector& operator+=(const IpuVector& rhs) noexcept {
    for (size_t i = 0; i < N; ++i) this->operator[](i) += rhs[i];
    return *this;
  }
  IpuVector& operator-=(const IpuVector& rhs) noexcept {
    for (size_t i = 0; i < N; ++i) this->operator[](i) -= rhs[i];
    return *this;
  }
  IpuVector& operator*=(const IpuVector& rhs) noexcept {
    for (size_t i = 0; i < N; ++i) this->operator[](i) *= rhs[i];
    return *this;
  }
  IpuVector& operator/=(const IpuVector& rhs) noexcept {
    for (size_t i = 0; i < N; ++i) this->operator[](i) /= rhs[i];
    return *this;
  }

  friend IpuVector operator+(const IpuVector& lhs,
                             const IpuVector& rhs) noexcept {
    IpuVector ret;
    for (size_t i = 0; i < N; ++i) ret[i] = lhs[i] + rhs[i];
    return ret;
  }
  friend IpuVector operator-(const IpuVector& lhs,
                             const IpuVector& rhs) noexcept {
    IpuVector ret;
    for (size_t i = 0; i < N; ++i) ret[i] = lhs[i] - rhs[i];
    return ret;
  }
  friend IpuVector operator*(const IpuVector& lhs,
                             const IpuVector& rhs) noexcept {
    IpuVector ret;
    for (size_t i = 0; i < N; ++i) ret[i] = lhs[i] * rhs[i];
    return ret;
  }
  friend IpuVector operator/(const IpuVector& lhs,
                             const IpuVector& rhs) noexcept {
    IpuVector ret;
    for (size_t i = 0; i < N; ++i) ret[i] = lhs[i] / rhs[i];
    return ret;
  }
};

// IPU vector typedefs.
using float2 = IpuVector<float, 2>;
using float4 = IpuVector<float, 4>;

using char2 = IpuVector<char, 2>;
using uchar2 = IpuVector<unsigned char, 2>;
using char4 = IpuVector<char, 4>;
using uchar4 = IpuVector<unsigned char, 4>;

using short2 = IpuVector<short, 2>;
using ushort2 = IpuVector<unsigned short, 2>;
using short4 = IpuVector<short, 4>;
using ushort4 = IpuVector<unsigned short, 4>;

using int2 = IpuVector<int, 2>;
using uint2 = IpuVector<unsigned int, 2>;
using int4 = IpuVector<int, 4>;
using uint4 = IpuVector<unsigned int, 4>;

using long2 = IpuVector<long, 2>;
using long4 = IpuVector<long, 4>;

#endif
