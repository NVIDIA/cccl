// SPDX-FileCopyrightText: Copyright (c) 2011-2021, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

/**
 * \file
 * Utilities for interacting with the opaque CUDA __nv_bfloat16 type
 */

#include <cuda_bf16.h>

#include <cub/util_type.cuh>

#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include <cstdint>
#include <iosfwd>

#ifdef __GNUC__
// There's a ton of type-punning going on in this file.
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

/******************************************************************************
 * bfloat16_t
 ******************************************************************************/

/**
 * Host-based fp16 data type compatible and convertible with __nv_bfloat16
 */
struct bfloat16_t
{
  uint16_t __x;

  /// Constructor from __nv_bfloat16
  __host__ __device__ __forceinline__ explicit bfloat16_t(const __nv_bfloat16& other)
  {
    __x = reinterpret_cast<const uint16_t&>(other);
  }

  /// Constructor from integer
  __host__ __device__ __forceinline__ explicit bfloat16_t(int a)
  {
    *this = bfloat16_t(float(a));
  }

  /// Constructor from std::size_t
  __host__ __device__ __forceinline__ explicit bfloat16_t(std::size_t a)
  {
    *this = bfloat16_t(float(a));
  }

  /// Constructor from double
  __host__ __device__ __forceinline__ explicit bfloat16_t(double a)
  {
    *this = bfloat16_t(float(a));
  }

  /// Constructor from unsigned long long int
  template <typename T,
            typename = typename ::cuda::std::enable_if<
              ::cuda::std::is_same<T, unsigned long long int>::value
              && (!::cuda::std::is_same<std::size_t, unsigned long long int>::value)>::type>
  __host__ __device__ __forceinline__ explicit bfloat16_t(T a)
  {
    *this = bfloat16_t(float(a));
  }

  /// Default constructor
  bfloat16_t() = default;

  /// Constructor from float
  __host__ __device__ __forceinline__ explicit bfloat16_t(float a)
  {
    // Reference:
    // https://github.com/pytorch/pytorch/blob/44cc873fba5e5ffc4d4d4eef3bd370b653ce1ce1/c10/util/BFloat16.h#L51
    uint16_t ir;
    if (a != a)
    {
      ir = UINT16_C(0x7FFF);
    }
    else
    {
      union
      {
        uint32_t U32;
        float F32;
      };

      F32                    = a;
      uint32_t rounding_bias = ((U32 >> 16) & 1) + UINT32_C(0x7FFF);
      ir                     = static_cast<uint16_t>((U32 + rounding_bias) >> 16);
    }
    this->__x = ir;
  }

  /// Cast to __nv_bfloat16
  __host__ __device__ __forceinline__ operator __nv_bfloat16() const
  {
    return reinterpret_cast<const __nv_bfloat16&>(__x);
  }

  /// Cast to float
  __host__ __device__ __forceinline__ operator float() const
  {
    float f     = 0;
    uint32_t* p = reinterpret_cast<uint32_t*>(&f);
    *p          = uint32_t(__x) << 16;
    return f;
  }

  /// Get raw storage
  __host__ __device__ __forceinline__ uint16_t raw() const
  {
    return this->__x;
  }

  /// Equality
  __host__ __device__ __forceinline__ friend bool operator==(const bfloat16_t& a, const bfloat16_t& b)
  {
    return (a.__x == b.__x);
  }

  /// Inequality
  __host__ __device__ __forceinline__ friend bool operator!=(const bfloat16_t& a, const bfloat16_t& b)
  {
    return (a.__x != b.__x);
  }

  /// Assignment by sum
  __host__ __device__ __forceinline__ bfloat16_t& operator+=(const bfloat16_t& rhs)
  {
    *this = bfloat16_t(float(*this) + float(rhs));
    return *this;
  }

  /// Multiply
  __host__ __device__ __forceinline__ bfloat16_t operator*(const bfloat16_t& other) const
  {
    return bfloat16_t(float(*this) * float(other));
  }

  /// Add
  __host__ __device__ __forceinline__ bfloat16_t operator+(const bfloat16_t& other) const
  {
    return bfloat16_t(float(*this) + float(other));
  }

  /// Sub
  __host__ __device__ __forceinline__ bfloat16_t operator-(const bfloat16_t& other) const
  {
    return bfloat16_t(float(*this) - float(other));
  }

  /// Less-than
  __host__ __device__ __forceinline__ bool operator<(const bfloat16_t& other) const
  {
    return float(*this) < float(other);
  }

  /// Less-than-equal
  __host__ __device__ __forceinline__ bool operator<=(const bfloat16_t& other) const
  {
    return float(*this) <= float(other);
  }

  /// Greater-than
  __host__ __device__ __forceinline__ bool operator>(const bfloat16_t& other) const
  {
    return float(*this) > float(other);
  }

  /// Greater-than-equal
  __host__ __device__ __forceinline__ bool operator>=(const bfloat16_t& other) const
  {
    return float(*this) >= float(other);
  }

  /// numeric_traits<bfloat16_t>::max
  __host__ __device__ __forceinline__ static bfloat16_t(max)()
  {
    uint16_t max_word = 0x7F7F;
    return reinterpret_cast<bfloat16_t&>(max_word);
  }

  /// numeric_traits<bfloat16_t>::lowest
  __host__ __device__ __forceinline__ static bfloat16_t lowest()
  {
    uint16_t lowest_word = 0xFF7F;
    return reinterpret_cast<bfloat16_t&>(lowest_word);
  }
};

/******************************************************************************
 * I/O stream overloads
 ******************************************************************************/

/// Insert formatted \p bfloat16_t into the output stream
inline std::ostream& operator<<(std::ostream& out, const bfloat16_t& x)
{
  out << (float) x;
  return out;
}

/// Insert formatted \p __nv_bfloat16 into the output stream
inline std::ostream& operator<<(std::ostream& out, const __nv_bfloat16& x)
{
  return out << bfloat16_t(x);
}

/******************************************************************************
 * Traits overloads
 ******************************************************************************/

namespace cuda
{
template <>
inline constexpr bool is_floating_point_v<bfloat16_t> = true;
}

template <>
class cuda::std::numeric_limits<bfloat16_t>
{
public:
  static constexpr bool is_specialized = true;

  static _CCCL_HOST_DEVICE _CCCL_FORCEINLINE bfloat16_t max()
  {
    return bfloat16_t(numeric_limits<__nv_bfloat16>::max());
  }

  static _CCCL_HOST_DEVICE _CCCL_FORCEINLINE bfloat16_t min()
  {
    return bfloat16_t(numeric_limits<__nv_bfloat16>::min());
  }

  static _CCCL_HOST_DEVICE _CCCL_FORCEINLINE bfloat16_t lowest()
  {
    return bfloat16_t(numeric_limits<__nv_bfloat16>::lowest());
  }
};

CUB_NAMESPACE_BEGIN

template <>
struct NumericTraits<bfloat16_t> : BaseTraits<FLOATING_POINT, true, uint16_t, bfloat16_t>
{};

CUB_NAMESPACE_END

#ifdef __GNUC__
#  pragma GCC diagnostic pop
#endif
