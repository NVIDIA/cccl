// SPDX-FileCopyrightText: Copyright (c) 2011, Duane Merrill. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2011-2019, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

/**
 * \file
 * Utilities for interacting with the opaque CUDA __half type
 */

#include <cuda_fp16.h>

#include <cub/util_type.cuh>

#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include <cstdint>
#include <cstring>
#include <iosfwd>

#ifdef __GNUC__
// There's a ton of type-punning going on in this file.
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

/******************************************************************************
 * half_t
 ******************************************************************************/

/**
 * Host-based fp16 data type compatible and convertible with __half
 */
// TODO(bgruber): drop this when CTK 12.2 is the minimum, since it provides __host__ __device__ operators of __half
struct half_t
{
  uint16_t __x;

  /// Constructor from __half
  __host__ __device__ __forceinline__ explicit half_t(const __half& other)
  {
    __x = reinterpret_cast<const uint16_t&>(other);
  }

  /// Constructor from integer
  __host__ __device__ __forceinline__ explicit half_t(int a)
  {
    *this = half_t(float(a));
  }

  /// Constructor from std::size_t
  __host__ __device__ __forceinline__ explicit half_t(std::size_t a)
  {
    *this = half_t(float(a));
  }

  /// Constructor from double
  __host__ __device__ __forceinline__ explicit half_t(double a)
  {
    *this = half_t(float(a));
  }

  /// Constructor from unsigned long long int
  template <typename T,
            typename = typename ::cuda::std::enable_if<
              ::cuda::std::is_same<T, unsigned long long int>::value
              && (!::cuda::std::is_same<std::size_t, unsigned long long int>::value)>::type>
  __host__ __device__ __forceinline__ explicit half_t(T a)
  {
    *this = half_t(float(a));
  }

  /// Default constructor
  half_t() = default;

  /// Constructor from float
  __host__ __device__ __forceinline__ explicit half_t(float a)
  {
    // Stolen from Norbert Juffa
    uint32_t ia = *reinterpret_cast<uint32_t*>(&a);
    uint16_t ir;

    ir = (ia >> 16) & 0x8000;

    if ((ia & 0x7f800000) == 0x7f800000)
    {
      if ((ia & 0x7fffffff) == 0x7f800000)
      {
        ir |= 0x7c00; /* infinity */
      }
      else
      {
        ir = 0x7fff; /* canonical NaN */
      }
    }
    else if ((ia & 0x7f800000) >= 0x33000000)
    {
      int32_t shift = (int32_t) ((ia >> 23) & 0xff) - 127;
      if (shift > 15)
      {
        ir |= 0x7c00; /* infinity */
      }
      else
      {
        ia = (ia & 0x007fffff) | 0x00800000; /* extract mantissa */
        if (shift < -14)
        { /* denormal */
          ir |= ia >> (-1 - shift);
          ia = ia << (32 - (-1 - shift));
        }
        else
        { /* normal */
          ir |= ia >> (24 - 11);
          ia = ia << (32 - (24 - 11));
          ir = static_cast<uint16_t>(ir + ((14 + shift) << 10));
        }
        /* IEEE-754 round to nearest of even */
        if ((ia > 0x80000000) || ((ia == 0x80000000) && (ir & 1)))
        {
          ir++;
        }
      }
    }

    this->__x = ir;
  }

  /// Cast to __half
  __host__ __device__ __forceinline__ operator __half() const
  {
    return reinterpret_cast<const __half&>(__x);
  }

  /// Cast to float
  __host__ __device__ __forceinline__ operator float() const
  {
    // Stolen from Andrew Kerr

    int sign        = ((this->__x >> 15) & 1);
    int exp         = ((this->__x >> 10) & 0x1f);
    int mantissa    = (this->__x & 0x3ff);
    std::uint32_t f = 0;

    if (exp > 0 && exp < 31)
    {
      // normal
      exp += 112;
      f = (sign << 31) | (exp << 23) | (mantissa << 13);
    }
    else if (exp == 0)
    {
      if (mantissa)
      {
        // subnormal
        exp += 113;
        while ((mantissa & (1 << 10)) == 0)
        {
          mantissa <<= 1;
          exp--;
        }
        mantissa &= 0x3ff;
        f = (sign << 31) | (exp << 23) | (mantissa << 13);
      }
      else if (sign)
      {
        f = 0x80000000; // negative zero
      }
      else
      {
        f = 0x0; // zero
      }
    }
    else if (exp == 31)
    {
      if (mantissa)
      {
        f = 0x7fffffff; // not a number
      }
      else
      {
        f = (0xff << 23) | (sign << 31); //  inf
      }
    }

    static_assert(sizeof(float) == sizeof(std::uint32_t), "4-byte size check");
    float ret{};
    std::memcpy(&ret, &f, sizeof(float));
    return ret;
  }

  /// Get raw storage
  __host__ __device__ __forceinline__ uint16_t raw() const
  {
    return this->__x;
  }

  /// Equality
  __host__ __device__ __forceinline__ friend bool operator==(const half_t& a, const half_t& b)
  {
    return (a.__x == b.__x);
  }

  /// Inequality
  __host__ __device__ __forceinline__ friend bool operator!=(const half_t& a, const half_t& b)
  {
    return (a.__x != b.__x);
  }

  /// Assignment by sum
  __host__ __device__ __forceinline__ half_t& operator+=(const half_t& rhs)
  {
    *this = half_t(float(*this) + float(rhs));
    return *this;
  }

  /// Multiply
  __host__ __device__ __forceinline__ half_t operator*(const half_t& other) const
  {
    return half_t(float(*this) * float(other));
  }

  /// Divide
  __host__ __device__ __forceinline__ half_t& operator/=(const half_t& other)
  {
    return *this = half_t(float(*this) / float(other));
  }

  friend __host__ __device__ __forceinline__ half_t operator/(half_t self, const half_t& other)
  {
    return self /= other;
  }

  /// Add
  __host__ __device__ __forceinline__ half_t operator+(const half_t& other) const
  {
    return half_t(float(*this) + float(other));
  }

  /// Sub
  __host__ __device__ __forceinline__ half_t operator-(const half_t& other) const
  {
    return half_t(float(*this) - float(other));
  }

  /// Less-than
  __host__ __device__ __forceinline__ bool operator<(const half_t& other) const
  {
    return float(*this) < float(other);
  }

  /// Less-than-equal
  __host__ __device__ __forceinline__ bool operator<=(const half_t& other) const
  {
    return float(*this) <= float(other);
  }

  /// Greater-than
  __host__ __device__ __forceinline__ bool operator>(const half_t& other) const
  {
    return float(*this) > float(other);
  }

  /// Greater-than-equal
  __host__ __device__ __forceinline__ bool operator>=(const half_t& other) const
  {
    return float(*this) >= float(other);
  }

  /// numeric_traits<half_t>::max
  __host__ __device__ __forceinline__ static half_t(max)()
  {
    uint16_t max_word = 0x7BFF;
    return reinterpret_cast<half_t&>(max_word);
  }

  /// numeric_traits<half_t>::lowest
  __host__ __device__ __forceinline__ static half_t lowest()
  {
    uint16_t lowest_word = 0xFBFF;
    return reinterpret_cast<half_t&>(lowest_word);
  }
};

/******************************************************************************
 * I/O stream overloads
 ******************************************************************************/

/// Insert formatted \p half_t into the output stream
inline std::ostream& operator<<(std::ostream& out, const half_t& x)
{
  out << (float) x;
  return out;
}

/// Insert formatted \p __half into the output stream
inline std::ostream& operator<<(std::ostream& out, const __half& x)
{
  return out << half_t(x);
}

/******************************************************************************
 * Traits overloads
 ******************************************************************************/

namespace cuda
{
template <>
inline constexpr bool is_floating_point_v<half_t> = true;
}

template <>
class cuda::std::numeric_limits<half_t>
{
public:
  static constexpr bool is_specialized = true;

  static _CCCL_HOST_DEVICE _CCCL_FORCEINLINE half_t max()
  {
    return half_t(numeric_limits<__half>::max());
  }

  static _CCCL_HOST_DEVICE _CCCL_FORCEINLINE half_t min()
  {
    return half_t(numeric_limits<__half>::min());
  }

  static _CCCL_HOST_DEVICE _CCCL_FORCEINLINE half_t lowest()
  {
    return half_t(numeric_limits<__half>::lowest());
  }
};

CUB_NAMESPACE_BEGIN

template <>
struct NumericTraits<half_t> : BaseTraits<FLOATING_POINT, true, uint16_t, half_t>
{};

CUB_NAMESPACE_END

#ifdef __GNUC__
#  pragma GCC diagnostic pop
#endif
