/******************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm_>
#include <cuda/std/array>
#include <cuda/std/climits>
#include <cuda/std/cmath>

#ifndef __CUDACC__
// disable zero checks
#  define DISABLE_ZERO

// disable nan / infinity checks
#  define DISABLE_NANINF
#endif

// jump table for indexing into data
#define _CUB_RFA_MAX_JUMP_ 5
static_assert(_CUB_RFA_MAX_JUMP_ <= 5, "MAX_JUMP greater than max");

CUB_NAMESPACE_BEGIN

namespace detail
{
namespace rfa
{

template <typename Float, int LEN>
static _CCCL_DEVICE Float* get_shared_bin_array()
{
  static __shared__ Float bin_computed_array[LEN];
  return bin_computed_array;
}

// vector types with RFA are not supported currently
#if 0
template <class T>
struct get_vector_type
{};
template <>
struct get_vector_type<double>
{
  using type = double2;
};
template <>
struct get_vector_type<float>
{
  using type = float4;
};

template <class T>
using vector_t = typename get_vector_type<T>::type;

template <class T>
auto vector_size()
{
  return sizeof(vector_t<T>) / sizeof(T);
}

template <class T>
_CCCL_FORCEINLINE static double abs_max(const T&);

template <>
_CCCL_HOST_DEVICE double abs_max(const float4& x)
{
  return fmax(fmaxf(fabs(x.x), fabs(x.y)), fmax(fabs(x.z), fabs(x.w)));
}

template <>
_CCCL_HOST_DEVICE double abs_max(const double4& x)
{
  return fmax(fmax(fabs(x.x), fabs(x.y)), fmax(fabs(x.z), fabs(x.w)));
}

template <>
_CCCL_HOST_DEVICE double abs_max(const double2& x)
{
  return fmax(fabs(x.x), fabs(x.y));
}

template <>
_CCCL_HOST_DEVICE double abs_max(const float2& x)
{
  return fmax(fabs(x.x), fabs(x.y));
}
#endif

template <class ftype>
struct RFA_bins
{
  static constexpr int BIN_WIDTH = ::cuda::std::is_same_v<ftype, double> ? 40 : 13;
  static constexpr int MIN_EXP   = ::cuda::std::numeric_limits<ftype>::min_exponent;
  static constexpr int MAX_EXP   = ::cuda::std::numeric_limits<ftype>::max_exponent;
  static constexpr int MANT_DIG  = ::cuda::std::numeric_limits<ftype>::digits;

  /// Binned floating-point maximum index
  static constexpr int MAXINDEX = ((MAX_EXP - MIN_EXP + MANT_DIG - 1) / BIN_WIDTH) - 1;
  // The maximum floating-point fold supported by the library
  static constexpr int MAXFOLD = MAXINDEX + 1;

  _CCCL_HOST_DEVICE static ftype initialize_bins(int index)
  {
    if (index == 0)
    {
      if constexpr (std::is_same_v<ftype, float>)
      {
        return ::cuda::std::ldexp(0.75, MAX_EXP);
      }
      else
      {
        return 2.0 * ::cuda::std::ldexp(0.75, MAX_EXP - 1);
      }
    }

    if (index > 0 && index <= MAXINDEX)
    {
      return ::cuda::std::ldexp(0.75, MAX_EXP + MANT_DIG - BIN_WIDTH + 1 - index * BIN_WIDTH);
    }
    else
    {
      return ::cuda::std::ldexp(0.75, MAX_EXP + MANT_DIG - BIN_WIDTH + 1 - MAXINDEX * BIN_WIDTH);
    }
  }
};

static char rfa_bin_host_buffer[sizeof(RFA_bins<double>)];

//! Class to hold a reproducible summation of the numbers passed to it
//!
//!@param ftype Floating-point data type; either `float` or `double
//!@param FOLD  The fold, uses 3 by default.
template <class ftype_,
          int FOLD                                                                     = 3,
          typename ::cuda::std::enable_if_t<::cuda::std::is_floating_point_v<ftype_>>* = nullptr>
class alignas(2 * sizeof(ftype_)) ReproducibleFloatingAccumulator
{
public:
  using ftype = ftype_;

private:
  ::cuda::std::array<ftype, 2 * FOLD> data = {0};

  /// Floating-point precision bin width
  static constexpr int BIN_WIDTH = ::cuda::std::is_same_v<ftype, double> ? 40 : 13;
  static constexpr int MIN_EXP   = ::cuda::std::numeric_limits<ftype>::min_exponent;
  static constexpr int MAX_EXP   = ::cuda::std::numeric_limits<ftype>::max_exponent;
  static constexpr int MANT_DIG  = ::cuda::std::numeric_limits<ftype>::digits;

public:
  /// Binned floating-point maximum index

  static constexpr int MAXINDEX = ((MAX_EXP - MIN_EXP + MANT_DIG - 1) / BIN_WIDTH) - 1;

  // The maximum floating-point fold supported by the library
  static constexpr auto MAXFOLD = MAXINDEX + 1;

private:
  /// Binned floating-point compression factor
  /// This factor is used to scale down inputs before deposition into the bin of
  /// highest index
  static constexpr auto COMPRESSION = 1.0 / (1 << (MANT_DIG - BIN_WIDTH + 1));
  /// Binned double precision expansion factor
  /// This factor is used to scale up inputs after deposition into the bin of
  /// highest index
  static constexpr auto EXPANSION = 1.0 * (1 << (MANT_DIG - BIN_WIDTH + 1));
  static constexpr auto EXP_BIAS  = MAX_EXP - 2;
  static constexpr auto EPSILON   = ::cuda::std::numeric_limits<ftype>::epsilon();
  /// Binned floating-point deposit endurance
  /// The number of deposits that can be performed before a renorm is necessary.
  /// Applies also to binned complex double precision.
  static constexpr auto ENDURANCE = 1 << (MANT_DIG - BIN_WIDTH - 2);

  /// Return a binned floating-point bin
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE ftype binned_bins(int index) const
  {
    NV_IF_ELSE_TARGET(
      NV_IS_HOST,
      (
        // clang-format off
          return static_cast<ftype>(reinterpret_cast<ftype*>( &rfa_bin_host_buffer)[index]);
        // clang-format on
        ),
      ( // NV_IS_DEVICE:
        // clang-format off
          ftype* bins = get_shared_bin_array<ftype, MAXINDEX + MAXFOLD>();
         return static_cast<ftype>(bins[index]);
        // clang-format on
        ));
  }

  /// Get the bit representation of a float
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE static uint32_t& get_bits(float& x)
  {
    return *reinterpret_cast<uint32_t*>(&x);
  }
  /// Get the bit representation of a double
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE static uint64_t& get_bits(double& x)
  {
    return *reinterpret_cast<uint64_t*>(&x);
  }
  /// Get the bit representation of a const float
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE static uint32_t get_bits(const float& x)
  {
    return ::cuda::std::bit_cast<uint32_t>(x);
  }
  /// Get the bit representation of a const double
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE static uint64_t get_bits(const double& x)
  {
    return ::cuda::std::bit_cast<uint64_t>(x);
  }

  /// Return primary vector value const ref
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE const ftype& primary(int i) const
  {
    if (FOLD <= _CUB_RFA_MAX_JUMP_)
    {
      switch (i)
      {
        case 0:
          if (FOLD >= 1)
          {
            return data[0];
          }
          [[fallthrough]];
        case 1:
          if (FOLD >= 2)
          {
            return data[1];
          }
          [[fallthrough]];
        case 2:
          if (FOLD >= 3)
          {
            return data[2];
          }
          [[fallthrough]];
        case 3:
          if (FOLD >= 4)
          {
            return data[3];
          }
          [[fallthrough]];
        case 4:
          if (FOLD >= 5)
          {
            return data[4];
          }
          [[fallthrough]];
        default:
          return data[FOLD - 1];
      }
    }
    else
    {
      return data[i];
    }
  }

  /// Return carry vector value const ref
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE const ftype& carry(int i) const
  {
    if (FOLD <= _CUB_RFA_MAX_JUMP_)
    {
      switch (i)
      {
        case 0:
          if (FOLD >= 1)
          {
            return data[FOLD + 0];
          }
          [[fallthrough]];
        case 1:
          if (FOLD >= 2)
          {
            return data[FOLD + 1];
          }
          [[fallthrough]];
        case 2:
          if (FOLD >= 3)
          {
            return data[FOLD + 2];
          }
          [[fallthrough]];
        case 3:
          if (FOLD >= 4)
          {
            return data[FOLD + 3];
          }
          [[fallthrough]];
        case 4:
          if (FOLD >= 5)
          {
            return data[FOLD + 4];
          }
          [[fallthrough]];
        default:
          return data[2 * FOLD - 1];
      }
    }
    else
    {
      return data[FOLD + i];
    }
  }

  /// Return primary vector value ref
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE ftype& primary(int i)
  {
    const auto& c = *this;
    return const_cast<ftype&>(c.primary(i));
  }

  /// Return carry vector value ref
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE ftype& carry(int i)
  {
    const auto& c = *this;
    return const_cast<ftype&>(c.carry(i));
  }

#ifdef DISABLE_ZERO
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE static constexpr bool is_zero_v(const ftype)
  {
    return false;
  }
#else
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE static constexpr bool is_zero_v(const ftype x)
  {
    return x == 0.0;
  }
#endif

#ifdef DISABLE_NANINF
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE static constexpr int is_nan_inf_v(const ftype)
  {
    return false;
  }
#else
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE static constexpr int is_nan_inf_v(const ftype x)
  {
    const auto bits = get_bits(x);
    return (bits & ((2ull * MAX_EXP - 1) << (MANT_DIG - 1))) == ((2ull * MAX_EXP - 1) << (MANT_DIG - 1));
  }
#endif

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE static int exp_val(const ftype x)
  {
    const auto bits = get_bits(x);
    return (bits >> (MANT_DIG - 1)) & (2 * MAX_EXP - 1);
  }

  /// Get index of float-point precision
  /// The index of a non-binned type is the smallest index a binned type would
  /// need to have to sum it reproducibly. Higher indices correspond to smaller
  /// bins.
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE static int binned_dindex(const ftype x)
  {
    int exp = exp_val(x);
    if (exp == 0)
    {
      if (x == 0.0)
      {
        return MAXINDEX;
      }
      else
      {
        (void) ::cuda::std::frexpf(x, &exp);
        // use +MAXINDEX to optimize the host constexpr variable `MAXINDEX` to be used in the device function
        return (::cuda::std::min)((MAX_EXP - exp) / BIN_WIDTH, +MAXINDEX);
      }
    }
    return ((MAX_EXP + EXP_BIAS) - exp) / BIN_WIDTH;
  }

  /// Get index of manually specified binned double precision
  /// The index of a binned type is the bin that it corresponds to. Higher
  /// indices correspond to smaller bins.
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE int binned_index() const
  {
    return ((MAX_EXP + MANT_DIG - BIN_WIDTH + 1 + EXP_BIAS) - exp_val(primary(0))) / BIN_WIDTH;
  }

  /// Check if index of manually specified binned floating-point is 0
  /// A quick check to determine if the index is 0
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE bool binned_index0() const
  {
    return exp_val(primary(0)) == MAX_EXP + EXP_BIAS;
  }

  //! Update manually specified binned fp with a scalar (X -> Y)
  //!
  //! This method updates the binned fp to an index suitable for adding numbers
  //! with absolute value less than @p max_abs_val
  //!
  //!@param incpriY stride within Y's primary vector (use every incpriY'th element)
  //!@param inccarY stride within Y's carry vector (use every inccarY'th element)
  _CCCL_HOST_DEVICE void binned_dmdupdate(const ftype max_abs_val, const int incpriY, const int inccarY)
  {
    if (is_nan_inf_v(primary(0)))
    {
      return;
    }

    int X_index = binned_dindex(max_abs_val);
    if (is_zero_v(primary(0)))
    {
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 0; i < FOLD; i++)
      {
        primary(i * incpriY) = binned_bins(i + X_index);
        carry(i * inccarY)   = 0.0;
      }
    }
    else
    {
      int shift = binned_index() - X_index;
      if (shift > 0)
      {
        _CCCL_PRAGMA_UNROLL_FULL()
        for (int i = FOLD - 1; i >= 1; i--)
        {
          if (i < shift)
          {
            break;
          }
          primary(i * incpriY) = primary((i - shift) * incpriY);
          carry(i * inccarY)   = carry((i - shift) * inccarY);
        }
        _CCCL_PRAGMA_UNROLL_FULL()
        for (int j = 0; j < FOLD; j++)
        {
          if (j >= shift)
          {
            break;
          }
          primary(j * incpriY) = binned_bins(j + X_index);
          carry(j * inccarY)   = 0.0;
        }
      }
    }
  }

  //! Add scalar @p X to suitably binned manually specified binned fp (Y += X)
  //!
  //! Performs the operation Y += X on an binned type Y where the index of Y is
  //! larger than the index of @p X
  //!
  //!@param incpriY stride within Y's primary vector (use every incpriY'th element)
  _CCCL_HOST_DEVICE void binned_dmddeposit(const ftype X, const int incpriY)
  {
    ftype M;
    ftype x = X;

    if (is_nan_inf_v(x) || is_nan_inf_v(primary(0)))
    {
      primary(0) += x;
      return;
    }

    if (binned_index0())
    {
      M        = primary(0);
      ftype qd = x * COMPRESSION;
      auto& ql = get_bits(qd);
      ql |= 1;
      qd += M;
      primary(0) = qd;
      M -= qd;
      M *= EXPANSION * 0.5;
      x += M;
      x += M;
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 1; i < FOLD - 1; i++)
      {
        M  = primary(i * incpriY);
        qd = x;
        ql |= 1;
        qd += M;
        primary(i * incpriY) = qd;
        M -= qd;
        x += M;
      }
      qd = x;
      ql |= 1;
      primary((FOLD - 1) * incpriY) += qd;
    }
    else
    {
      ftype qd = x;
      auto& ql = get_bits(qd);
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 0; i < FOLD - 1; i++)
      {
        M  = primary(i * incpriY);
        qd = x;
        ql |= 1;
        qd += M;
        primary(i * incpriY) = qd;
        M -= qd;
        x += M;
      }
      qd = x;
      ql |= 1;
      primary((FOLD - 1) * incpriY) += qd;
    }
  }

  //! Renormalize manually specified binned double precision
  //!
  //! Renormalization keeps the primary vector within the necessary bins by
  //! shifting over to the carry vector
  //!
  //!@param incpriX stride within X's primary vector (use every incpriX'th element)
  //!@param inccarX stride within X's carry vector (use every inccarX'th element)
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE void binned_dmrenorm(const int incpriX, const int inccarX)
  {
    if (is_zero_v(primary(0)) || is_nan_inf_v(primary(0)))
    {
      return;
    }

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < FOLD; i++)
    {
      auto tmp_renormd  = primary(i * incpriX);
      auto& tmp_renorml = get_bits(tmp_renormd);

      carry(i * inccarX) += (int) ((tmp_renorml >> (MANT_DIG - 3)) & 3) - 2;

      tmp_renorml &= ~(1ull << (MANT_DIG - 3));
      tmp_renorml |= 1ull << (MANT_DIG - 2);
      primary(i * incpriX) = tmp_renormd;
    }
  }

  //! Add scalar to manually specified binned fp (Y += X)
  //!
  //! Performs the operation Y += X on an binned type Y
  //!
  //!@param incpriY stride within Y's primary vector (use every incpriY'th element)
  //!@param inccarY stride within Y's carry vector (use every inccarY'th element)
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE void binned_dmdadd(const ftype X, const int incpriY, const int inccarY)
  {
    binned_dmdupdate(X, 1, 1);
    binned_dmddeposit(X, 1);
    binned_dmrenorm(1, 1);
    (void) incpriY;
    (void) inccarY;
  }

  //! Convert manually specified binned fp to native double-precision (X -> Y)
  //!
  //!@param incpriX stride within X's primary vector (use every incpriX'th element)
  //!@param inccarX stride within X's carry vector (use every inccarX'th element)
  _CCCL_HOST_DEVICE double binned_conv_double(const int incpriX, const int inccarX) const
  {
    int i = 0;

    if (is_nan_inf_v(primary(0)))
    {
      return primary(0);
    }
    if (is_zero_v(primary(0)))
    {
      return 0.0;
    }

    double Y = 0.0;
    double scale_down;
    double scale_up;
    int scaled;
    const auto X_index = binned_index();
    if (X_index <= (3 * MANT_DIG) / BIN_WIDTH)
    {
      scale_down = ::cuda::std::ldexpf(0.5f, 1 - (2 * MANT_DIG - BIN_WIDTH));
      scale_up   = ::cuda::std::ldexpf(0.5f, 1 - (2 * MANT_DIG - BIN_WIDTH));
      scaled     = ::cuda::std::max(::cuda::std::min(FOLD, (3 * MANT_DIG) / BIN_WIDTH - X_index), 0);
      if (X_index == 0)
      {
        Y += carry(0) * ((binned_bins(0 + X_index) / 6.0) * scale_down * EXPANSION);
        Y += carry(inccarX) * ((binned_bins(1 + X_index) / 6.0) * scale_down);
        Y += (primary(0) - binned_bins(0 + X_index)) * scale_down * EXPANSION;
        i = 2;
      }
      else
      {
        Y += carry(0) * ((binned_bins(0 + X_index) / 6.0) * scale_down);
        i = 1;
      }
      for (; i < scaled; i++)
      {
        Y += carry(i * inccarX) * ((binned_bins(i + X_index) / 6.0) * scale_down);
        Y += (primary((i - 1) * incpriX) - binned_bins(i - 1 + X_index)) * scale_down;
      }
      if (i == FOLD)
      {
        Y += (primary((FOLD - 1) * incpriX) - binned_bins(FOLD - 1 + X_index)) * scale_down;
        return Y * scale_up;
      }
      if (::cuda::std::isinf(Y * scale_up))
      {
        return Y * scale_up;
      }
      Y *= scale_up;
      for (; i < FOLD; i++)
      {
        Y += carry(i * inccarX) * (binned_bins(i + X_index) / 6.0);
        Y += primary((i - 1) * incpriX) - binned_bins(i - 1 + X_index);
      }
      Y += primary((FOLD - 1) * incpriX) - binned_bins(FOLD - 1 + X_index);
    }
    else
    {
      Y += carry(0) * (binned_bins(0 + X_index) / 6.0);
      for (i = 1; i < FOLD; i++)
      {
        Y += carry(i * inccarX) * (binned_bins(i + X_index) / 6.0);
        Y += (primary((i - 1) * incpriX) - binned_bins(i - 1 + X_index));
      }
      Y += (primary((FOLD - 1) * incpriX) - binned_bins(FOLD - 1 + X_index));
    }
    return Y;
  }

  //! Convert manually specified binned fp to native single-precision (X -> Y)
  //!
  //!@param incpriX stride within X's primary vector (use every incpriX'th element)
  //!@param inccarX stride within X's carry vector (use every inccarX'th element)
  _CCCL_HOST_DEVICE float binned_conv_single(const int incpriX, const int inccarX) const
  {
    int i    = 0;
    double Y = 0.0;

    if (is_nan_inf_v(primary(0)))
    {
      return primary(0);
    }
    if (is_zero_v(primary(0)))
    {
      return 0.0;
    }

    // Note that the following order of summation is in order of decreasing
    // exponent. The following code is specific to SBWIDTH=13, FLT_MANT_DIG=24, and
    // the number of carries equal to 1.
    const auto X_index = binned_index();
    if (X_index == 0)
    {
      Y += (double) carry(0) * (double) (binned_bins(0 + X_index) / 6.0) * (double) EXPANSION;
      Y += (double) carry(inccarX) * (double) (binned_bins(1 + X_index) / 6.0);
      Y += (double) (primary(0) - binned_bins(0 + X_index)) * (double) EXPANSION;
      i = 2;
    }
    else
    {
      Y += (double) carry(0) * (double) (binned_bins(0 + X_index) / 6.0);
      i = 1;
    }
    for (; i < FOLD; i++)
    {
      Y += (double) carry(i * inccarX) * (double) (binned_bins(i + X_index) / 6.0);
      Y += (double) (primary((i - 1) * incpriX) - binned_bins(i - 1 + X_index));
    }
    Y += (double) (primary((FOLD - 1) * incpriX) - binned_bins(FOLD - 1 + X_index));

    return (float) Y;
  }

  //! Add two manually specified binned fp (Y += X)
  //! Performs the operation Y += X
  //!
  //!@param other   Another binned fp of the same type
  //!@param incpriX stride within X's primary vector (use every incpriX'th element)
  //!@param inccarX stride within X's carry vector (use every inccarX'th element)
  //!@param incpriY stride within Y's primary vector (use every incpriY'th element)
  //!@param inccarY stride within Y's carry vector (use every inccarY'th element)
  _CCCL_HOST_DEVICE void binned_dmdmadd(
    const ReproducibleFloatingAccumulator& x, const int incpriX, const int inccarX, const int incpriY, const int inccarY)
  {
    if (is_zero_v(x.primary(0)))
    {
      return;
    }

    if (is_zero_v(primary(0)))
    {
      for (int i = 0; i < FOLD; i++)
      {
        primary(i * incpriY) = x.primary(i * incpriX);
        carry(i * inccarY)   = x.carry(i * inccarX);
      }
      return;
    }

    if (is_nan_inf_v(x.primary(0)) || is_nan_inf_v(primary(0)))
    {
      primary(0) += x.primary(0);
      return;
    }

    const auto X_index = x.binned_index();
    const auto Y_index = this->binned_index();
    const auto shift   = Y_index - X_index;
    if (shift > 0)
    {
      // shift Y upwards and add X to Y
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = FOLD - 1; i >= 1; i--)
      {
        if (i < shift)
        {
          break;
        }
        primary(i * incpriY) =
          x.primary(i * incpriX) + (primary((i - shift) * incpriY) - binned_bins(i - shift + Y_index));
        carry(i * inccarY) = x.carry(i * inccarX) + carry((i - shift) * inccarY);
      }
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 0; i < FOLD; i++)
      {
        if (i == shift)
        {
          break;
        }
        primary(i * incpriY) = x.primary(i * incpriX);
        carry(i * inccarY)   = x.carry(i * inccarX);
      }
    }
    else if (shift < 0)
    {
      // shift X upwards and add X to Y
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 0; i < FOLD; i++)
      {
        if (i < -shift)
        {
          continue;
        }
        primary(i * incpriY) += x.primary((i + shift) * incpriX) - binned_bins(X_index + i + shift);
        carry(i * inccarY) += x.carry((i + shift) * inccarX);
      }
    }
    else if (shift == 0)
    {
      // add X to Y
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 0; i < FOLD; i++)
      {
        primary(i * incpriY) += x.primary(i * incpriX) - binned_bins(i + X_index);
        carry(i * inccarY) += x.carry(i * inccarX);
      }
    }

    binned_dmrenorm(incpriY, inccarY);
  }

  /// Add two manually specified binned fp (Y += X)
  /// Performs the operation Y += X
  _CCCL_HOST_DEVICE void binned_dbdbadd(const ReproducibleFloatingAccumulator& other)
  {
    binned_dmdmadd(other, 1, 1, 1, 1);
  }

public:
  ReproducibleFloatingAccumulator()                                       = default;
  ReproducibleFloatingAccumulator(const ReproducibleFloatingAccumulator&) = default;
  /// Sets this binned fp equal to another binned fp
  ReproducibleFloatingAccumulator& operator=(const ReproducibleFloatingAccumulator&) = default;

  /// Set the binned fp to zero
  _CCCL_HOST_DEVICE void zero()
  {
    data = {0};
  }

  /// Return the fold of the binned fp
  constexpr int fold() const
  {
    return FOLD;
  }

  /// Return the endurance of the binned fp
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE constexpr int endurance() const
  {
    return ENDURANCE;
  }

  /// Returns the number of reference bins. Used for judging memory usage.
  _CCCL_HOST _CCCL_FORCEINLINE static constexpr auto number_of_reference_bins()
  {
    return ::cuda::std::array<ftype, MAXINDEX + MAXFOLD>::size();
  }

  //! Accumulate an arithmetic @p x into the binned fp.
  //! NOTE: Casts @p x to the type of the binned fp
  template <typename U, typename ::cuda::std::enable_if_t<::cuda::std::is_arithmetic_v<U>>* = nullptr>
  _CCCL_HOST_DEVICE ReproducibleFloatingAccumulator& operator+=(const U x)
  {
    binned_dmdadd(static_cast<ftype>(x), 1, 1);
    return *this;
  }

  //! Accumulate-subtract an arithmetic @p x into the binned fp.
  //! NOTE: Casts @p x to the type of the binned fp
  template <typename U, typename ::cuda::std::enable_if_t<::cuda::std::is_arithmetic_v<U>>* = nullptr>
  _CCCL_HOST_DEVICE ReproducibleFloatingAccumulator& operator-=(const U x)
  {
    binned_dmdadd(-static_cast<ftype>(x), 1, 1);
    return *this;
  }

  /// Accumulate a binned fp @p x into the binned fp.
  _CCCL_HOST_DEVICE ReproducibleFloatingAccumulator& operator+=(const ReproducibleFloatingAccumulator& other)
  {
    binned_dbdbadd(other);
    return *this;
  }

  //! Accumulate-subtract a binned fp @p x into the binned fp.
  //! NOTE: Makes a copy and performs arithmetic; slow.
  _CCCL_HOST_DEVICE ReproducibleFloatingAccumulator& operator-=(const ReproducibleFloatingAccumulator& other)
  {
    const auto temp = -other;
    binned_dbdbadd(temp);
  }

  /// Determines if two binned fp are equal
  _CCCL_HOST_DEVICE bool operator==(const ReproducibleFloatingAccumulator& other) const
  {
    return data == other.data;
  }

  /// Determines if two binned fp are not equal
  _CCCL_HOST_DEVICE bool operator!=(const ReproducibleFloatingAccumulator& other) const
  {
    return !operator==(other);
  }

  //! Sets this binned fp equal to the arithmetic value @p x
  //! NOTE: Casts @p x to the type of the binned fp
  template <typename U, typename ::cuda::std::enable_if_t<::cuda::std::is_arithmetic_v<U>>* = nullptr>
  _CCCL_HOST_DEVICE ReproducibleFloatingAccumulator& operator=(const U x)
  {
    zero();
    binned_dmdadd(static_cast<ftype>(x), 1, 1);
    return *this;
  }

  //! Returns the negative of this binned fp
  //! NOTE: Makes a copy and performs arithmetic; slow.
  _CCCL_HOST_DEVICE ReproducibleFloatingAccumulator operator-()
  {
    constexpr int incpriX                = 1;
    constexpr int inccarX                = 1;
    ReproducibleFloatingAccumulator temp = *this;
    if (primary(0) != 0.0)
    {
      // constexpr auto const bins = binned_bins(binned_index());

      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 0; i < FOLD; i++)
      {
        temp.primary(i * incpriX) =
          binned_bins(i + binned_index()) - (primary(i * incpriX) - binned_bins(i + binned_index()));
        temp.carry(i * inccarX) = -carry(i * inccarX);
      }
    }
    return temp;
  }

  /// Convert this binned fp into its native floating-point representation
  _CCCL_HOST_DEVICE ftype conv() const
  {
    if (::cuda::std::is_same_v<ftype, float>)
    {
      return binned_conv_single(1, 1);
    }
    else
    {
      return binned_conv_double(1, 1);
    }
  }

  /// Add @p x to the binned fp
  _CCCL_HOST_DEVICE void add(const ftype x)
  {
    binned_dmdadd(x, 1, 1);
  }

  //////////////////////////////////////
  // MANUAL OPERATIONS; USE WISELY
  //////////////////////////////////////

  //! Rebins for repeated accumulation of scalars with magnitude <= @p mav
  //!
  //! Once rebinned, `ENDURANCE` values <= @p mav can be added to the accumulator
  //! with `unsafe_add` after which `renorm()` must be called. See the source of
  //!`add()` for an example
  _CCCL_HOST_DEVICE void set_max_val(const ftype mav)
  {
    binned_dmdupdate(static_cast<ftype>(mav), 1, 1);
  }

  //! Add @p x to the binned fp
  //!
  //! This is intended to be used after a call to `set_max_abs_val()`
  _CCCL_HOST_DEVICE void unsafe_add(const ftype x)
  {
    binned_dmddeposit(x, 1);
  }

  //! Renormalizes the binned fp
  //!
  //! This is intended to be used after a call to `set_max_abs_val()` and one or
  //! more calls to `unsafe_add()`
  _CCCL_HOST_DEVICE void renorm()
  {
    binned_dmrenorm(1, 1);
  }
};

template <typename FloatType = float, typename ::cuda::std::enable_if_t<std::is_floating_point_v<FloatType>>* = nullptr>
struct rfa_float_transform_t
{
  _CCCL_HOST_DEVICE FloatType operator()(FloatType accum)
  {
    return accum;
  }

  _CCCL_HOST_DEVICE FloatType operator()(ReproducibleFloatingAccumulator<FloatType> accum)
  {
    return accum.conv();
  }
};
} // namespace rfa

} // namespace detail

CUB_NAMESPACE_END
