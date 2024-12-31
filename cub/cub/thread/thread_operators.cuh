/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2024, NVIDIA CORPORATION.  All rights reserved.
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

/**
 * @file
 * Simple binary operator functor types
 */

/******************************************************************************
 * Simple functor operators
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

#include <cub/detail/type_traits.cuh> // always_false
#include <cub/util_type.cuh>

#include <cuda/functional> // cuda::maximum, cuda::minimum
#include <cuda/std/bit> // cuda::std::bit_cast
#include <cuda/std/functional> // cuda::std::plus
#include <cuda/std/type_traits> // cuda::std::common_type
#include <cuda/std/utility> // cuda::std::forward

#if defined(_CCCL_HAS_NVFP16)
#  include <cuda_fp16.h>
#endif // _CCCL_HAS_NVFP16

#if defined(_CCCL_HAS_NVBF16)
_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_CLANG("-Wunused-function")
#  include <cuda_bf16.h>
_CCCL_DIAG_POP
#endif // _CCCL_HAS_NVFP16

CUB_NAMESPACE_BEGIN

// TODO(bgruber): deprecate in C++17 with a note: "replace by decltype(cuda::std::not_fn(EqualityOp{}))"
/// @brief Inequality functor (wraps equality functor)
template <typename EqualityOp>
struct InequalityWrapper
{
  /// Wrapped equality operator
  EqualityOp op;

  /// Constructor
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE InequalityWrapper(EqualityOp op)
      : op(op)
  {}

  /// Boolean inequality operator, returns `t != u`
  template <typename T, typename U>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE bool operator()(T&& t, U&& u)
  {
    return !op(::cuda::std::forward<T>(t), ::cuda::std::forward<U>(u));
  }
};

using Equality CCCL_DEPRECATED_BECAUSE("use cuda::std::equal_to instead")       = ::cuda::std::equal_to<>;
using Inequality CCCL_DEPRECATED_BECAUSE("use cuda::std::not_equal_to instead") = ::cuda::std::not_equal_to<>;
using Sum CCCL_DEPRECATED_BECAUSE("use cuda::std::plus instead")                = ::cuda::std::plus<>;
using Difference CCCL_DEPRECATED_BECAUSE("use cuda::std::minus instead")        = ::cuda::std::minus<>;
using Division CCCL_DEPRECATED_BECAUSE("use cuda::std::divides instead")        = ::cuda::std::divides<>;
using Max CCCL_DEPRECATED_BECAUSE("use cuda::maximum instead")                  = ::cuda::maximum<>;
using Min CCCL_DEPRECATED_BECAUSE("use cuda::minimum instead")                  = ::cuda::minimum<>;

/// @brief Arg max functor (keeps the value and offset of the first occurrence
///        of the larger item)
struct ArgMax
{
  /// Boolean max operator, preferring the item having the smaller offset in
  /// case of ties
  template <typename T, typename OffsetT>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE KeyValuePair<OffsetT, T>
  operator()(const KeyValuePair<OffsetT, T>& a, const KeyValuePair<OffsetT, T>& b) const
  {
    // Mooch BUG (device reduce argmax gk110 3.2 million random fp32)
    // return ((b.value > a.value) ||
    //         ((a.value == b.value) && (b.key < a.key)))
    //      ? b : a;

    if ((b.value > a.value) || ((a.value == b.value) && (b.key < a.key)))
    {
      return b;
    }

    return a;
  }
};

/// @brief Arg min functor (keeps the value and offset of the first occurrence
///        of the smallest item)
struct ArgMin
{
  /// Boolean min operator, preferring the item having the smaller offset in
  /// case of ties
  template <typename T, typename OffsetT>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE KeyValuePair<OffsetT, T>
  operator()(const KeyValuePair<OffsetT, T>& a, const KeyValuePair<OffsetT, T>& b) const
  {
    // Mooch BUG (device reduce argmax gk110 3.2 million random fp32)
    // return ((b.value < a.value) ||
    //         ((a.value == b.value) && (b.key < a.key)))
    //      ? b : a;

    if ((b.value < a.value) || ((a.value == b.value) && (b.key < a.key)))
    {
      return b;
    }

    return a;
  }
};

namespace detail
{
template <typename ScanOpT>
struct ScanBySegmentOp
{
  /// Wrapped operator
  ScanOpT op;

  /// Constructor
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE ScanBySegmentOp() {}

  /// Constructor
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE ScanBySegmentOp(ScanOpT op)
      : op(op)
  {}

  /**
   * @brief Scan operator
   *
   * @tparam KeyValuePairT
   *   KeyValuePair pairing of T (value) and int (head flag)
   *
   * @param[in] first
   *   First partial reduction
   *
   * @param[in] second
   *   Second partial reduction
   */
  template <typename KeyValuePairT>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE KeyValuePairT operator()(const KeyValuePairT& first, const KeyValuePairT& second)
  {
    KeyValuePairT retval;
    retval.key = first.key | second.key;
#ifdef _NVHPC_CUDA // WAR bug on nvc++
    if (second.key)
    {
      retval.value = second.value;
    }
    else
    {
      // If second.value isn't copied into a temporary here, nvc++ will
      // crash while compiling the TestScanByKeyWithLargeTypes test in
      // thrust/testing/scan_by_key.cu:
      auto v2      = second.value;
      retval.value = op(first.value, v2);
    }
#else // not nvc++:
    // if (second.key) {
    //   The second partial reduction spans a segment reset, so it's value
    //   aggregate becomes the running aggregate
    // else {
    //   The second partial reduction does not span a reset, so accumulate both
    //   into the running aggregate
    // }
    retval.value = (second.key) ? second.value : op(first.value, second.value);
#endif
    return retval;
  }
};

template <class OpT>
struct basic_binary_op_t
{
  static constexpr bool value = false;
};

template <>
struct basic_binary_op_t<Sum>
{
  static constexpr bool value = true;
};

template <>
struct basic_binary_op_t<Min>
{
  static constexpr bool value = true;
};

template <>
struct basic_binary_op_t<Max>
{
  static constexpr bool value = true;
};
} // namespace detail

/// @brief Default cast functor
template <typename B>
struct CastOp
{
  /// Cast operator, returns `(B) a`
  template <typename A>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE B operator()(A&& a) const
  {
    return (B) a;
  }
};

/// @brief Binary operator wrapper for switching non-commutative scan arguments
template <typename ScanOp>
class SwizzleScanOp
{
private:
  /// Wrapped scan operator
  ScanOp scan_op;

public:
  /// Constructor
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE SwizzleScanOp(ScanOp scan_op)
      : scan_op(scan_op)
  {}

  /// Switch the scan arguments
  template <typename T>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE T operator()(const T& a, const T& b)
  {
    T _a(a);
    T _b(b);

    return scan_op(_b, _a);
  }
};

/**
 * @brief Reduce-by-segment functor.
 *
 * Given two cub::KeyValuePair inputs `a` and `b` and a binary associative
 * combining operator `f(const T &x, const T &y)`, an instance of this functor
 * returns a cub::KeyValuePair whose `key` field is `a.key + b.key`, and whose
 * `value` field is either `b.value` if `b.key` is non-zero, or
 * `f(a.value, b.value)` otherwise.
 *
 * ReduceBySegmentOp is an associative, non-commutative binary combining
 * operator for input sequences of cub::KeyValuePair pairings. Such sequences
 * are typically used to represent a segmented set of values to be reduced
 * and a corresponding set of {0,1}-valued integer "head flags" demarcating the
 * first value of each segment.
 *
 * @tparam ReductionOpT Binary reduction operator to apply to values
 */
template <typename ReductionOpT>
struct ReduceBySegmentOp
{
  /// Wrapped reduction operator
  ReductionOpT op;

  /// Constructor
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE ReduceBySegmentOp() {}

  /// Constructor
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE ReduceBySegmentOp(ReductionOpT op)
      : op(op)
  {}

  /**
   * @brief Scan operator
   *
   * @tparam KeyValuePairT
   *   KeyValuePair pairing of T (value) and OffsetT (head flag)
   *
   * @param[in] first
   *   First partial reduction
   *
   * @param[in] second
   *   Second partial reduction
   */
  template <typename KeyValuePairT>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE KeyValuePairT operator()(const KeyValuePairT& first, const KeyValuePairT& second)
  {
    KeyValuePairT retval;
    retval.key = first.key + second.key;
#ifdef _NVHPC_CUDA // WAR bug on nvc++
    if (second.key)
    {
      retval.value = second.value;
    }
    else
    {
      // If second.value isn't copied into a temporary here, nvc++ will
      // crash while compiling the TestScanByKeyWithLargeTypes test in
      // thrust/testing/scan_by_key.cu:
      auto v2      = second.value;
      retval.value = op(first.value, v2);
    }
#else // not nvc++:
    // if (second.key) {
    //   The second partial reduction spans a segment reset, so it's value
    //   aggregate becomes the running aggregate
    // else {
    //   The second partial reduction does not span a reset, so accumulate both
    //   into the running aggregate
    // }
    retval.value = (second.key) ? second.value : op(first.value, second.value);
#endif
    return retval;
  }
};

/**
 * @tparam ReductionOpT Binary reduction operator to apply to values
 */
template <typename ReductionOpT>
struct ReduceByKeyOp
{
  /// Wrapped reduction operator
  ReductionOpT op;

  /// Constructor
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE ReduceByKeyOp() {}

  /// Constructor
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE ReduceByKeyOp(ReductionOpT op)
      : op(op)
  {}

  /**
   * @brief Scan operator
   *
   * @param[in] first First partial reduction
   * @param[in] second Second partial reduction
   */
  template <typename KeyValuePairT>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE KeyValuePairT operator()(const KeyValuePairT& first, const KeyValuePairT& second)
  {
    KeyValuePairT retval = second;

    if (first.key == second.key)
    {
      retval.value = op(first.value, retval.value);
    }

    return retval;
  }
};

//! Deprecated [Since 2.8]
template <typename BinaryOpT>
struct CCCL_DEPRECATED BinaryFlip
{
  BinaryOpT binary_op;

  _CCCL_HOST_DEVICE explicit BinaryFlip(BinaryOpT binary_op)
      : binary_op(binary_op)
  {}

  template <typename T, typename U>
  _CCCL_DEVICE auto
  operator()(T&& t, U&& u) -> decltype(binary_op(::cuda::std::forward<U>(u), ::cuda::std::forward<T>(t)))
  {
    return binary_op(::cuda::std::forward<U>(u), ::cuda::std::forward<T>(t));
  }
};

_CCCL_SUPPRESS_DEPRECATED_PUSH
//! Deprecated [Since 2.8]
template <typename BinaryOpT>
CCCL_DEPRECATED _CCCL_HOST_DEVICE BinaryFlip<BinaryOpT> MakeBinaryFlip(BinaryOpT binary_op)
{
  return BinaryFlip<BinaryOpT>(binary_op);
}
_CCCL_SUPPRESS_DEPRECATED_POP

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document

namespace internal
{

template <typename T>
struct SimdMin
{
  static_assert(cub::detail::always_false<T>(), "Unsupported specialization");
};

template <>
struct SimdMin<::cuda::std::int16_t>
{
  using simd_type = ::cuda::std::uint32_t;

  _CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE ::cuda::std::uint32_t
  operator()(::cuda::std::uint32_t a, ::cuda::std::uint32_t b) const
  {
    return __vmins2(a, b);
  }
};

template <>
struct SimdMin<::cuda::std::uint16_t>
{
  using simd_type = ::cuda::std::uint32_t;

  _CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE ::cuda::std::uint32_t
  operator()(::cuda::std::uint32_t a, ::cuda::std::uint32_t b) const
  {
    return __vminu2(a, b);
  }
};

#  if defined(_CCCL_HAS_NVFP16)

template <>
struct SimdMin<__half>
{
  using simd_type = __half2;

  _CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE __half2 operator()(__half2 a, __half2 b) const
  {
#    if _CCCL_CUDACC_BELOW(12) && _CCCL_CUDA_COMPILER(NVHPC)
    return __floats2half2_rn(::cuda::minimum<>{}(__half2float(a.x), __half2float(b.x)),
                             ::cuda::minimum<>{}(__half2float(a.y), __half2float(b.y)));
#    else // ^^^ _CCCL_CUDACC_BELOW(12) && _CCCL_CUDA_COMPILER(NVHPC) ^^^ / vvv otherwise vvv
    NV_IF_TARGET(NV_PROVIDES_SM_80,
                 (return __hmin2(a, b);),
                 (return __halves2half2(__float2half(::cuda::minimum<>{}(__half2float(a.x), __half2float(b.x))),
                                        __float2half(::cuda::minimum<>{}(__half2float(a.y), __half2float(b.y))));));
#    endif // !_CCCL_CUDACC_BELOW(12) || !_CCCL_CUDA_COMPILER(NVHPC)
  }
};

#  endif // defined(_CCCL_HAS_NVFP16)

#  if defined(_CCCL_HAS_NVBF16)

// NOTE: __halves2bfloat162 is not always available on older CUDA Toolkits for __CUDA_ARCH__ < 800
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE __nv_bfloat162 halves2bfloat162(__nv_bfloat16 a, __nv_bfloat16 b)
{
  ::cuda::std::uint32_t tmp;
  auto a_uint16 = ::cuda::std::bit_cast<::cuda::std::uint16_t>(a);
  auto b_uint16 = ::cuda::std::bit_cast<::cuda::std::uint16_t>(b);
  asm("{mov.b32 %0, {%1,%2};}\n" : "=r"(tmp) : "h"(a_uint16), "h"(b_uint16));
  __nv_bfloat162 ret;
  ::memcpy(&ret, &tmp, sizeof(ret));
  return ret; // TODO: replace with ::cuda::std::bit_cast<__nv_bfloat162>(tmp);
}

template <>
struct SimdMin<__nv_bfloat16>
{
  using simd_type = __nv_bfloat162;

  _CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE __nv_bfloat162 operator()(__nv_bfloat162 a, __nv_bfloat162 b) const
  {
#    if _CCCL_CUDACC_BELOW(12) && _CCCL_CUDA_COMPILER(NVHPC)
    return __floats2bfloat162_rn(::cuda::minimum<>{}(__bfloat162float(a.x), __bfloat162float(b.x)),
                                 ::cuda::minimum<>{}(__bfloat162float(a.y), __bfloat162float(b.y)));
#    else // ^^^ _CCCL_CUDACC_BELOW(12) && _CCCL_CUDA_COMPILER(NVHPC) ^^^ / vvv otherwise vvv
    NV_IF_TARGET(NV_PROVIDES_SM_80,
                 (return __hmin2(a, b);),
                 (return cub::internal::halves2bfloat162(
                           __float2bfloat16(::cuda::minimum<>{}(__bfloat162float(a.x), __bfloat162float(b.x))),
                           __float2bfloat16(::cuda::minimum<>{}(__bfloat162float(a.y), __bfloat162float(b.y))));));
#    endif // !_CCCL_CUDACC_BELOW(12) || !_CCCL_CUDA_COMPILER(NVHPC)
  }
};

#  endif // defined(_CCCL_HAS_NVBF16)

//----------------------------------------------------------------------------------------------------------------------

template <typename T>
struct SimdMax
{
  static_assert(cub::detail::always_false<T>(), "Unsupported specialization");
};

template <>
struct SimdMax<::cuda::std::int16_t>
{
  using simd_type = ::cuda::std::uint32_t;

  _CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE ::cuda::std::uint32_t
  operator()(::cuda::std::uint32_t a, ::cuda::std::uint32_t b) const
  {
    return __vmaxs2(a, b);
  }
};

template <>
struct SimdMax<::cuda::std::uint16_t>
{
  using simd_type = ::cuda::std::uint32_t;

  _CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE ::cuda::std::uint32_t
  operator()(::cuda::std::uint32_t a, ::cuda::std::uint32_t b) const
  {
    return __vmaxu2(a, b);
  }
};

#  if defined(_CCCL_HAS_NVFP16)

template <>
struct SimdMax<__half>
{
  using simd_type = __half2;

  _CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE __half2 operator()(__half2 a, __half2 b) const
  {
#    if _CCCL_CUDACC_BELOW(12) && _CCCL_CUDA_COMPILER(NVHPC)
    return __floats2half2_rn(::cuda::maximum<>{}(__half2float(a.x), __half2float(b.x)),
                             ::cuda::maximum<>{}(__half2float(a.y), __half2float(b.y)));
#    else // ^^^ _CCCL_CUDACC_BELOW(12) && _CCCL_CUDA_COMPILER(NVHPC) ^^^ / vvv otherwise vvv
    NV_IF_TARGET(NV_PROVIDES_SM_80,
                 (return __hmax2(a, b);),
                 (return __halves2half2(__float2half(::cuda::maximum<>{}(__half2float(a.x), __half2float(b.x))),
                                        __float2half(::cuda::maximum<>{}(__half2float(a.y), __half2float(b.y))));));
#    endif // !_CCCL_CUDACC_BELOW(12) || !_CCCL_CUDA_COMPILER(NVHPC)
  }
};

#  endif // defined(_CCCL_HAS_NVFP16)

#  if defined(_CCCL_HAS_NVBF16)

template <>
struct SimdMax<__nv_bfloat16>
{
  using simd_type = __nv_bfloat162;

  _CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE __nv_bfloat162 operator()(__nv_bfloat162 a, __nv_bfloat162 b) const
  {
#    if _CCCL_CUDACC_BELOW(12) && _CCCL_CUDA_COMPILER(NVHPC)
    return __floats2bfloat162_rn(::cuda::maximum<>{}(__bfloat162float(a.x), __bfloat162float(b.x)),
                                 ::cuda::maximum<>{}(__bfloat162float(a.y), __bfloat162float(b.y)));
#    else // ^^^ _CCCL_CUDACC_BELOW(12) && _CCCL_CUDA_COMPILER(NVHPC) ^^^ / vvv otherwise vvv
    NV_IF_TARGET(NV_PROVIDES_SM_80,
                 (return __hmax2(a, b);),
                 (return cub::internal::halves2bfloat162(
                           __float2bfloat16(::cuda::maximum<>{}(__bfloat162float(a.x), __bfloat162float(b.x))),
                           __float2bfloat16(::cuda::maximum<>{}(__bfloat162float(a.y), __bfloat162float(b.y))));));
#    endif // !_CCCL_CUDACC_BELOW(12) || !_CCCL_CUDA_COMPILER(NVHPC)
  }
};

#  endif // defined(_CCCL_HAS_NVBF16)

//----------------------------------------------------------------------------------------------------------------------

template <typename T>
struct SimdSum
{
  static_assert(cub::detail::always_false<T>(), "Unsupported specialization");
};

#  if defined(_CCCL_HAS_NVFP16)

template <>
struct SimdSum<__half>
{
  using simd_type = __half2;

  _CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE __half2 operator()(__half2 a, __half2 b) const
  {
#    if _CCCL_CUDACC_BELOW(12) && _CCCL_CUDA_COMPILER(NVHPC)
    return __floats2half2_rn(__half2float(a.x) + __half2float(b.x), __half2float(a.y) + __half2float(b.y));
#    else // ^^^ _CCCL_CUDACC_BELOW(12) && _CCCL_CUDA_COMPILER(NVHPC) ^^^ / vvv otherwise vvv
    NV_IF_TARGET(NV_PROVIDES_SM_53,
                 (return __hadd2(a, b);),
                 (return __halves2half2(__float2half(__half2float(a.x) + __half2float(b.x)),
                                        __float2half(__half2float(a.y) + __half2float(b.y)));));
#    endif // !_CCCL_CUDACC_BELOW(12) || !_CCCL_CUDA_COMPILER(NVHPC)
  }
};

#  endif // defined(_CCCL_HAS_NVFP16)

#  if defined(_CCCL_HAS_NVBF16)

template <>
struct SimdSum<__nv_bfloat16>
{
  using simd_type = __nv_bfloat162;

  _CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE __nv_bfloat162 operator()(__nv_bfloat162 a, __nv_bfloat162 b) const
  {
#    if _CCCL_CUDACC_BELOW(12) && _CCCL_CUDA_COMPILER(NVHPC)
    return __floats2bfloat162_rn(
      __bfloat162float(a.x) + __bfloat162float(b.x), __bfloat162float(a.y) + __bfloat162float(b.y));
#    else // ^^^ _CCCL_CUDACC_BELOW(12) && _CCCL_CUDA_COMPILER(NVHPC) ^^^ / vvv otherwise vvv
    NV_IF_TARGET(
      NV_PROVIDES_SM_80,
      (return __hadd2(a, b);),
      (return cub::internal::halves2bfloat162(__float2bfloat16(__bfloat162float(a.x) + __bfloat162float(b.x)),
                                              __float2bfloat16(__bfloat162float(a.y) + __bfloat162float(b.y)));));
#    endif // !_CCCL_CUDACC_BELOW(12) || !_CCCL_CUDA_COMPILER(NVHPC)
  }
};

#  endif // defined(_CCCL_HAS_NVBF16)

//----------------------------------------------------------------------------------------------------------------------

template <typename T>
struct SimdMul
{
  static_assert(cub::detail::always_false<T>(), "Unsupported specialization");
};

#  if defined(_CCCL_HAS_NVFP16)

template <>
struct SimdMul<__half>
{
  using simd_type = __half2;

  _CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE __half2 operator()(__half2 a, __half2 b) const
  {
#    if _CCCL_CUDACC_BELOW(12) && _CCCL_CUDA_COMPILER(NVHPC)
    return __floats2half2_rn(__half2float(a.x) * __half2float(b.x), __half2float(a.y) * __half2float(b.y));
#    else // ^^^ _CCCL_CUDACC_BELOW(12) && _CCCL_CUDA_COMPILER(NVHPC) ^^^ / vvv otherwise vvv
    NV_IF_TARGET(NV_PROVIDES_SM_53,
                 (return __hmul2(a, b);),
                 (return __halves2half2(__float2half(__half2float(a.x) * __half2float(b.x)),
                                        __float2half(__half2float(a.y) * __half2float(b.y)));));
#    endif // !_CCCL_CUDACC_BELOW(12) || !_CCCL_CUDA_COMPILER(NVHPC)
  }
};

#  endif // defined(_CCCL_HAS_NVFP16)

#  if defined(_CCCL_HAS_NVBF16)

template <>
struct SimdMul<__nv_bfloat16>
{
  using simd_type = __nv_bfloat162;

  _CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE __nv_bfloat162 operator()(__nv_bfloat162 a, __nv_bfloat162 b) const
  {
#    if _CCCL_CUDACC_BELOW(12) && _CCCL_CUDA_COMPILER(NVHPC)
    return __floats2bfloat162_rn(
      __bfloat162float(a.x) * __bfloat162float(b.x), __bfloat162float(a.y) * __bfloat162float(b.y));
#    else // ^^^ _CCCL_CUDACC_BELOW(12) && _CCCL_CUDA_COMPILER(NVHPC) ^^^ / vvv otherwise vvv
    NV_IF_TARGET(NV_PROVIDES_SM_80,
                 (return __hmul2(a, b);),
                 (return halves2bfloat162(__float2bfloat16(__bfloat162float(a.x) * __bfloat162float(b.x)),
                                          __float2bfloat16(__bfloat162float(a.y) * __bfloat162float(b.y)));));
#    endif // !_CCCL_CUDACC_BELOW(12) || !_CCCL_CUDA_COMPILER(NVHPC)
  }
};

#  endif // defined(_CCCL_HAS_NVBF16)

//----------------------------------------------------------------------------------------------------------------------

template <typename ReduceOp, typename T>
struct CubOperatorToSimdOperator
{
  static_assert(cub::detail::always_false<T>(), "Unsupported specialization");
};

template <typename T>
struct CubOperatorToSimdOperator<::cuda::minimum<>, T>
{
  using type      = SimdMin<T>;
  using simd_type = typename type::simd_type;
};

template <typename T>
struct CubOperatorToSimdOperator<::cuda::minimum<T>, T> : CubOperatorToSimdOperator<::cuda::minimum<>, T>
{};

template <typename T>
struct CubOperatorToSimdOperator<::cuda::maximum<>, T>
{
  using type      = SimdMax<T>;
  using simd_type = typename type::simd_type;
};

template <typename T>
struct CubOperatorToSimdOperator<::cuda::maximum<T>, T> : CubOperatorToSimdOperator<::cuda::maximum<>, T>
{};

template <typename T>
struct CubOperatorToSimdOperator<::cuda::std::plus<>, T>
{
  using type      = SimdSum<T>;
  using simd_type = typename type::simd_type;
};

template <typename T>
struct CubOperatorToSimdOperator<::cuda::std::plus<T>, T> : CubOperatorToSimdOperator<::cuda::std::plus<>, T>
{};

template <typename T>
struct CubOperatorToSimdOperator<::cuda::std::multiplies<>, T>
{
  using type      = SimdMul<T>;
  using simd_type = typename type::simd_type;
};

template <typename T>
struct CubOperatorToSimdOperator<::cuda::std::multiplies<T>, T>
    : CubOperatorToSimdOperator<::cuda::std::multiplies<>, T>
{};

template <typename ReduceOp, typename T>
using cub_operator_to_simd_operator_t = typename CubOperatorToSimdOperator<ReduceOp, T>::type;

template <typename ReduceOp, typename T>
using simd_type_t = typename CubOperatorToSimdOperator<ReduceOp, T>::simd_type;

} // namespace internal

#endif // !_CCCL_DOXYGEN_INVOKED

CUB_NAMESPACE_END
