/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
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
 * Thread utilities for sequential prefix scan over statically-sized array types
 */

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/thread/thread_operators.cuh>

CUB_NAMESPACE_BEGIN

/// Internal namespace (to prevent ADL mishaps between static functions when mixing different CUB installations)
namespace detail
{

/**
 * @name Sequential prefix scan over statically-sized array types
 * @{
 */

/**
 * @param[in] input
 *   Input array
 *
 * @param[out] output
 *   Output array (may be aliased to @p input)
 *
 * @param[in] scan_op
 *   Binary scan operator
 */
template <int LENGTH, typename T, typename ScanOp>
_CCCL_DEVICE _CCCL_FORCEINLINE T ThreadScanExclusive(
  T inclusive, T exclusive, T* input, T* output, ScanOp scan_op, detail::constant_t<LENGTH> /*length*/)
{
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < LENGTH; ++i)
  {
    inclusive = scan_op(exclusive, input[i]);
    output[i] = exclusive;
    exclusive = inclusive;
  }

  return inclusive;
}

/**
 * @brief Perform a sequential exclusive prefix scan over @p LENGTH elements of
 *        the @p input array, seeded with the specified @p prefix. The aggregate is returned.
 *
 * @tparam LENGTH
 *   LengthT of @p input and @p output arrays
 *
 * @tparam T
 *   <b>[inferred]</b> The data type to be scanned.
 *
 * @tparam ScanOp
 *   <b>[inferred]</b> Binary scan operator type having member
 *   <tt>T operator()(const T &a, const T &b)</tt>
 *
 * @param[in] input
 *   Input array
 *
 * @param[out] output
 *   Output array (may be aliased to @p input)
 *
 * @param[in] scan_op
 *   Binary scan operator
 *
 * @param[in] prefix
 *   Prefix to seed scan with
 *
 * @param[in] apply_prefix
 *   Whether or not the calling thread should apply its prefix.
 *   If not, the first output element is undefined.
 *   (Handy for preventing thread-0 from applying a prefix.)
 */
template <int LENGTH, typename T, typename ScanOp>
_CCCL_DEVICE _CCCL_FORCEINLINE T
ThreadScanExclusive(T* input, T* output, ScanOp scan_op, T prefix, bool apply_prefix = true)
{
  T inclusive = input[0];
  if (apply_prefix)
  {
    inclusive = scan_op(prefix, inclusive);
  }
  output[0]   = prefix;
  T exclusive = inclusive;

  return ThreadScanExclusive(inclusive, exclusive, input + 1, output + 1, scan_op, detail::constant_v<LENGTH - 1>);
}

/**
 * @brief Perform a sequential exclusive prefix scan over the statically-sized
 *        @p input array, seeded with the specified @p prefix. The aggregate is returned.
 *
 * @tparam LENGTH
 *   <b>[inferred]</b> LengthT of @p input and @p output arrays
 *
 * @tparam T
 *   <b>[inferred]</b> The data type to be scanned.
 *
 * @tparam ScanOp
 *   <b>[inferred]</b> Binary scan operator type having member
 *   <tt>T operator()(const T &a, const T &b)</tt>
 *
 * @param[in] input
 *   Input array
 *
 * @param[out] output
 *   Output array (may be aliased to @p input)
 *
 * @param[in] scan_op
 *   Binary scan operator
 *
 * @param[in] prefix
 *   Prefix to seed scan with
 *
 * @param[in] apply_prefix
 *   Whether or not the calling thread should apply its prefix.
 *   (Handy for preventing thread-0 from applying a prefix.)
 */
template <int LENGTH, typename T, typename ScanOp>
_CCCL_DEVICE _CCCL_FORCEINLINE T
ThreadScanExclusive(T (&input)[LENGTH], T (&output)[LENGTH], ScanOp scan_op, T prefix, bool apply_prefix = true)
{
  return ThreadScanExclusive<LENGTH>((T*) input, (T*) output, scan_op, prefix, apply_prefix);
}

/**
 * @brief Perform a sequential exclusive prefix scan over the first @p valid_items elements
 *        of the statically-sized @p input array, seeded with the specified @p prefix.
 *        Only the first @p valid_items elements of @p output are modified.
 *
 * @tparam Input
 *   <b>[inferred]</b> The data type holding the input data having member
 *   <tt>operator[](int i)</tt> and must be statically-sized (<tt>size()</tt> method or static array)
 *
 * @tparam Output
 *   <b>[inferred]</b> The data type to hold the scan result having member
 *   <tt>operator[](int i)</tt> and must be statically-sized (<tt>size()</tt> method or static array)
 *
 * @tparam ScanOp
 *   <b>[inferred]</b> Binary scan operator type having member
 *   <tt>T operator()(const T &a, const T &b)</tt>
 *
 * @tparam PrefixT
 *   <b>[inferred]</b> The data type of the prefix.
 *
 * @param[in] input
 *   Array-like input
 *
 * @param[out] output
 *   Array-like output (may be aliased to @p input)
 *
 * @param[in] scan_op
 *   Binary scan operator
 *
 * @param[in] valid_items
 *   Number of valid items
 *
 * @param[in] prefix
 *   Prefix to seed scan with
 *
 * @param[in] apply_prefix
 *   Whether or not the calling thread should apply its prefix.
 *   (Handy for preventing thread-0 from applying a prefix.)
 */
template <typename Input,
          typename Output,
          typename ScanOp,
          typename PrefixT,
          typename ValueT = ::cuda::std::iter_value_t<Input>,
          typename AccumT = ::cuda::std::__accumulator_t<ScanOp, ValueT, PrefixT>>
_CCCL_DEVICE _CCCL_FORCEINLINE void ThreadScanExclusivePartial(
  Input& input, Output& output, ScanOp scan_op, int valid_items, PrefixT prefix, bool apply_prefix = true)
{
  static_assert(is_fixed_size_random_access_range_v<Input>,
                "Input must support the subscript operator[] and have a compile-time size");
  static_assert(is_fixed_size_random_access_range_v<Output>,
                "Output must support the subscript operator[] and have a compile-time size");
  static_assert(has_binary_call_operator<ScanOp, ValueT>::value,
                "ScanOp must have the binary call operator: operator(ValueT, ValueT)");
  constexpr auto length = static_size_v<Input>;
  static_assert(static_size_v<Output> == length);
  AccumT inclusive = input[0];
  if (valid_items > 0)
  {
    if (apply_prefix)
    {
      inclusive = scan_op(prefix, inclusive);
    }
    output[0] = prefix;
  }

  AccumT exclusive = inclusive;

  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 1; i < length; ++i)
  {
    if (i < valid_items)
    {
      inclusive = scan_op(exclusive, input[i]);
      output[i] = exclusive;
      exclusive = inclusive;
    }
  }
}

/**
 * @param[in] input
 *   Input array
 *
 * @param[out] output
 *   Output array (may be aliased to @p input)
 *
 * @param[in] scan_op
 *   Binary scan operator
 */
template <int LENGTH, typename T, typename ScanOp>
_CCCL_DEVICE _CCCL_FORCEINLINE T
ThreadScanInclusive(T inclusive, T* input, T* output, ScanOp scan_op, detail::constant_t<LENGTH> /*length*/)
{
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < LENGTH; ++i)
  {
    inclusive = scan_op(inclusive, input[i]);
    output[i] = inclusive;
  }

  return inclusive;
}

/**
 * @brief Perform a sequential inclusive prefix scan over
 *        @p LENGTH elements of the @p input array. The aggregate is returned.
 *
 * @tparam LENGTH
 *   LengthT of @p input and @p output arrays
 *
 * @tparam T
 *   <b>[inferred]</b> The data type to be scanned.
 *
 * @tparam ScanOp
 *   <b>[inferred]</b> Binary scan operator type having member
 *   <tt>T operator()(const T &a, const T &b)</tt>
 *
 * @param[in] input
 *   Input array
 *
 * @param[out] output
 *   Output array (may be aliased to @p input)
 *
 * @param[in] scan_op
 *   Binary scan operator
 */
template <int LENGTH, typename T, typename ScanOp>
_CCCL_DEVICE _CCCL_FORCEINLINE T ThreadScanInclusive(T* input, T* output, ScanOp scan_op)
{
  T inclusive = input[0];
  output[0]   = inclusive;

  // Continue scan
  return ThreadScanInclusive(inclusive, input + 1, output + 1, scan_op, detail::constant_v<LENGTH - 1>);
}

/**
 * @brief Perform a sequential inclusive prefix scan over the
 *        statically-sized @p input array. The aggregate is returned.
 *
 * @tparam LENGTH
 *   <b>[inferred]</b> LengthT of @p input and @p output arrays
 *
 * @tparam T
 *   <b>[inferred]</b> The data type to be scanned.
 *
 * @tparam ScanOp
 *   <b>[inferred]</b> Binary scan operator type having member
 *   <tt>T operator()(const T &a, const T &b)</tt>
 *
 * @param[in] input
 *   Input array
 *
 * @param[out] output
 *   Output array (may be aliased to @p input)
 *
 * @param[in] scan_op
 *   Binary scan operator
 */
template <int LENGTH, typename T, typename ScanOp>
_CCCL_DEVICE _CCCL_FORCEINLINE T ThreadScanInclusive(T (&input)[LENGTH], T (&output)[LENGTH], ScanOp scan_op)
{
  return ThreadScanInclusive<LENGTH>((T*) input, (T*) output, scan_op);
}

/**
 * @brief Perform a sequential inclusive prefix scan over
 *        @p LENGTH elements of the @p input array, seeded with the
 *        specified @p prefix. The aggregate is returned.
 *
 * @tparam LENGTH
 *   LengthT of @p input and @p output arrays
 *
 * @tparam T
 *   <b>[inferred]</b> The data type to be scanned.
 *
 * @tparam ScanOp
 *   <b>[inferred]</b> Binary scan operator type having member
 *   <tt>T operator()(const T &a, const T &b)</tt>
 *
 * @param[in] input
 *   Input array
 *
 * @param[out] output
 *   Output array (may be aliased to @p input)
 *
 * @param[in] scan_op
 *   Binary scan operator
 *
 * @param[in] prefix
 *   Prefix to seed scan with
 *
 * @param[in] apply_prefix
 *   Whether or not the calling thread should apply its prefix.
 *   (Handy for preventing thread-0 from applying a prefix.)
 */
template <int LENGTH, typename T, typename ScanOp>
_CCCL_DEVICE _CCCL_FORCEINLINE T
ThreadScanInclusive(T* input, T* output, ScanOp scan_op, T prefix, bool apply_prefix = true)
{
  T inclusive = input[0];
  if (apply_prefix)
  {
    inclusive = scan_op(prefix, inclusive);
  }
  output[0] = inclusive;

  // Continue scan
  return ThreadScanInclusive(inclusive, input + 1, output + 1, scan_op, detail::constant_v<LENGTH - 1>);
}

/**
 * @brief Perform a sequential inclusive prefix scan over the
 *        statically-sized @p input array, seeded with the specified @p prefix.
 *        The aggregate is returned.
 *
 * @tparam LENGTH
 *   <b>[inferred]</b> LengthT of @p input and @p output arrays
 *
 * @tparam T
 *   <b>[inferred]</b> The data type to be scanned.
 *
 * @tparam ScanOp
 *   <b>[inferred]</b> Binary scan operator type having member
 *   <tt>T operator()(const T &a, const T &b)</tt>
 *
 * @param[in] input
 *   Input array
 *
 * @param[out] output
 *   Output array (may be aliased to @p input)
 *
 * @param[in] scan_op
 *   Binary scan operator
 *
 * @param[in] prefix
 *   Prefix to seed scan with
 *
 * @param[in] apply_prefix
 *   Whether or not the calling thread should apply its prefix.
 *   (Handy for preventing thread-0 from applying a prefix.)
 */
template <int LENGTH, typename T, typename ScanOp>
_CCCL_DEVICE _CCCL_FORCEINLINE T
ThreadScanInclusive(T (&input)[LENGTH], T (&output)[LENGTH], ScanOp scan_op, T prefix, bool apply_prefix = true)
{
  return ThreadScanInclusive<LENGTH>((T*) input, (T*) output, scan_op, prefix, apply_prefix);
}

/**
 * @brief Perform a sequential inclusive prefix scan over the first @p valid_items elements
 *        of the statically-sized @p input array, seeded with the specified @p prefix.
 *        Only the first @p valid_items elements of @p output are modified.
 *
 * @tparam Input
 *   <b>[inferred]</b> The data type holding the input data having member
 *   <tt>operator[](int i)</tt> and must be statically-sized (<tt>size()</tt> method or static array)
 *
 * @tparam Output
 *   <b>[inferred]</b> The data type to hold the scan result having member
 *   <tt>operator[](int i)</tt> and must be statically-sized (<tt>size()</tt> method or static array)
 *
 * @tparam ScanOp
 *   <b>[inferred]</b> Binary scan operator type having member
 *   <tt>T operator()(const T &a, const T &b)</tt>
 *
 * @tparam PrefixT
 *   <b>[inferred]</b> The data type of the prefix.
 *
 * @param[in] input
 *   Array-like input
 *
 * @param[out] output
 *   Array-like output (may be aliased to @p input)
 *
 * @param[in] scan_op
 *   Binary scan operator
 *
 * @param[in] valid_items
 *   Number of valid items
 *
 * @param[in] prefix
 *   Prefix to seed scan with
 *
 * @param[in] apply_prefix
 *   Whether or not the calling thread should apply its prefix.
 *   (Handy for preventing thread-0 from applying a prefix.)
 */
template <typename Input,
          typename Output,
          typename ScanOp,
          typename PrefixT,
          typename ValueT = ::cuda::std::iter_value_t<Input>,
          typename AccumT = ::cuda::std::__accumulator_t<ScanOp, ValueT, PrefixT>>
_CCCL_DEVICE _CCCL_FORCEINLINE void ThreadScanInclusivePartial(
  Input& input, Output& output, ScanOp scan_op, int valid_items, PrefixT prefix, bool apply_prefix = true)
{
  static_assert(is_fixed_size_random_access_range_v<Input>,
                "Input must support the subscript operator[] and have a compile-time size");
  static_assert(is_fixed_size_random_access_range_v<Output>,
                "Output must support the subscript operator[] and have a compile-time size");
  static_assert(has_binary_call_operator<ScanOp, ValueT>::value,
                "ScanOp must have the binary call operator: operator(ValueT, ValueT)");
  constexpr auto length = static_size_v<Input>;
  static_assert(static_size_v<Output> == length);
  AccumT inclusive = input[0];
  if (valid_items > 0)
  {
    if (apply_prefix)
    {
      inclusive = scan_op(prefix, inclusive);
    }
    output[0] = inclusive;
  }

  // Continue scan
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 1; i < length; ++i)
  {
    if (i < valid_items)
    {
      inclusive = scan_op(inclusive, input[i]);
      output[i] = inclusive;
    }
  }
}

//@}  end member group

} // namespace detail
CUB_NAMESPACE_END
