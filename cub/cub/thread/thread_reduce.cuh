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
 * Thread utilities for sequential reduction over statically-sized array types
 */

#pragma once

#include <cub/config.cuh>
#include <cuda/cmath>                   // ceil_div
#include <cuda/std/__cccl/attributes.h> // _CCCL_NODISCARD
#include <cuda/std/cstdint>             // uint16_t
#include <cuda/std/limits>              // numeric_limits
#include <cuda/std/type_traits>         // enable_if_t

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/type_traits.cuh>        // are_same_v
#include <cub/thread/thread_operators.cuh>  // DpxMin
#include <cub/util_namespace.cuh>
#include <cub/util_type.cuh>

CUB_NAMESPACE_BEGIN

/// Internal namespace (to prevent ADL mishaps between static functions when mixing different CUB installations)
namespace internal
{

//----------------------------------------------------------------------------------------------------------------------

/// Enable DPX reduction if the following conditions are met:
/// - Hopper+ architectures. DPX instructions are emulated before Hopper.
/// - The number of elements must be large enough for performance reasons (see below)
/// - All types must be the same
/// - Only works with integral types of 1 bytes or 2 bytes
/// - DPX instructions provide Min and Max SIMD operations

// clang-format off
template <int LENGTH, typename T, typename ReductionOp, typename PrefixT = T, typename AccumT = T>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE
constexpr bool enable_dpx_reduction() // clang-format on
{
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (return (LENGTH == 6 || LENGTH == 8 || LENGTH >= 10) && detail::are_same_v<T, PrefixT, AccumT>
           && detail::is_one_of_v<T, int16_t, uint16_t> && detail::is_one_of_v<ReductionOp, cub::Min, cub::Max>;),
    (return false;));
}

// Considering compiler vectorization with 3-way reduction, the number of SASS instructions is
// standard: ceil((L - 3) / 2) + 1
//   replacing L with L/2 for SIMD
// DPX:      ceil((L/2 - 3) / 2) + 1 + 1 [for halfword comparison] + L % 2 [for last element]
//
// LENGTH | Standard |  DPX
//  2     |    1     |  NA
//  3     |    1     |  NA
//  4     |    2     |  3
//  5     |    2     |  4
//  6     |    3     |  2 // *** (3-way comparison for DPX)
//  7     |    3     |  3
//  8     |    4     |  3 // ***
//  9     |    4     |  4
// 10     |    5     |  3 // ***
// 11     |    5     |  4 // ***
// 12     |    6     |  4 // ***
// 13     |    6     |  5 // ***
// 14     |    7     |  4 // ***
// 15     |    7     |  5 // ***
// 16     |    8     |  5 // ***

//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief Sequential reduction over statically-sized array types
 *
 * @param[in] input
 *   Input array
 *
 * @param[in] reduction_op
 *   Binary reduction operator
 *
 * @param[in] prefix
 *   Prefix to seed reduction with
 */
// clang-format off
template <int LENGTH,
          typename T,
          typename ReductionOp,
          typename PrefixT,
          typename AccumT = detail::accumulator_t<ReductionOp, PrefixT, T>>
_CCCL_DEVICE _CCCL_FORCEINLINE
::cuda::std::enable_if_t<!enable_dpx_reduction<LENGTH, T, ReductionOp, PrefixT, AccumT>(), AccumT>
ThreadReduce(T* input, ReductionOp reduction_op, PrefixT prefix, Int2Type<LENGTH> /*length*/) // clang-format on
{
  AccumT retval = prefix;
#pragma unroll
  for (int i = 0; i < LENGTH; ++i)
  {
    retval = reduction_op(retval, input[i]);
  }
  return retval;
}

//----------------------------------------------------------------------------------------------------------------------

/// Specialization for single-element arrays
template <int LENGTH, typename T, typename ReductionOp>
_CCCL_DEVICE _CCCL_FORCEINLINE ::cuda::std::enable_if_t<(LENGTH == 1), T>
ThreadReduce(T* input, ReductionOp reduction_op)
{
  return input[0];
}

/**
 * @brief Perform a sequential reduction over @p LENGTH elements of the @p input array.
 *        The aggregate is returned.
 *
 * @tparam LENGTH
 *   LengthT of input array
 *
 * @tparam T
 *   <b>[inferred]</b> The data type to be reduced.
 *
 * @tparam ReductionOp
 *   <b>[inferred]</b> Binary reduction operator type having member
 *   <tt>T operator()(const T &a, const T &b)</tt>
 *
 * @param[in] input
 *   Input array
 *
 * @param[in] reduction_op
 *   Binary reduction operator
 */
template <int LENGTH, typename T, typename ReductionOp>
_CCCL_DEVICE _CCCL_FORCEINLINE // clang-format off
::cuda::std::enable_if_t<!enable_dpx_reduction<LENGTH, T, ReductionOp>(), T>
ThreadReduce(T* input, ReductionOp reduction_op) // clang-format on
{
  T prefix = input[0];
  return ThreadReduce(input + 1, reduction_op, prefix, Int2Type<LENGTH - 1>{});
}

/// Specialization for DPX reduction
template <int LENGTH, // clang-format off
          typename T,
          typename ReductionOp>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE
::cuda::std::enable_if_t<enable_dpx_reduction<LENGTH, T, ReductionOp>(), T> // clang-format on
ThreadReduce(T* input, ReductionOp reduction_op)
{
  constexpr auto IS_MIN = ::cuda::std::is_same_v<ReductionOp, cub::Min>;
  using DpxReduceOp     = ::cuda::std::conditional_t<IS_MIN, DpxMin<T>, DpxMax<T>>;
  auto unsigned_input   = reinterpret_cast<unsigned*>(input);
  auto simd_reduction   = ThreadReduce<LENGTH / 2>(unsigned_input, DpxReduceOp{});
  T simd_values[2]; // TODO (fbusato): use bit_cast
  ::memcpy(simd_values, &simd_reduction, sizeof(simd_values));
  auto ret_value = reduction_op(simd_values[0], simd_values[1]);
  return (LENGTH % 2 == 0) ? ret_value : reduction_op(ret_value, input[LENGTH - 1]);
}

/// Specialization for DPX reduction with prefix
template <int LENGTH, // clang-format off
          typename T,
          typename ReductionOp,
          typename PrefixT,
          typename AccumT = detail::accumulator_t<ReductionOp, PrefixT, T>>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE
::cuda::std::enable_if_t<enable_dpx_reduction<LENGTH, T, ReductionOp, PrefixT, AccumT>(), T>
ThreadReduce(T* input, ReductionOp reduction_op, PrefixT prefix, Int2Type<LENGTH>) // clang-format on
{
  return reduction_op(ThreadReduce<LENGTH>(input, reduction_op), prefix);
}

/**
 * @brief Perform a sequential reduction over @p LENGTH elements of the @p input array,
 *        seeded with the specified @p prefix. The aggregate is returned.
 *
 * @tparam LENGTH
 *   LengthT of input array
 *
 * @tparam T
 *   <b>[inferred]</b> The data type to be reduced.
 *
 * @tparam ReductionOp
 *   <b>[inferred]</b> Binary reduction operator type having member
 *   <tt>T operator()(const T &a, const T &b)</tt>
 *
 * @param[in] input
 *   Input array
 *
 * @param[in] reduction_op
 *   Binary reduction operator
 *
 * @param[in] prefix
 *   Prefix to seed reduction with
 */
template <int LENGTH,
          typename T,
          typename ReductionOp,
          typename PrefixT,
          typename AccumT = detail::accumulator_t<ReductionOp, PrefixT, T>>
_CCCL_DEVICE _CCCL_FORCEINLINE AccumT ThreadReduce(T* input, ReductionOp reduction_op, PrefixT prefix)
{
  return ThreadReduce(input, reduction_op, prefix, Int2Type<LENGTH>());
}

/**
 * @brief Perform a sequential reduction over the statically-sized @p input array,
 *        seeded with the specified @p prefix. The aggregate is returned.
 *
 * @tparam LENGTH
 *   <b>[inferred]</b> LengthT of @p input array
 *
 * @tparam T
 *   <b>[inferred]</b> The data type to be reduced.
 *
 * @tparam ReductionOp
 *   <b>[inferred]</b> Binary reduction operator type having member
 *   <tt>T operator()(const T &a, const T &b)</tt>
 *
 * @param[in] input
 *   Input array
 *
 * @param[in] reduction_op
 *   Binary reduction operator
 *
 * @param[in] prefix
 *   Prefix to seed reduction with
 */
template <int LENGTH,
          typename T,
          typename ReductionOp,
          typename PrefixT,
          typename AccumT = detail::accumulator_t<ReductionOp, PrefixT, T>>
_CCCL_DEVICE _CCCL_FORCEINLINE AccumT ThreadReduce(T (&input)[LENGTH], ReductionOp reduction_op, PrefixT prefix)
{
  return ThreadReduce(input, reduction_op, prefix, Int2Type<LENGTH>());
}

/**
 * @brief Serial reduction with the specified operator
 *
 * @tparam LENGTH
 *   <b>[inferred]</b> LengthT of @p input array
 *
 * @tparam T
 *   <b>[inferred]</b> The data type to be reduced.
 *
 * @tparam ReductionOp
 *   <b>[inferred]</b> Binary reduction operator type having member
 *   <tt>T operator()(const T &a, const T &b)</tt>
 *
 * @param[in] input
 *   Input array
 *
 * @param[in] reduction_op
 *   Binary reduction operator
 */
template <int LENGTH, typename T, typename ReductionOp>
_CCCL_DEVICE _CCCL_FORCEINLINE T ThreadReduce(T (&input)[LENGTH], ReductionOp reduction_op)
{
  return ThreadReduce<LENGTH>((T*) input, reduction_op);
}

} // namespace internal
CUB_NAMESPACE_END
