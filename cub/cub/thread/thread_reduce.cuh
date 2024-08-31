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

#include <cuda/cmath> // ceil_div
#include <cuda/std/__cccl/attributes.h> // _CCCL_NODISCARD
#include <cuda/std/bit> // bit_cast
#include <cuda/std/cstdint> // uint16_t
#include <cuda/std/limits> // numeric_limits
#include <cuda/std/type_traits> // __enable_if_t
#include <cuda/std/utility> // pair

#include "cuda/std/__cccl/dialect.h"

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/type_traits.cuh> // are_same()
#include <cub/thread/thread_operators.cuh> // DpxMin
#include <cub/util_namespace.cuh>
#include <cub/util_type.cuh>

CUB_NAMESPACE_BEGIN

/// Internal namespace (to prevent ADL mishaps between static functions when mixing different CUB installations)
namespace internal
{

#ifndef DOXYGEN_SHOULD_SKIP_THIS // Do not document

/// DPX instructions compute min, max, and sum for up to three 16 and 32-bit signed or unsigned integer parameters
/// see DPX documetation https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#dpx
/// NOTE: The compiler is able to automatically vectorize all cases with 3 operands
///       However, all other cases with per-halfword comparison need to be explicitly vectorized
/// TODO: Remove DPX specilization when nvbug 4823237 is fixed
///
/// DPX reduction is enabled if the following conditions are met:
/// - Hopper+ architectures. DPX instructions are emulated before Hopper
/// - The number of elements must be large enough for performance reasons (see below)
/// - All types must be the same
/// - Only works with integral types of 2 bytes
/// - DPX instructions provide Min, Max, and Sum SIMD operations
/// If the number of instructions is the same, we favor the compiler

template <int LENGTH, typename T, typename ReductionOp, typename PrefixT = T, typename AccumT = T>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE // clang-format off
_CCCL_CONSTEXPR_CXX14 bool enable_dpx_reduction()
{
  return ((LENGTH >= 9 && ::cuda::std::is_same<ReductionOp, cub::Sum>::value) || LENGTH >= 10)
            && detail::are_same<T, PrefixT, AccumT>()
            && detail::is_one_of<T, int16_t, uint16_t>()
            && detail::is_one_of<ReductionOp, cub::Min, cub::Max, cub::Sum>();
}
// clang-format on

// Considering compiler vectorization with 3-way comparison, the number of SASS instructions is
// Standard: ceil((L - 3) / 2) + 1
//   replacing L with L/2 for SIMD
// DPX:      ceil((L/2 - 3) / 2) + 1 + 2 [for halfword comparison: PRMT, VIMNMX] + L % 2 [for last element]
//   finally, the last two comparision operations are vectorized in a 3-way reduction
//           ceil((L/2 - 3) / 2) + 3
//
// LENGTH | Standard |  DPX
//  2     |    1     |  NA
//  3     |    1     |  NA
//  4     |    2     |  3
//  5     |    2     |  3
//  6     |    3     |  3
//  7     |    3     |  3
//  8     |    4     |  4
//  9     |    4     |  4
// 10     |    5     |  4 // ***
// 11     |    5     |  4 // ***
// 12     |    6     |  5 // ***
// 13     |    6     |  5 // ***
// 14     |    7     |  5 // ***
// 15     |    7     |  5 // ***
// 16     |    8     |  6 // ***

// Forward declaration
template <int LENGTH,
          typename T,
          typename ReductionOp,
          bool ENABLE_DPX = enable_dpx_reduction<LENGTH, T, ReductionOp>(),
          _CUB_TEMPLATE_REQUIRES(ENABLE_DPX)>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE T ThreadReduce(T* input, ReductionOp reduction_op);

#endif // DOXYGEN_SHOULD_SKIP_THIS

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
template <int LENGTH,
          typename T,
          typename ReductionOp,
          typename PrefixT,
          typename AccumT = ::cuda::std::__accumulator_t<ReductionOp, T, PrefixT>>
_CCCL_DEVICE _CCCL_FORCEINLINE AccumT ThreadReduce(T* input, ReductionOp reduction_op, PrefixT prefix)
{
  _CCCL_IF_CONSTEXPR (enable_dpx_reduction<LENGTH, T, ReductionOp, PrefixT, AccumT>())
  {
    return reduction_op(ThreadReduce<LENGTH>(input, reduction_op), prefix);
  }
  else
  {
    AccumT retval = prefix;
#pragma unroll
    for (int i = 0; i < LENGTH; ++i)
    {
      retval = reduction_op(retval, input[i]);
    }
    return retval;
  }
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
template <int LENGTH,
          typename T,
          typename ReductionOp,
          bool ENABLE_DPX = enable_dpx_reduction<LENGTH, T, ReductionOp>(),
          _CUB_TEMPLATE_REQUIRES(!ENABLE_DPX)>
_CCCL_DEVICE _CCCL_FORCEINLINE T ThreadReduce(T* input, ReductionOp reduction_op)
{
  _CCCL_IF_CONSTEXPR (LENGTH == 1)
  {
    return input[0];
  }
  else
  {
    T prefix = input[0];
    return ThreadReduce<LENGTH - 1>(input + 1, reduction_op, prefix);
  }
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS // Do not document

/// Specialization for DPX reduction
template <int LENGTH, typename T, typename ReductionOp, bool ENABLE_DPX, ::cuda::std::__enable_if_t<ENABLE_DPX>*>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE T ThreadReduce(T* input, ReductionOp reduction_op)
{
  // clang-format off
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (using DpxReduceOp   = cub_operator_to_dpx_t<ReductionOp, T>;
     using SimdType      = ::cuda::std::pair<T, T>;
     auto unsigned_input = reinterpret_cast<unsigned*>(input);
     auto simd_reduction = ThreadReduce<LENGTH / 2>(unsigned_input, DpxReduceOp{});
     auto simd_values    = ::cuda::std::bit_cast<SimdType>(simd_reduction);
     auto ret_value      = reduction_op(simd_values.first, simd_values.second);
     return (LENGTH % 2 == 0) ? ret_value : reduction_op(ret_value, input[LENGTH - 1]);),
    // < SM90
    (return ThreadReduce<LENGTH, T, ReductionOp, false>(input, reduction_op);))
  // clang-format on
}

#endif // !DOXYGEN_SHOULD_SKIP_THIS

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
          typename AccumT = ::cuda::std::__accumulator_t<ReductionOp, T, PrefixT>>
_CCCL_DEVICE _CCCL_FORCEINLINE AccumT ThreadReduce(T (&input)[LENGTH], ReductionOp reduction_op, PrefixT prefix)
{
  return ThreadReduce<LENGTH>(static_cast<T*>(input), reduction_op, prefix);
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
  return ThreadReduce<LENGTH>(static_cast<T*>(input), reduction_op);
}

} // namespace internal
CUB_NAMESPACE_END
