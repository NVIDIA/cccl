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

//! @file
//! Thread reduction over statically-sized array-like types
#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/array_utils.cuh> // to_array()
#include <cub/detail/type_traits.cuh> // are_same()
#include <cub/thread/thread_load.cuh> // UnrolledCopy
#include <cub/thread/thread_operators.cuh> // cub_operator_to_dpx_t
#include <cub/util_namespace.cuh>

#include <cuda/functional> // cuda::std::maximum
#include <cuda/std/array> // array
#include <cuda/std/bit> // bit_cast
#include <cuda/std/cassert> // assert
#include <cuda/std/cstdint> // uint16_t
#include <cuda/std/functional> // cuda::std::plus

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

//! @rst
//! The ``ThreadReduce`` function computes a reduction of items assigned to a single CUDA thread.
//!
//! Overview
//! ++++++++++++++++++++++++++
//!
//! - A `reduction <http://en.wikipedia.org/wiki/Reduce_(higher-order_function)>`__ (or *fold*)
//!   uses a binary combining operator to compute a single aggregate from a list of input elements.
//! - Supports array-like types that are statically-sized and can be indexed with the ``[] operator``:
//!   raw arrays, ``std::array``, ``std::span``,  ``std::mdspan`` (C++23)
//!
//! Overloading
//! ++++++++++++++++++++++++++
//!
//! Reduction over statically-sized array-like types, seeded with the specified prefix
//!
//! .. code-block:: c++
//!
//!    template <typename Input,
//!              typename ReductionOp,
//!              typename PrefixT,
//!              typename ValueT = ::cuda::std::remove_cvref_t<decltype(::cuda::std::declval<Input>()[0])>,
//!              typename AccumT = ::cuda::std::__accumulator_t<ReductionOp, ValueT, PrefixT>>
//!    _CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE AccumT
//!    ThreadReduce(const Input& input, ReductionOp reduction_op, PrefixT prefix)
//!
//! Performance Considerations
//! ++++++++++++++++++++++++++
//!
//! The function provides the following optimizations
//!
//! - Vectorization/SIMD for
//!
//!   - Sum (``cuda::std::plus<>``) and Multiplication (``cuda::std::multiplies<>``) on SM70+ for ``__half`` data type
//!   - Sum (``cuda::std::plus<>``) and Multiplication (``cuda::std::multiplies<>``) on SM80+ for ``__nv_bfloat16``
//!     data type
//!   - Minimum (``cuda::minimum<>``) and Maximum (``cuda::maximum<>``) on SM80+ for ``__half`` and ``__nv_bfloat16``
//!     data types
//!   - Minimum (``cuda::minimum<>``) and Maximum (``cuda::maximum<>``) on SM90+ for ``int16_t`` and ``uint16_t``
//!     data types (Hopper DPX instructions)
//!
//! - Instruction-Level Parallelism (ILP) by exploiting a ternary tree reduction for
//!
//!   - Sum (``cuda::std::plus<>``), Bitwise AND (``cuda::std::bit_and<>``), OR (``cuda::std::bit_or<>``), XOR
//!     (``cuda::std::bit_xor<>``) on SM50+ for integer data types
//!   - Minimum (``cuda::minimum<>``) and Maximum (``cuda::maximum<>``) on SM80+ for integer data types (Hopper DPX
//!     instructions), ``__half2``, ``__nv_bfloat162``, ``__half`` (after vectorization), and ``__nv_bfloat16``
//!     (after vectorization) data types
//!   - Minimum (``cuda::minimum<>``) and Maximum (``cuda::maximum<>``) on SM90+ for integer data types (Hopper DPX
//!     instructions)
//!
//! - Instruction-Level Parallelism (ILP) by exploiting a binary tree reduction for all other cases
//!
//! Simple Example
//! ++++++++++++++++++++++++++
//!
//! The code snippet below illustrates a simple sum reductions over 4 integer values.
//!
//! .. code-block:: c++
//!
//!    #include <cub/cub.cuh>
//!
//!    __global__ void ExampleKernel(...)
//!    {
//!        int array[4] = {1, 2, 3, 4};
//!        int sum      = cub::ThreadReduce(array, ::cuda::std::plus<>()); // sum = 10
//!
//! @endrst
//!
//! @brief Reduction over statically-sized array-like types.
//!
//! @tparam Input
//!   <b>[inferred]</b> The data type to be reduced having member
//!   <tt>operator[](int i)</tt> and must be statically-sized (size() method or static array)
//!
//! @tparam ReductionOp
//!   <b>[inferred]</b> Binary reduction operator type having member
//!   <tt>T operator()(const T &a, const T &b)</tt>
//!
//! @param[in] input
//!   Array=like input
//!
//! @param[in] reduction_op
//!   Binary reduction operator
//!
//! @return Aggregate of type <tt>cuda::std::__accumulator_t<ReductionOp, ValueT, PrefixT></tt>
//!
template <typename Input,
          typename ReductionOp,
#ifndef _CCCL_DOXYGEN_INVOKED // Do not document
          typename ValueT = ::cuda::std::remove_cvref_t<decltype(::cuda::std::declval<Input>()[0])>,
#else
          typename ValueT = random_access_value_t<Input>,
#endif // !_CCCL_DOXYGEN_INVOKED
          typename AccumT = ::cuda::std::__accumulator_t<ReductionOp, ValueT>>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE AccumT ThreadReduce(const Input& input, ReductionOp reduction_op);
// forward declaration

/***********************************************************************************************************************
 * Internal Reduction Implementations
 **********************************************************************************************************************/

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document

namespace detail
{

// NOTE: bit_cast cannot be always used because __half, __nv_bfloat16, etc. are not trivially copyable
template <typename Output, typename Input>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE Output unsafe_bitcast(const Input& input)
{
  Output output;
  static_assert(sizeof(input) == sizeof(output), "wrong size");
  ::memcpy(&output, &input, sizeof(input));
  return output;
}

} // namespace detail

/// Internal namespace (to prevent ADL mishaps between static functions when mixing different CUB installations)
namespace internal
{

/***********************************************************************************************************************
 * Enable SIMD/Tree reduction heuristics
 **********************************************************************************************************************/

/// DPX instructions compute min, max, and sum for up to three 16 and 32-bit signed or unsigned integer parameters
/// see DPX documentation https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#dpx
/// NOTE: The compiler is able to automatically vectorize all cases with 3 operands
///       However, all other cases with per-halfword comparison need to be explicitly vectorized
///
/// DPX reduction is enabled if the following conditions are met:
/// - Hopper+ architectures. DPX instructions are emulated before Hopper
/// - The number of elements must be large enough for performance reasons (see below)
/// - All types must be the same
/// - Only works with integral types of 2 bytes
/// - DPX instructions provide Min, Max SIMD operations
/// If the number of instructions is the same, we favor the compiler
///
/// length | Standard |  DPX
///  2     |    1     |  NA
///  3     |    1     |  NA
///  4     |    2     |  3
///  5     |    2     |  3
///  6     |    3     |  3
///  7     |    3     |  3
///  8     |    4     |  4
///  9     |    4     |  4
/// 10     |    5     |  4 // ***
/// 11     |    5     |  4 // ***
/// 12     |    6     |  5 // ***
/// 13     |    6     |  5 // ***
/// 14     |    7     |  5 // ***
/// 15     |    7     |  5 // ***
/// 16     |    8     |  6 // ***

// TODO: add Blackwell support

//----------------------------------------------------------------------------------------------------------------------
// SM90 SIMD

template <typename T, typename ReductionOp, int Length>
inline constexpr bool enable_sm90_simd_reduction_v =
  is_one_of_v<T, int16_t, uint16_t> && is_predefined_comparison_v<ReductionOp, T> && Length >= 10;

//----------------------------------------------------------------------------------------------------------------------
// SM80 SIMD

template <typename T, typename ReductionOp, int Length>
inline constexpr bool enable_sm80_simd_reduction_v = false;

#  if defined(_CCCL_HAS_NVFP16)

template <typename ReductionOp, int Length>
inline constexpr bool enable_sm80_simd_reduction_v<__half, ReductionOp, Length> =
  (is_predefined_comparison_v<ReductionOp, __half> || is_predefined_arithmetic_v<ReductionOp, __half>) && Length >= 4;

#  endif // defined(_CCCL_HAS_NVFP16)

#  if defined(_CCCL_HAS_NVBF16)

template <typename ReductionOp, int Length>
inline constexpr bool enable_sm80_simd_reduction_v<__nv_bfloat16, ReductionOp, Length> =
  (is_predefined_comparison_v<ReductionOp, __nv_bfloat16> || is_predefined_arithmetic_v<ReductionOp, __nv_bfloat16>)
  && Length >= 4;

#  endif // defined(_CCCL_HAS_NVBF16)

//----------------------------------------------------------------------------------------------------------------------
// SM70 SIMD

#  if defined(_CCCL_HAS_NVFP16)

template <typename T, typename ReductionOp, int Length>
inline constexpr bool enable_sm70_simd_reduction_v =
  ::cuda::std::is_same_v<T, __half> && is_predefined_arithmetic_v<ReductionOp, T> && Length >= 4;

#  else // defined(_CCCL_HAS_NVFP16) ^^^^ / !defined(_CCCL_HAS_NVFP16) vvvv

template <typename T, typename ReductionOp, int Length>
inline constexpr bool enable_sm70_simd_reduction_v = false;

#  endif // !defined(_CCCL_HAS_NVFP16) ^^^^

//----------------------------------------------------------------------------------------------------------------------
// All architectures SIMD

template <typename Input, typename ReductionOp, typename AccumT>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE constexpr bool enable_simd_reduction()
{
  using T = detail::random_access_range_elem_t<Input>;
  if constexpr (!::cuda::std::is_same_v<T, AccumT>)
  {
    return false;
  }
  else
  {
    constexpr auto length = cub::detail::static_size_v<Input>;
    // clang-format off
    _NV_TARGET_DISPATCH(
      NV_PROVIDES_SM_90,
        (return enable_sm90_simd_reduction_v<T, ReductionOp, length> ||
                enable_sm80_simd_reduction_v<T, ReductionOp, length> ||
                enable_sm70_simd_reduction_v<T, ReductionOp, length>;),
      NV_PROVIDES_SM_80,
        (return enable_sm80_simd_reduction_v<T, ReductionOp, length> ||
                enable_sm70_simd_reduction_v<T, ReductionOp, length>;),
      NV_PROVIDES_SM_70,
        (return enable_sm70_simd_reduction_v<T, ReductionOp, length>;),
      NV_IS_DEVICE,
        (static_cast<void>(length); // maybe unused
         return false;)
    );
    // clang-format on
    return false;
  }
}

/***********************************************************************************************************************
 * enable_ternary_reduction
 **********************************************************************************************************************/

template <typename T, typename ReductionOp>
inline constexpr bool enable_ternary_reduction_sm90_v =
  cub::detail::is_one_of_v<T, int32_t, uint32_t>
  && (is_predefined_comparison_v<ReductionOp, T> || is_predefined_bitwise_v<ReductionOp, T>
      || cub::detail::is_one_of_v<ReductionOp, ::cuda::std::plus<>, ::cuda::std::plus<T>>);

#  if defined(_CCCL_HAS_NVFP16)

template <typename ReductionOp>
inline constexpr bool enable_ternary_reduction_sm90_v<__half2, ReductionOp> =
  is_predefined_comparison_v<ReductionOp, __half2> || is_one_of_v<ReductionOp, SimdMin<__half>, SimdMax<__half>>;

#  endif // defined(_CCCL_HAS_NVFP16)

#  if defined(_CCCL_HAS_NVBF16)

template <typename ReductionOp>
inline constexpr bool enable_ternary_reduction_sm90_v<__nv_bfloat162, ReductionOp> =
  is_predefined_comparison_v<ReductionOp, __nv_bfloat162>
  || is_one_of_v<ReductionOp, SimdMin<__nv_bfloat16>, SimdMax<__nv_bfloat16>>;

#  endif // defined(_CCCL_HAS_NVBF16)

template <typename Input, typename ReductionOp, typename AccumT>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE constexpr bool enable_ternary_reduction()
{
  constexpr auto length = cub::detail::static_size_v<Input>;
  if constexpr (length < 6)
  {
    return false;
  }
  else
  {
    using T = detail::random_access_range_elem_t<Input>;
    using cub::detail::is_one_of_v;
    // clang-format off
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_90,
        (return enable_ternary_reduction_sm90_v<T, ReductionOp>;),
      NV_PROVIDES_SM_50,
        (return is_one_of_v<AccumT, int32_t, uint32_t>
             && is_one_of_v<ReductionOp, ::cuda::std::plus<>,    ::cuda::std::plus<T>,
                                       ::cuda::std::bit_and<>, ::cuda::std::bit_and<T>,
                                       ::cuda::std::bit_or<>,  ::cuda::std::bit_or<T>,
                                       ::cuda::std::bit_xor<>, ::cuda::std::bit_xor<T>>;),
      NV_ANY_TARGET,
        (return false;)
    );
    // clang-format on
  }
  return false; // nvcc 11.x warning workaround
}

template <typename Input, typename ReductionOp, typename AccumT, typename T = detail::random_access_range_elem_t<Input>>
inline constexpr bool enable_promotion_v =
  ::cuda::std::is_integral_v<T> && sizeof(T) <= 2 && is_predefined_operator_v<ReductionOp, T>;

/***********************************************************************************************************************
 * Internal Reduction Algorithms: Sequential, Binary, Ternary
 **********************************************************************************************************************/

template <typename AccumT, typename Input, typename ReductionOp>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE AccumT
ThreadReduceSequential(const Input& input, ReductionOp reduction_op)
{
  auto retval = static_cast<AccumT>(input[0]);
#  pragma unroll
  for (int i = 1; i < cub::detail::static_size_v<Input>; ++i)
  {
    retval = reduction_op(retval, input[i]);
  }
  return retval;
}

template <typename AccumT, typename Input, typename ReductionOp>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE AccumT
ThreadReduceBinaryTree(const Input& input, ReductionOp reduction_op)
{
  constexpr auto length = cub::detail::static_size_v<Input>;
  auto array            = cub::detail::to_array<AccumT>(input);
#  pragma unroll
  for (int i = 1; i < length; i *= 2)
  {
#  pragma unroll
    for (int j = 0; j + i < length; j += i * 2)
    {
      array[j] = reduction_op(array[j], array[j + i]);
    }
  }
  return array[0];
}

template <typename AccumT, typename Input, typename ReductionOp>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE AccumT
ThreadReduceTernaryTree(const Input& input, ReductionOp reduction_op)
{
  constexpr auto length = cub::detail::static_size_v<Input>;
  auto array            = cub::detail::to_array<AccumT>(input);
#  pragma unroll
  for (int i = 1; i < length; i *= 3)
  {
#  pragma unroll
    for (int j = 0; j + i < length; j += i * 3)
    {
      auto value = reduction_op(array[j], array[j + i]);
      array[j]   = (j + i * 2 < length) ? reduction_op(value, array[j + i * 2]) : value;
    }
  }
  return array[0];
}

/***********************************************************************************************************************
 * SIMD Reduction
 **********************************************************************************************************************/

template <typename Input, typename ReductionOp>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE auto ThreadReduceSimd(const Input& input, ReductionOp)
{
  using cub::detail::unsafe_bitcast;
  using T                       = ::cuda::std::remove_cvref_t<decltype(input[0])>;
  using SimdReduceOp            = cub::internal::cub_operator_to_simd_operator_t<ReductionOp, T>;
  using SimdType                = simd_type_t<ReductionOp, T>;
  constexpr auto length         = cub::detail::static_size_v<Input>;
  constexpr auto simd_ratio     = sizeof(SimdType) / sizeof(T);
  constexpr auto length_rounded = ::cuda::round_down(length, simd_ratio);
  using UnpackedType            = ::cuda::std::array<T, simd_ratio>;
  using SimdArray               = ::cuda::std::array<SimdType, length / simd_ratio>;
  static_assert(simd_ratio == 1 || simd_ratio == 2, "Only SIMD size <= 2 is supported");
  T local_array[length_rounded];
  UnrolledCopy<length_rounded>(input, local_array);
  auto simd_input         = unsafe_bitcast<SimdArray>(local_array);
  SimdType simd_reduction = cub::ThreadReduce(simd_input, SimdReduceOp{});
  auto unpacked_values    = unsafe_bitcast<UnpackedType>(simd_reduction);
  if constexpr (simd_ratio == 1)
  {
    return unpacked_values[0];
  }
  else // simd_ratio == 2
  {
    // Create a reversed copy of the SIMD reduction result and apply the SIMD operator.
    // This avoids redundant instructions for converting to and from 32-bit register size
    T unpacked_values_rev[] = {unpacked_values[1], unpacked_values[0]};
    auto simd_reduction_rev = unsafe_bitcast<SimdType>(unpacked_values_rev);
    SimdType result         = SimdReduceOp{}(simd_reduction, simd_reduction_rev);
    // repeat the same optimization for the last element
    if constexpr (length % simd_ratio == 1)
    {
      T tail[]       = {input[length - 1], T{}};
      auto tail_simd = unsafe_bitcast<SimdType>(tail);
      result         = SimdReduceOp{}(result, tail_simd);
    }
    return unsafe_bitcast<UnpackedType>(result)[0];
  }
  _CCCL_UNREACHABLE(); // nvcc 11.x warning workaround (never reached)
}

} // namespace internal

/***********************************************************************************************************************
 * Reduction Interface/Dispatch (public)
 **********************************************************************************************************************/

template <typename Input, typename ReductionOp, typename ValueT, typename AccumT>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE AccumT ThreadReduce(const Input& input, ReductionOp reduction_op)
{
  static_assert(detail::is_fixed_size_random_access_range_t<Input>::value,
                "Input must support the subscript operator[] and have a compile-time size");
  static_assert(cub::detail::has_binary_call_operator<ReductionOp, ValueT>::value,
                "ReductionOp must have the binary call operator: operator(ValueT, ValueT)");
  if constexpr (cub::detail::static_size_v<Input> == 1)
  {
    return static_cast<AccumT>(input[0]);
  }
  using cub::detail::is_one_of_v;
  using cub::internal::enable_promotion_v;
  using cub::internal::enable_simd_reduction;
  using cub::internal::enable_ternary_reduction;
  using PromT = ::cuda::std::_If<enable_promotion_v<Input, ReductionOp, AccumT>, int, AccumT>;
  if constexpr ((!cub::internal::is_predefined_operator_v<ReductionOp, ValueT>
                 && !is_one_of_v<ReductionOp, cub::internal::SimdMin<ValueT>, cub::internal::SimdMax<ValueT>>)
                || sizeof(ValueT) >= 8)
  {
    return cub::internal::ThreadReduceSequential<AccumT>(input, reduction_op);
  }
  else if constexpr (is_one_of_v<ReductionOp, ::cuda::std::plus<>, ::cuda::std::plus<ValueT>>
                     && is_one_of_v<ValueT, int, uint32_t>)
  {
    NV_IF_TARGET(NV_PROVIDES_SM_90, //
                 (return cub::internal::ThreadReduceSequential<AccumT>(input, reduction_op);),
                 (return cub::internal::ThreadReduceTernaryTree<PromT>(input, reduction_op);));
  }
  else if constexpr (enable_simd_reduction<Input, ReductionOp, AccumT>())
  {
    return cub::internal::ThreadReduceSimd(input, reduction_op);
  }
  else if constexpr (enable_ternary_reduction<Input, ReductionOp, PromT>())
  {
    return cub::internal::ThreadReduceTernaryTree<PromT>(input, reduction_op);
  }
  else
  {
    return cub::internal::ThreadReduceBinaryTree<PromT>(input, reduction_op);
  }
}

//! @brief Reduction over statically-sized array-like types, seeded with the specified @p prefix.
//!
//! @tparam Input
//!   <b>[inferred]</b> The data type to be reduced having member
//!   <tt>operator[](int i)</tt> and must be statically-sized (size() method or static array)
//!
//! @tparam ReductionOp
//!   <b>[inferred]</b> Binary reduction operator type having member
//!   <tt>T operator()(const T &a, const T &b)</tt>
//!
//! @tparam PrefixT
//!   <b>[inferred]</b> The prefix type
//!
//! @param[in] input
//!   Input array
//!
//! @param[in] reduction_op
//!   Binary reduction operator
//!
//! @param[in] prefix
//!   Prefix to seed reduction with
//!
//! @return Aggregate of type <tt>cuda::std::__accumulator_t<ReductionOp, ValueT, PrefixT></tt>
//!
template <typename Input,
          typename ReductionOp,
          typename PrefixT,
          typename ValueT = ::cuda::std::remove_cvref_t<decltype(::cuda::std::declval<Input>()[0])>,
          typename AccumT = ::cuda::std::__accumulator_t<ReductionOp, ValueT, PrefixT>>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE AccumT
ThreadReduce(const Input& input, ReductionOp reduction_op, PrefixT prefix)
{
  static_assert(detail::is_fixed_size_random_access_range_t<Input>::value,
                "Input must support the subscript operator[] and have a compile-time size");
  static_assert(detail::has_binary_call_operator<ReductionOp, ValueT>::value,
                "ReductionOp must have the binary call operator: operator(ValueT, ValueT)");
  constexpr int length = cub::detail::static_size_v<Input>;
  // copy to a temporary array of type AccumT
  AccumT array[length + 1];
  array[0] = prefix;
#  pragma unroll
  for (int i = 0; i < length; ++i)
  {
    array[i + 1] = input[i];
  }
  return cub::ThreadReduce<decltype(array), ReductionOp, AccumT, AccumT>(array, reduction_op);
}

#endif // !_CCCL_DOXYGEN_INVOKED

CUB_NAMESPACE_END
