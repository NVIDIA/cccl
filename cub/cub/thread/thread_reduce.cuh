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

#include <cuda/std/array> // array
#include <cuda/std/bit> // bit_cast
#include <cuda/std/cassert> // assert
#include <cuda/std/cstdint> // uint16_t

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
//!   raw arrays, ``std::array``, ``std::span``,  ``std::mdspan`` (C++23),
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
//!              typename ValueT = ::cuda::std::__remove_cvref_t<decltype(::cuda::std::declval<Input>()[0])>,
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
//!   * Sum (``cub::Sum``) and Multiplication (``cub::Mul``) on SM70+ for ``__half`` data type
//!   * Sum (``cub::Sum``) and Multiplication (``cub::Mul``) on SM80+ for ``__nv_bfloat16`` data type
//!   * Minimum (``cub::Min``) and Maximum (``cub::Max``) on SM80+ for ``__half`` and ``__nv_bfloat16`` data types
//!   * Minimum (``cub::Min``) and Maximum (``cub::Max``) on SM90+ for ``int16_t`` and ``uint16_t`` data types
//!     (Hopper DPX instructions)
//!
//! - Instruction-Level Parallelism (ILP) by exploiting a ternary tree reduction for
//!
//!   * Sum (``cub::Sum``), Bitwise AND (``cub::BitAnd``), OR (``cub::BitOr``), XOR (``cub::BitXor``) on SM50+ for
//!     integer data types
//!   * Minimum (``cub::Min``) and Maximum (``cub::Max``) on SM80+ for integer data types (Hopper DPX instructions),
//!     ``__half2``, ``__nv_bfloat162``, ``__half`` (after vectorization), and ``__nv_bfloat16`` (after vectorization)
//!     data types
//!   * Minimum (``cub::Min``) and Maximum (``cub::Max``) on SM90+ for integer data types (Hopper DPX instructions)
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
//!        int sum      = cub::ThreadReduce(array, cub::Sum()); // sum = 10
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
#ifndef DOXYGEN_SHOULD_SKIP_THIS // Do not document
          typename ValueT = ::cuda::std::__remove_cvref_t<decltype(::cuda::std::declval<Input>()[0])>,
#else
          typename ValueT = random_access_value_t<Input>,
#endif // !DOXYGEN_SHOULD_SKIP_THIS
          typename AccumT = ::cuda::std::__accumulator_t<ReductionOp, ValueT>>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE AccumT ThreadReduce(const Input& input, ReductionOp reduction_op);
// forward declaration

/***********************************************************************************************************************
 * Internal Reduction Implementations
 **********************************************************************************************************************/

#ifndef DOXYGEN_SHOULD_SKIP_THIS // Do not document

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
/// see DPX documetation https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#dpx
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

template <typename Input, typename ReductionOp>
_CCCL_NODISCARD _CCCL_DEVICE constexpr bool enable_generic_simd_reduction()
{
  using cub::detail::is_one_of;
  using T      = ::cuda::std::__remove_cvref_t<decltype(::cuda::std::declval<Input>()[0])>;
  using Length = ::cuda::std::integral_constant<int, cub::detail::static_size_v<Input>()>;
  // clang-format off
  return ((is_one_of<T, ::cuda::std::int16_t, ::cuda::std::uint16_t>() && is_one_of<ReductionOp, cub::Min, cub::Max>())
#  if defined(_CCCL_HAS_NVFP16)
     || (::cuda::std::is_same<T, __half>::value && is_one_of<ReductionOp, cub::Min, cub::Max, cub::Sum, cub::Mul>())
#  endif
#  if defined(_CCCL_HAS_NVBF16)
     || (::cuda::std::is_same<T, __nv_bfloat16>::value &&
         is_one_of<ReductionOp, cub::Min, cub::Max, cub::Sum, cub::Mul>())
#  endif
   ) && Length{} >= 4;
  // clang-format on
}

template <typename T, typename ReductionOp, int Length>
_CCCL_NODISCARD _CCCL_DEVICE constexpr bool enable_sm90_simd_reduction()
{
  using cub::detail::is_one_of;
  // cub::Sum not handled: IADD3 always produces less instructions than VIADD2
  return is_one_of<T, ::cuda::std::int16_t, ::cuda::std::uint16_t>() && //
         is_one_of<ReductionOp, cub::Min, cub::Max>() && Length >= 10;
}

template <typename T, typename ReductionOp, int Length>
_CCCL_NODISCARD _CCCL_DEVICE constexpr bool enable_sm80_simd_reduction()
{
  using cub::detail::is_one_of;
  using ::cuda::std::is_same;
  return is_one_of<ReductionOp, cub::Min, cub::Max, cub::Sum, cub::Mul>() && Length >= 4
#  if defined(_CCCL_HAS_NVFP16) && defined(_CCCL_HAS_NVBF16)
      && (is_same<T, __half>::value || is_same<T, __nv_bfloat16>::value)
#  elif defined(_CCCL_HAS_NVFP16)
      && is_same<T, __half>::value
#  elif defined(_CCCL_HAS_NVBF16)
      && is_same<T, __nv_bfloat16>::value
#  endif
    ;
}

template <typename T, typename ReductionOp, int Length>
_CCCL_NODISCARD _CCCL_DEVICE constexpr bool enable_sm70_simd_reduction()
{
  using cub::detail::is_one_of;
  using ::cuda::std::is_same;
#  if defined(_CCCL_HAS_NVFP16)
  return is_same<T, __half>::value && is_one_of<ReductionOp, cub::Sum, cub::Mul>() && Length >= 4;
#  else
  return false;
#  endif
}

template <typename Input, typename ReductionOp, typename AccumT>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE _CCCL_CONSTEXPR_CXX14 bool enable_simd_reduction()
{
  using cub::detail::is_one_of;
  using ::cuda::std::is_same;
  using T = ::cuda::std::__remove_cvref_t<decltype(::cuda::std::declval<Input>()[0])>;
  if _CCCL_CONSTEXPR_CXX17 (!is_same<T, AccumT>::value)
  {
    return false;
  }
  else
  {
    constexpr auto length = cub::detail::static_size_v<Input>();
    // clang-format off
    _NV_TARGET_DISPATCH(
      NV_PROVIDES_SM_90,
        (return enable_sm90_simd_reduction<T, ReductionOp, length>() ||
                enable_sm80_simd_reduction<T, ReductionOp, length>() ||
                enable_sm70_simd_reduction<T, ReductionOp, length>();),
      NV_PROVIDES_SM_80,
        (return enable_sm80_simd_reduction<T, ReductionOp, length>() ||
                enable_sm70_simd_reduction<T, ReductionOp, length>();),
      NV_PROVIDES_SM_70,
        (return enable_sm70_simd_reduction<T, ReductionOp, length>();),
      NV_IS_DEVICE,
        (static_cast<void>(length); // maybe unused
         return false;)
    );
    // clang-format on
    return false;
  }
  return false; // nvcc 11.x warning workaround
}

template <typename Input, typename ReductionOp, typename AccumT>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE _CCCL_CONSTEXPR_CXX14 bool enable_ternary_reduction()
{
  using cub::detail::is_one_of;
  using ::cuda::std::is_same;
  using T               = ::cuda::std::__remove_cvref_t<decltype(::cuda::std::declval<Input>()[0])>;
  constexpr auto length = cub::detail::static_size_v<Input>();
  if _CCCL_CONSTEXPR_CXX17 (length < 6)
  {
    return false;
  }
  else
  {
    // clang-format off
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_90,
        (return (is_one_of<T, ::cuda::std::int32_t, ::cuda::std::uint32_t, ::cuda::std::int64_t, ::cuda::std::uint64_t>
                 && is_one_of<ReductionOp, cub::Min, cub::Max, cub::Sum, cub::BitAnd, cub::BitOr, cub::BitXor>())
#if defined(_CCCL_HAS_NVFP16)
               || (is_same<T, __half2>::value &&
                   is_one_of<ReductionOp, cub::Min, cub::Max, SimdMin<__half>, SimdMax<__half>>())
#endif
#if defined(_CCCL_HAS_NVBF16)
               || (is_same<T, __nv_bfloat162>::value &&
                   is_one_of<ReductionOp, cub::Min, cub::Max, SimdMin<__nv_bfloat16>, SimdMax<__nv_bfloat16>>())
#endif
         ;),
      NV_PROVIDES_SM_50,
        (return is_one_of<AccumT, ::cuda::std::int32_t, ::cuda::std::uint32_t, ::cuda::std::int64_t,
                                  ::cuda::std::uint64_t>()
             && is_one_of<ReductionOp, cub::Sum, cub::BitAnd, cub::BitOr, cub::BitXor>();),
      NV_ANY_TARGET,
        (return false;)
    );
    // clang-format on
  }
  return false; // nvcc 11.x warning workaround
}

template <typename Input, typename ReductionOp, typename AccumT>
_CCCL_NODISCARD _CCCL_DEVICE constexpr bool enable_promotion()
{
  using cub::detail::is_one_of;
  using ::cuda::std::is_same;
  using T = ::cuda::std::__remove_cvref_t<decltype(::cuda::std::declval<Input>()[0])>;
  return ::cuda::std::is_integral<T>::value && sizeof(T) <= 2
      && is_one_of<ReductionOp, cub::Sum, cub::Mul, cub::BitAnd, cub::BitOr, cub::BitXor, cub::Max, cub::Min>();
}

/***********************************************************************************************************************
 * Internal Reduction Algorithms: Sequential, Binary, Ternary
 **********************************************************************************************************************/

template <typename AccumT, typename Input, typename ReductionOp>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE AccumT
ThreadReduceSequential(const Input& input, ReductionOp reduction_op)
{
  AccumT retval = input[0];
#  pragma unroll
  for (int i = 1; i < cub::detail::static_size_v<Input>(); ++i)
  {
    retval = reduction_op(retval, input[i]);
  }
  return retval;
}

template <typename AccumT, typename Input, typename ReductionOp>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE AccumT
ThreadReduceBinaryTree(const Input& input, ReductionOp reduction_op)
{
  constexpr auto length = cub::detail::static_size_v<Input>();
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
  constexpr auto length = cub::detail::static_size_v<Input>();
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

// never reached. Protect instantion of ThreadReduceSimd with arbitrary types and operators
template <typename Input,
          typename ReductionOp,
          _CUB_TEMPLATE_REQUIRES(!cub::internal::enable_generic_simd_reduction<Input, ReductionOp>())>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE auto
ThreadReduceSimd(const Input& input, ReductionOp) -> ::cuda::std::__remove_cvref_t<decltype(input[0])>
{
  assert(false);
  return input[0];
}

template <typename Input,
          typename ReductionOp,
          _CUB_TEMPLATE_REQUIRES(cub::internal::enable_generic_simd_reduction<Input, ReductionOp>())>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE auto
ThreadReduceSimd(const Input& input, ReductionOp reduction_op) -> ::cuda::std::__remove_cvref_t<decltype(input[0])>
{
  using cub::detail::unsafe_bitcast;
  using T                       = ::cuda::std::__remove_cvref_t<decltype(input[0])>;
  using SimdReduceOp            = cub::internal::cub_operator_to_simd_operator_t<ReductionOp, T>;
  using SimdType                = simd_type_t<ReductionOp, T>;
  constexpr auto length         = cub::detail::static_size_v<Input>();
  constexpr auto simd_ratio     = sizeof(SimdType) / sizeof(T);
  constexpr auto length_rounded = (length / simd_ratio) * simd_ratio; // TODO: replace with round_up()
  using UnpackedType            = ::cuda::std::array<T, simd_ratio>;
  using SimdArray               = ::cuda::std::array<SimdType, length / simd_ratio>;
  static_assert(simd_ratio == 1 || simd_ratio == 2, "Only SIMD size <= 2 is supported");
  T local_array[length_rounded];
  UnrolledCopy<length_rounded>(input, local_array);
  auto simd_input         = unsafe_bitcast<SimdArray>(local_array);
  SimdType simd_reduction = cub::ThreadReduce(simd_input, SimdReduceOp{});
  auto unpacked_values    = unsafe_bitcast<UnpackedType>(simd_reduction);
  if _CCCL_CONSTEXPR_CXX17 (simd_ratio == 1)
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
    if _CCCL_CONSTEXPR_CXX17 (length % simd_ratio == 1)
    {
      T tail[]       = {input[length - 1], T{}};
      auto tail_simd = unsafe_bitcast<SimdType>(tail);
      result         = SimdReduceOp{}(result, tail_simd);
    }
    return unsafe_bitcast<UnpackedType>(result)[0];
  }
  return input[0]; // nvcc 11.x warning workaround (never reached)
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
  using cub::internal::enable_promotion;
  using cub::internal::enable_simd_reduction;
  using cub::internal::enable_ternary_reduction;
  using PromT = ::cuda::std::_If<enable_promotion<Input, ReductionOp, AccumT>(), int, AccumT>;
  if (enable_simd_reduction<Input, ReductionOp, AccumT>())
  {
    return cub::internal::ThreadReduceSimd(input, reduction_op);
  }
  else if (enable_ternary_reduction<Input, ReductionOp, PromT>())
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
#  ifndef DOXYGEN_SHOULD_SKIP_THIS // Do not document
          typename ValueT = ::cuda::std::__remove_cvref_t<decltype(::cuda::std::declval<Input>()[0])>,
#  endif // !DOXYGEN_SHOULD_SKIP_THIS
          typename AccumT = ::cuda::std::__accumulator_t<ReductionOp, ValueT, PrefixT>>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE AccumT
ThreadReduce(const Input& input, ReductionOp reduction_op, PrefixT prefix)
{
  static_assert(detail::is_fixed_size_random_access_range_t<Input>::value,
                "Input must support the subscript operator[] and have a compile-time size");
  static_assert(detail::has_binary_call_operator<ReductionOp, ValueT>::value,
                "ReductionOp must have the binary call operator: operator(ValueT, ValueT)");
  constexpr int length = cub::detail::static_size_v<Input>();
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

/***********************************************************************************************************************
 * Pointer Interfaces with explicit Length (internal use only)
 **********************************************************************************************************************/

/// Internal namespace (to prevent ADL mishaps between static functions when mixing different CUB installations)
namespace internal
{

/**
 * @remark The pointer interface adds little value and requires Length to be explicit.
 *         Prefer using the array-like interface
 *
 * @brief Perform a sequential reduction over @p length elements of the @p input pointer. The aggregate is returned.
 *
 * @tparam T
 *   <b>[inferred]</b> The data type to be reduced
 *
 * @tparam ReductionOp
 *   <b>[inferred]</b> Binary reduction operator type having member
 *   <tt>T operator()(const T &a, const T &b)</tt>
 *
 * @param[in] input
 *   Input pointer
 *
 * @param[in] reduction_op
 *   Binary reduction operator
 *
 * @return Aggregate of type <tt>cuda::std::__accumulator_t<ReductionOp, T></tt>
 */
template <int Length, typename T, typename ReductionOp, typename AccumT = ::cuda::std::__accumulator_t<ReductionOp, T>>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE AccumT ThreadReduce(const T* input, ReductionOp reduction_op)
{
  static_assert(Length > 0, "Length must be greater than 0");
  static_assert(cub::detail::has_binary_call_operator<ReductionOp, T>::value,
                "ReductionOp must have the binary call operator: operator(V1, V2)");
  using ArrayT = T[Length];
  auto array   = reinterpret_cast<const T(*)[Length]>(input);
  return cub::ThreadReduce(*array, reduction_op);
}

/**
 * @remark The pointer interface adds little value and requires Length to be explicit.
 *         Prefer using the array-like interface
 *
 * @brief Perform a sequential reduction over @p length elements of the @p input pointer, seeded with the specified @p
 *        prefix. The aggregate is returned.
 *
 * @tparam length
 *   Length of input pointer
 *
 * @tparam T
 *   <b>[inferred]</b> The data type to be reduced
 *
 * @tparam ReductionOp
 *   <b>[inferred]</b> Binary reduction operator type having member
 *   <tt>T operator()(const T &a, const T &b)</tt>
 *
 * @tparam PrefixT
 *   <b>[inferred]</b> The prefix type
 *
 * @param[in] input
 *   Input pointer
 *
 * @param[in] reduction_op
 *   Binary reduction operator
 *
 * @param[in] prefix
 *   Prefix to seed reduction with
 *
 * @return Aggregate of type <tt>cuda::std::__accumulator_t<ReductionOp, T, PrefixT></tt>
 */
template <int Length,
          typename T,
          typename ReductionOp,
          typename PrefixT,
          typename AccumT = ::cuda::std::__accumulator_t<ReductionOp, T, PrefixT>,
          _CUB_TEMPLATE_REQUIRES(Length > 0)>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE AccumT
ThreadReduce(const T* input, ReductionOp reduction_op, PrefixT prefix)
{
  static_assert(detail::has_binary_call_operator<ReductionOp, T>::value,
                "ReductionOp must have the binary call operator: operator(V1, V2)");
  auto array = reinterpret_cast<const T(*)[Length]>(input);
  return cub::ThreadReduce(*array, reduction_op, prefix);
}

template <int Length, typename T, typename ReductionOp, typename PrefixT, _CUB_TEMPLATE_REQUIRES(Length == 0)>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE T ThreadReduce(const T*, ReductionOp, PrefixT prefix)
{
  return prefix;
}

} // namespace internal

#endif // !DOXYGEN_SHOULD_SKIP_THIS

CUB_NAMESPACE_END
