//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___SIMD_UTILITY_H
#define _CUDA_STD___SIMD_UTILITY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__memory/is_aligned.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__fwd/span.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__simd/abi.h>
#include <cuda/std/__simd/concepts.h>
#include <cuda/std/__simd/flag.h>
#include <cuda/std/__simd/specializations/fixed_size_vec.h>
#include <cuda/std/__simd/type_traits.h>
#include <cuda/std/__tuple_dir/tuple_size.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/integer_sequence.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_SIMD

template <typename _Abi>
inline constexpr bool __is_enabled_abi_v = false;

// c++ specification sets 1 <= N <= 64
template <__simd_size_type _Np>
inline constexpr bool __is_enabled_abi_v<__fixed_size<_Np>> = (_Np >= 1 && _Np <= 64);

//----------------------------------------------------------------------------------------------------------------------
// __can_generate_v

template <typename _Tp, typename _Generator, __simd_size_type _Idx, typename = void>
inline constexpr bool __is_well_formed = false;

template <typename _Tp, typename _Generator, __simd_size_type _Idx>
inline constexpr bool
  __is_well_formed<_Tp,
                   _Generator,
                   _Idx,
                   void_t<decltype(declval<_Generator>()(integral_constant<__simd_size_type, _Idx>()))>> =
    is_convertible_v<decltype(declval<_Generator>()(integral_constant<__simd_size_type, _Idx>())), _Tp>;

template <typename _Tp, typename _Generator, __simd_size_type... _Indices>
[[nodiscard]]
_CCCL_API _CCCL_CONSTEVAL bool __can_generate(integer_sequence<__simd_size_type, _Indices...>) noexcept
{
  return (true && ... && __is_well_formed<_Tp, _Generator, _Indices>);
}

template <typename _Tp, typename _Generator, __simd_size_type _Size>
inline constexpr bool __can_generate_v =
  __can_generate<_Tp, _Generator>(make_integer_sequence<__simd_size_type, _Size>());

//----------------------------------------------------------------------------------------------------------------------
// __is_compatible_range_v

template <typename _Range, typename = void>
inline constexpr bool __has_tuple_size_v = false;

template <typename _Range>
inline constexpr bool __has_tuple_size_v<_Range, void_t<decltype(tuple_size<remove_cvref_t<_Range>>::value)>> = true;

template <typename _Range, typename = void>
inline constexpr bool __has_static_extent_v = false;

template <typename _Range>
inline constexpr bool __has_static_extent_v<_Range, void_t<decltype(remove_cvref_t<_Range>::extent)>> =
  remove_cvref_t<_Range>::extent != dynamic_extent;

// Proxy for ranges::size(r) is a constant expression.
template <typename _Range>
_CCCL_CONCEPT __has_static_size = __has_tuple_size_v<_Range> || __has_static_extent_v<_Range>;

template <typename _Range>
[[nodiscard]] _CCCL_API _CCCL_CONSTEVAL __simd_size_type __get_static_range_size() noexcept
{
  using __range_t = remove_cvref_t<_Range>;
  if constexpr (__has_tuple_size_v<_Range>)
  {
    return __simd_size_type{tuple_size_v<__range_t>};
  }
  else
  {
    return __simd_size_type{__range_t::extent};
  }
}

template <typename _Range>
inline constexpr __simd_size_type __static_range_size_v = __get_static_range_size<_Range>();

// This trait is defined at namespace scope (not as a static member of basic_vec) because GCC 13 rejects partial
// specialization of static member variable templates. The static-size detection intentionally avoids directly using
// tuple_size_v<T> in the guard because that causes a hard error (instead of SFINAE) on NVCC with
// clang-19/clang-14/nvc++ when T is an incomplete specialization of tuple_size.
template <typename _Range>
inline constexpr bool __is_compatible_range_guard_v =
  __has_static_size<_Range> && ranges::contiguous_range<_Range> && ranges::sized_range<_Range>;

template <typename _Tp, __simd_size_type _Size, typename _Range, bool = __is_compatible_range_guard_v<_Range>>
inline constexpr bool __is_compatible_range_v = false;

template <typename _Tp, __simd_size_type _Size, typename _Range>
inline constexpr bool __is_compatible_range_v<_Tp, _Size, _Range, true> =
  (__static_range_size_v<_Range> == _Size) //
  && __is_vectorizable_v<ranges::range_value_t<_Range>> //
  && __explicitly_convertible_to<ranges::range_value_t<_Range>, _Tp>;

//----------------------------------------------------------------------------------------------------------------------
// [simd.flags] alignment assertion for load/store pointers

template <typename _Vec, typename _Up, typename... _Flags>
_CCCL_API constexpr void __assert_load_store_alignment([[maybe_unused]] const _Up* __data) noexcept
{
  _CCCL_IF_NOT_CONSTEVAL_DEFAULT
  {
    if constexpr (__has_aligned_flag_v<_Flags...>)
    {
      _CCCL_ASSERT(::cuda::is_aligned(__data, alignment_v<_Vec, _Up>),
                   "flag_aligned requires data to be aligned to alignment_v<V, range_value_t<R>>");
    }
    else if constexpr (__has_overaligned_flag_v<_Flags...>)
    {
      _CCCL_ASSERT(::cuda::is_aligned(__data, __overaligned_alignment_v<_Flags...>),
                   "flag_overaligned<N> requires data to be aligned to N");
    }
  }
}

_CCCL_END_NAMESPACE_CUDA_STD_SIMD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___SIMD_UTILITY_H
