//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___SIMD_PERMUTE_MEMORY_H
#define _CUDA_STD___SIMD_PERMUTE_MEMORY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__utility/in_range.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__fwd/simd.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__ranges/access.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__ranges/data.h>
#include <cuda/std/__ranges/size.h>
#include <cuda/std/__simd/abi.h>
#include <cuda/std/__simd/basic_mask.h>
#include <cuda/std/__simd/basic_vec.h>
#include <cuda/std/__simd/concepts.h>
#include <cuda/std/__simd/exposition.h>
#include <cuda/std/__simd/flag.h>
#include <cuda/std/__simd/utility.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/cmp.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_SIMD

// [simd.permute.memory] gather

//----------------------------------------------------------------------------------------------------------------------
// gather helpers

template <typename _Range, typename _Ip, typename _IAbi>
using __default_gather_vec_t = vec<ranges::range_value_t<_Range>, __simd_size_v<_Ip, _IAbi>>;

template <typename _Vp, typename _Range, typename _Ip, typename _IAbi>
using __gather_result_t = conditional_t<is_same_v<_Vp, void>, __default_gather_vec_t<_Range, _Ip, _IAbi>, _Vp>;

template <typename>
inline constexpr bool __is_basic_vec_v = false;

template <typename _Tp, typename _Abi>
inline constexpr bool __is_basic_vec_v<basic_vec<_Tp, _Abi>> = __is_vectorizable_v<_Tp> && __is_enabled_abi_v<_Abi>;

//----------------------------------------------------------------------------------------------------------------------
// gather constraints concept

template <typename _Vp, typename _Range, typename _Ip, typename _IAbi>
_CCCL_CONCEPT __gather_constraints =
  ranges::contiguous_range<_Range> && ranges::sized_range<_Range> && is_integral_v<_Ip>
  && __simd_vec_type<__gather_result_t<_Vp, _Range, _Ip, _IAbi>> && __is_vectorizable_v<ranges::range_value_t<_Range>>
  && __explicitly_convertible_to<ranges::range_value_t<_Range>,
                                 typename __gather_result_t<_Vp, _Range, _Ip, _IAbi>::value_type>;

//----------------------------------------------------------------------------------------------------------------------
// gather generator

template <typename _Vp, typename _Ptr, typename _Ip, typename _IAbi, typename _Mp>
struct __gather_generator
{
  using __value_type = typename _Vp::value_type;

  const _Ptr __data_;
  const __simd_size_type __size_;
  const basic_vec<_Ip, _IAbi>& __indices_;
  const _Mp& __mask_;

  template <__simd_size_type _Idx>
  [[nodiscard]] _CCCL_API constexpr __value_type operator()(__simd_size_constant<_Idx>) const noexcept
  {
    if (!__mask_[_Idx])
    {
      return __value_type{};
    }
    const auto __raw_idx = __indices_[_Idx];
    if (::cuda::std::cmp_greater_equal(__raw_idx, 0) && ::cuda::std::cmp_less(__raw_idx, __size_))
    {
      const auto __idx = static_cast<__simd_size_type>(__raw_idx);
      return static_cast<__value_type>(__data_[__idx]);
    }
    return __value_type{};
  }
};

template <typename _Result, typename _Range, typename _Ip, typename _IAbi, typename... _Flags>
_CCCL_API _CCCL_CONSTEVAL void __check_gather_mandates() noexcept
{
  // same_as<remove_cvref_t<V>, V> is true (checked first so that later accesses to _Result's members are well-formed)
  static_assert(is_same_v<remove_cvref_t<_Result>, _Result>,
                "cuda::std::simd::partial_gather_from / unchecked_gather_from: V must not be cv- or ref-qualified");
  // V is an enabled specialization of basic_vec
  static_assert(__is_basic_vec_v<_Result>,
                "cuda::std::simd::partial_gather_from / unchecked_gather_from: V must be a specialization of "
                "basic_vec");
  // ranges​::​range_value_t<R> is a vectorizable type
  static_assert(__is_vectorizable_v<ranges::range_value_t<_Range>>,
                "cuda::std::simd::partial_gather_from / unchecked_gather_from: range_value_t<R> must be vectorizable");
  // V​::​size() == I​::​size() is true
  static_assert(_Result::__size == __simd_size_v<_Ip, _IAbi>,
                "cuda::std::simd::partial_gather_from / unchecked_gather_from: V::size() must equal indices.size()");
  // if the template parameter pack Flags does not contain convert-flag, then the conversion from
  // ranges​::​range_value_t<R> to T is value-preserving
  static_assert(__has_convert_flag_v<_Flags...>
                  || __is_value_preserving_v<ranges::range_value_t<_Range>, typename _Result::value_type>,
                "cuda::std::simd::partial_gather_from / unchecked_gather_from: conversion from range_value_t<R> to "
                "V::value_type is not value-preserving; use flag_convert");
}

//----------------------------------------------------------------------------------------------------------------------
// [simd.permute.memory] partial_gather_from

// masked
_CCCL_TEMPLATE(typename _Vp = void, typename _Range, typename _Ip, typename _IAbi, typename... _Flags)
_CCCL_REQUIRES(__gather_constraints<_Vp, _Range, _Ip, _IAbi>)
[[nodiscard]] _CCCL_API constexpr __gather_result_t<_Vp, _Range, _Ip, _IAbi> partial_gather_from(
  _Range&& __in,
  const typename basic_vec<_Ip, _IAbi>::mask_type& __mask,
  const basic_vec<_Ip, _IAbi>& __indices,
  flags<_Flags...> = {})
{
  using _Result = __gather_result_t<_Vp, _Range, _Ip, _IAbi>;
  ::cuda::std::simd::__check_gather_mandates<_Result, _Range, _Ip, _IAbi, _Flags...>();
  _CCCL_ASSERT(::cuda::std::in_range<__simd_size_type>(::cuda::std::ranges::size(__in)),
               "cuda::std::simd::partial_gather_from: ranges::size(in) is not representable as __simd_size_type");

  const auto __data = ::cuda::std::ranges::data(__in);
  const auto __size = static_cast<__simd_size_type>(::cuda::std::ranges::size(__in));
  _CCCL_ASSERT(__size == 0 || __data != nullptr,
               "cuda::std::simd::partial_gather_from: ranges::data(in) is null but ranges::size(in) > 0");
  ::cuda::std::simd::__assert_load_store_alignment<_Result, ranges::range_value_t<_Range>, _Flags...>(__data);
  using __mask_t      = typename basic_vec<_Ip, _IAbi>::mask_type;
  using __generator_t = __gather_generator<_Result, decltype(__data), _Ip, _IAbi, __mask_t>;
  return _Result{__generator_t{__data, __size, __indices, __mask}};
}

// unmasked: delegate to the masked overload with an all-true mask.
_CCCL_TEMPLATE(typename _Vp = void, typename _Range, typename _Ip, typename _IAbi, typename... _Flags)
_CCCL_REQUIRES(__gather_constraints<_Vp, _Range, _Ip, _IAbi>)
[[nodiscard]] _CCCL_API constexpr __gather_result_t<_Vp, _Range, _Ip, _IAbi>
partial_gather_from(_Range&& __in, const basic_vec<_Ip, _IAbi>& __indices, flags<_Flags...> __f = {})
{
  using __mask_t = typename basic_vec<_Ip, _IAbi>::mask_type;
  constexpr __mask_t __all_true{true};
  return ::cuda::std::simd::partial_gather_from<_Vp>(static_cast<_Range&&>(__in), __all_true, __indices, __f);
}

//----------------------------------------------------------------------------------------------------------------------
// [simd.permute.memory] unchecked_gather_from

// masked
_CCCL_TEMPLATE(typename _Vp = void, typename _Range, typename _Ip, typename _IAbi, typename... _Flags)
_CCCL_REQUIRES(__gather_constraints<_Vp, _Range, _Ip, _IAbi>)
[[nodiscard]] _CCCL_API constexpr __gather_result_t<_Vp, _Range, _Ip, _IAbi> unchecked_gather_from(
  _Range&& __in,
  const typename basic_vec<_Ip, _IAbi>::mask_type& __mask,
  const basic_vec<_Ip, _IAbi>& __indices,
  flags<_Flags...> __f = {})
{
  _CCCL_IF_NOT_CONSTEVAL_DEFAULT
  {
    _CCCL_ASSERT(::cuda::std::in_range<__simd_size_type>(::cuda::std::ranges::size(__in)),
                 "cuda::std::simd::unchecked_gather_from: ranges::size(in) is not representable as __simd_size_type");
    const auto __range_size = static_cast<__simd_size_type>(::cuda::std::ranges::size(__in));
    _CCCL_PRAGMA_UNROLL_FULL()
    for (__simd_size_type __i = 0; __i < __simd_size_v<_Ip, _IAbi>; ++__i)
    {
      if (__mask[__i])
      {
        const auto __idx = static_cast<__simd_size_type>(__indices[__i]);
        _CCCL_ASSERT(::cuda::std::in_range<__simd_size_type>(__indices[__i])
                       && ::cuda::in_range(__idx, __simd_size_type{0}, __range_size),
                     "cuda::std::simd::unchecked_gather_from: indices[i] must be in [0, ranges::size(in)) for every "
                     "selected i");
      }
    }
  }
  return ::cuda::std::simd::partial_gather_from<_Vp>(static_cast<_Range&&>(__in), __mask, __indices, __f);
}

// unmasked: delegate to the masked overload with an all-true mask to avoid duplicating the precondition check.
_CCCL_TEMPLATE(typename _Vp = void, typename _Range, typename _Ip, typename _IAbi, typename... _Flags)
_CCCL_REQUIRES(__gather_constraints<_Vp, _Range, _Ip, _IAbi>)
[[nodiscard]] _CCCL_API constexpr __gather_result_t<_Vp, _Range, _Ip, _IAbi>
unchecked_gather_from(_Range&& __in, const basic_vec<_Ip, _IAbi>& __indices, flags<_Flags...> __f = {})
{
  using __mask_t = typename basic_vec<_Ip, _IAbi>::mask_type;
  constexpr __mask_t __all_true{true};
  return ::cuda::std::simd::unchecked_gather_from<_Vp>(static_cast<_Range&&>(__in), __all_true, __indices, __f);
}

//----------------------------------------------------------------------------------------------------------------------
// [simd.permute.memory] scatter

// scatter constraints concept

template <typename _Tp, typename _Abi, typename _Range, typename _Ip, typename _IAbi>
_CCCL_CONCEPT __scatter_constraints =
  __simd_vec_type<basic_vec<_Tp, _Abi>> && ranges::contiguous_range<_Range> && ranges::sized_range<_Range>
  && is_integral_v<_Ip> && (__simd_size_v<_Tp, _Abi> == __simd_size_v<_Ip, _IAbi>)
  && __is_vectorizable_v<ranges::range_value_t<_Range>> && indirectly_writable<ranges::iterator_t<_Range>, _Tp>
  && __explicitly_convertible_to<_Tp, ranges::range_value_t<_Range>>;

//----------------------------------------------------------------------------------------------------------------------
// scatter mandates

template <typename _Tp, typename _Range, typename... _Flags>
_CCCL_API _CCCL_CONSTEVAL void __check_scatter_mandates() noexcept
{
  static_assert(__is_vectorizable_v<ranges::range_value_t<_Range>>,
                "cuda::std::simd::partial_scatter_to / unchecked_scatter_to: range_value_t<R> must be vectorizable");
  static_assert(__has_convert_flag_v<_Flags...> || __is_value_preserving_v<_Tp, ranges::range_value_t<_Range>>,
                "cuda::std::simd::partial_scatter_to / unchecked_scatter_to: conversion from V::value_type to "
                "range_value_t<R> is not value-preserving; use flag_convert");
}

//----------------------------------------------------------------------------------------------------------------------
// [simd.permute.memory] partial_scatter_to

// masked
_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _Range, typename _Ip, typename _IAbi, typename... _Flags)
_CCCL_REQUIRES(__scatter_constraints<_Tp, _Abi, _Range, _Ip, _IAbi>)
_CCCL_API constexpr void partial_scatter_to(
  const basic_vec<_Tp, _Abi>& __v,
  _Range&& __out,
  const typename basic_vec<_Ip, _IAbi>::mask_type& __mask,
  const basic_vec<_Ip, _IAbi>& __indices,
  flags<_Flags...> = {})
{
  using __vec_t = basic_vec<_Tp, _Abi>;
  ::cuda::std::simd::__check_scatter_mandates<_Tp, _Range, _Flags...>();
  _CCCL_ASSERT(::cuda::std::in_range<__simd_size_type>(::cuda::std::ranges::size(__out)),
               "cuda::std::simd::partial_scatter_to: ranges::size(out) is not representable as __simd_size_type");

  const auto __data     = ::cuda::std::ranges::data(__out);
  const auto __out_size = static_cast<__simd_size_type>(::cuda::std::ranges::size(__out));
  _CCCL_ASSERT(__out_size == 0 || __data != nullptr,
               "cuda::std::simd::partial_scatter_to: ranges::data(out) is null but ranges::size(out) > 0");
  ::cuda::std::simd::__assert_load_store_alignment<__vec_t, ranges::range_value_t<_Range>, _Flags...>(__data);

  _CCCL_PRAGMA_UNROLL_FULL()
  for (__simd_size_type __i = 0; __i < __vec_t::__size; ++__i)
  {
    if (!__mask[__i])
    {
      continue;
    }
    const auto __raw_idx = __indices[__i];
    if (::cuda::std::cmp_greater_equal(__raw_idx, 0) && ::cuda::std::cmp_less(__raw_idx, __out_size))
    {
      const auto __idx = static_cast<__simd_size_type>(__raw_idx);
      __data[__idx]    = static_cast<ranges::range_value_t<_Range>>(__v[__i]);
    }
  }
}

// unmasked: delegate to the masked overload with an all-true mask.
_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _Range, typename _Ip, typename _IAbi, typename... _Flags)
_CCCL_REQUIRES(__scatter_constraints<_Tp, _Abi, _Range, _Ip, _IAbi>)
_CCCL_API constexpr void partial_scatter_to(
  const basic_vec<_Tp, _Abi>& __v, _Range&& __out, const basic_vec<_Ip, _IAbi>& __indices, flags<_Flags...> __f = {})
{
  using __mask_t = typename basic_vec<_Ip, _IAbi>::mask_type;
  constexpr __mask_t __all_true{true};
  ::cuda::std::simd::partial_scatter_to(__v, static_cast<_Range&&>(__out), __all_true, __indices, __f);
}

//----------------------------------------------------------------------------------------------------------------------
// [simd.permute.memory] unchecked_scatter_to

// masked
_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _Range, typename _Ip, typename _IAbi, typename... _Flags)
_CCCL_REQUIRES(__scatter_constraints<_Tp, _Abi, _Range, _Ip, _IAbi>)
_CCCL_API constexpr void unchecked_scatter_to(
  const basic_vec<_Tp, _Abi>& __v,
  _Range&& __out,
  const typename basic_vec<_Ip, _IAbi>::mask_type& __mask,
  const basic_vec<_Ip, _IAbi>& __indices,
  flags<_Flags...> __f = {})
{
  _CCCL_IF_NOT_CONSTEVAL_DEFAULT
  {
    _CCCL_ASSERT(::cuda::std::in_range<__simd_size_type>(::cuda::std::ranges::size(__out)),
                 "cuda::std::simd::unchecked_scatter_to: ranges::size(out) is not representable as __simd_size_type");
    const auto __range_size = static_cast<__simd_size_type>(::cuda::std::ranges::size(__out));
    _CCCL_PRAGMA_UNROLL_FULL()
    for (__simd_size_type __i = 0; __i < basic_vec<_Tp, _Abi>::__size; ++__i)
    {
      if (__mask[__i])
      {
        const auto __idx = static_cast<__simd_size_type>(__indices[__i]);
        _CCCL_ASSERT(::cuda::std::in_range<__simd_size_type>(__indices[__i])
                       && ::cuda::in_range(__idx, __simd_size_type{0}, __range_size),
                     "cuda::std::simd::unchecked_scatter_to: indices[i] must be in [0, ranges::size(out)) for every "
                     "selected i");
      }
    }
  }
  ::cuda::std::simd::partial_scatter_to(__v, static_cast<_Range&&>(__out), __mask, __indices, __f);
}

// unmasked: delegate to the masked overload with an all-true mask to avoid duplicating the precondition check.
_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _Range, typename _Ip, typename _IAbi, typename... _Flags)
_CCCL_REQUIRES(__scatter_constraints<_Tp, _Abi, _Range, _Ip, _IAbi>)
_CCCL_API constexpr void unchecked_scatter_to(
  const basic_vec<_Tp, _Abi>& __v, _Range&& __out, const basic_vec<_Ip, _IAbi>& __indices, flags<_Flags...> __f = {})
{
  using __mask_t = typename basic_vec<_Ip, _IAbi>::mask_type;
  constexpr __mask_t __all_true{true};
  ::cuda::std::simd::unchecked_scatter_to(__v, static_cast<_Range&&>(__out), __all_true, __indices, __f);
}

_CCCL_END_NAMESPACE_CUDA_STD_SIMD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___SIMD_PERMUTE_MEMORY_H
