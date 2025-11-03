//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___VARIANT_VARIANT_VISIT_H
#define _CUDA_STD___VARIANT_VARIANT_VISIT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/__variant/variant_access.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

namespace __variant_detail::__visitation
{
struct __variant
{
  // We need to guard against the final invocation where we have processed all variants
  template <size_t _Remaining, size_t _CurrentVariant, class... _Variants, enable_if_t<_Remaining == 0, int> = 0>
  [[nodiscard]] _CCCL_API static constexpr size_t __get_runtime_index(const _Variants&...) noexcept
  {
    return 0;
  }

  template <size_t _Remaining,
            size_t _CurrentVariant,
            class _Variant,
            class... _OtherVariants,
            enable_if_t<(_Remaining != 0) && (_CurrentVariant == 0), int> = 0>
  [[nodiscard]] _CCCL_API static constexpr size_t
  __get_runtime_index(const _Variant& __v, const _OtherVariants&...) noexcept
  {
    return __v.__impl_.index();
  }

  template <size_t _Remaining,
            size_t _CurrentVariant,
            class _Variant,
            class... _OtherVariants,
            enable_if_t<(_Remaining != 0) && (_CurrentVariant != 0), int> = 0>
  [[nodiscard]] _CCCL_API static constexpr size_t
  __get_runtime_index(const _Variant&, const _OtherVariants&... __vs) noexcept
  {
    return __get_runtime_index<_Remaining, _CurrentVariant - 1>(__vs...);
  }

  // Terminal function call with all indexes determined
  template <class _Visitor, class... _Vs, size_t... _ProcessedIndices>
  [[nodiscard]] _CCCL_API static constexpr decltype(auto) __visit_impl(
    index_sequence<_ProcessedIndices...>, integer_sequence<size_t>, const size_t, _Visitor&& __visitor, _Vs&&... __vs)
  {
    return ::cuda::std::__invoke(
      ::cuda::std::forward<_Visitor>(__visitor),
      __access::__base::__get_alt<_ProcessedIndices>(::cuda::std::forward<_Vs>(__vs).__impl_)...);
  }

  template <size_t _CurrentIndex,
            class _Visitor,
            class... _Vs,
            size_t... _ProcessedIndices,
            size_t... _UnprocessedIndices,
            enable_if_t<_CurrentIndex != 0, int> = 0>
  [[nodiscard]] _CCCL_API static constexpr decltype(auto) __visit_impl(
    index_sequence<_ProcessedIndices...>,
    index_sequence<_CurrentIndex, _UnprocessedIndices...>,
    const size_t __current_index,
    _Visitor&& __visitor,
    _Vs&&... __vs)
  {
    // We found the right index, move to the next variant
    if (__current_index == _CurrentIndex)
    {
      const size_t __next_index =
        __get_runtime_index<sizeof...(_UnprocessedIndices), sizeof...(_ProcessedIndices) + 1>(__vs...);
      return __visit_impl(
        index_sequence<_ProcessedIndices..., _CurrentIndex>{},
        index_sequence<_UnprocessedIndices...>{},
        __next_index,
        ::cuda::std::forward<_Visitor>(__visitor),
        ::cuda::std::forward<_Vs>(__vs)...);
    }

    return __visit_impl(
      index_sequence<_ProcessedIndices...>{},
      index_sequence<_CurrentIndex - 1, _UnprocessedIndices...>{},
      __current_index,
      ::cuda::std::forward<_Visitor>(__visitor),
      ::cuda::std::forward<_Vs>(__vs)...);
  }

  _CCCL_BEGIN_NV_DIAG_SUPPRESS(940) // Suppress no return at end of function
  // This overload is needed to tell the compiler that the recursion is indeed limited
  template <class _Visitor, class... _Vs, size_t... _ProcessedIndices, size_t... _UnprocessedIndices>
  [[nodiscard]] _CCCL_API static constexpr decltype(auto) __visit_impl(
    index_sequence<_ProcessedIndices...>,
    index_sequence<0, _UnprocessedIndices...>,
    const size_t __current_index,
    _Visitor&& __visitor,
    _Vs&&... __vs)
  {
    // We found the right index, move to the next variant
    if (__current_index == 0)
    {
      const size_t __next_index =
        __get_runtime_index<sizeof...(_UnprocessedIndices), sizeof...(_ProcessedIndices) + 1>(__vs...);
      return __visit_impl(
        index_sequence<_ProcessedIndices..., 0>{},
        index_sequence<_UnprocessedIndices...>{},
        __next_index,
        ::cuda::std::forward<_Visitor>(__visitor),
        ::cuda::std::forward<_Vs>(__vs)...);
    }
    _CCCL_UNREACHABLE();
  }
  _CCCL_END_NV_DIAG_SUPPRESS() // End suppression of no return at end of function

  template <class _Visitor, class... _Vs>
  [[nodiscard]] _CCCL_API static constexpr decltype(auto) __visit_value(_Visitor&& __visitor, _Vs&&... __vs)
  {
    // NOTE: We use a recursive implementation strategy here. That means we can omit the manual return type checks from
    // the common function pointer implementation, as the compiler will abort if the return types do not match.
    const size_t __first_index = __get_runtime_index<sizeof...(_Vs), 0>(__vs...);
    return __visit_impl(
      integer_sequence<size_t>{},
      index_sequence<(remove_cvref_t<_Vs>::__size() - 1)...>{},
      __first_index,
      __make_value_visitor(::cuda::std::forward<_Visitor>(__visitor)),
      ::cuda::std::forward<_Vs>(__vs)...);
  }

  template <class _Rp, class _Visitor, class... _Vs>
  [[nodiscard]] _CCCL_API static constexpr _Rp __visit_value(_Visitor&& __visitor, _Vs&&... __vs)
  {
    const size_t __first_index = __get_runtime_index<sizeof...(_Vs), 0>(__vs...);
    return __visit_impl(
      integer_sequence<size_t>{},
      index_sequence<(remove_cvref_t<_Vs>::__size() - 1)...>{},
      __first_index,
      __make_value_visitor<_Rp>(::cuda::std::forward<_Visitor>(__visitor)),
      ::cuda::std::forward<_Vs>(__vs)...);
  }

  template <class _Visitor, class... _Values>
  _CCCL_API static constexpr void __cccl_visit_exhaustive_visitor_check()
  {
    static_assert(is_invocable_v<_Visitor, _Values...>, "`std::visit` requires the visitor to be exhaustive.");
  }

  template <class _Visitor>
  struct __value_visitor
  {
    template <class... _Alts>
    [[nodiscard]] _CCCL_API constexpr decltype(auto) operator()(_Alts&&... __alts) const
    {
      __cccl_visit_exhaustive_visitor_check<_Visitor, decltype((::cuda::std::forward<_Alts>(__alts).__value))...>();
      return ::cuda::std::__invoke(
        ::cuda::std::forward<_Visitor>(__visitor), ::cuda::std::forward<_Alts>(__alts).__value...);
    }
    _Visitor&& __visitor;
  };

  template <class _Rp, class _Visitor>
  struct __value_visitor_return_type
  {
    template <class... _Alts>
    [[nodiscard]] _CCCL_API constexpr _Rp operator()(_Alts&&... __alts) const
    {
      __cccl_visit_exhaustive_visitor_check<_Visitor, decltype((::cuda::std::forward<_Alts>(__alts).__value))...>();
      return ::cuda::std::__invoke(
        ::cuda::std::forward<_Visitor>(__visitor), ::cuda::std::forward<_Alts>(__alts).__value...);
    }
    _Visitor&& __visitor;
  };

  template <class _Visitor>
  struct __value_visitor_return_type<void, _Visitor>
  {
    template <class... _Alts>
    _CCCL_API constexpr void operator()(_Alts&&... __alts) const
    {
      __cccl_visit_exhaustive_visitor_check<_Visitor, decltype((::cuda::std::forward<_Alts>(__alts).__value))...>();
      ::cuda::std::__invoke(::cuda::std::forward<_Visitor>(__visitor), ::cuda::std::forward<_Alts>(__alts).__value...);
    }
    _Visitor&& __visitor;
  };

  template <class _Visitor>
  [[nodiscard]] _CCCL_API static constexpr auto __make_value_visitor(_Visitor&& __visitor)
  {
    return __value_visitor<_Visitor>{::cuda::std::forward<_Visitor>(__visitor)};
  }

  template <class _Rp, class _Visitor>
  [[nodiscard]] _CCCL_API static constexpr auto __make_value_visitor(_Visitor&& __visitor)
  {
    return __value_visitor_return_type<_Rp, _Visitor>{::cuda::std::forward<_Visitor>(__visitor)};
  }
};
} // namespace __variant_detail::__visitation

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___VARIANT_VARIANT_VISIT_H
