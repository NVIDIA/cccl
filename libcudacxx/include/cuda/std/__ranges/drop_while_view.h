// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___RANGES_DROP_WHILE_VIEW_H
#define _LIBCUDACXX___RANGES_DROP_WHILE_VIEW_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/ranges_find_if_not.h>
#include <cuda/std/__concepts/constructible.h>
#include <cuda/std/__functional/bind_back.h>
#include <cuda/std/__functional/reference_wrapper.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__ranges/access.h>
#include <cuda/std/__ranges/all.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__ranges/enable_borrowed_range.h>
#include <cuda/std/__ranges/movable_box.h>
#include <cuda/std/__ranges/non_propagating_cache.h>
#include <cuda/std/__ranges/range_adaptor.h>
#include <cuda/std/__ranges/view_interface.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/is_nothrow_constructible.h>
#include <cuda/std/__type_traits/is_object.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/in_place.h>
#include <cuda/std/__utility/move.h>

#if _CCCL_STD_VER >= 2017 && !defined(_CCCL_COMPILER_MSVC_2017)

// MSVC complains about [[msvc::no_unique_address]] prior to C++20 as a vendor extension
_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4848)

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES
_LIBCUDACXX_BEGIN_NAMESPACE_RANGES_ABI

#  if _CCCL_STD_VER >= 2020
template <view _View, class _Pred>
  requires input_range<_View> && is_object_v<_Pred> && indirect_unary_predicate<const _Pred, iterator_t<_View>>
#  else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class _View,
          class _Pred,
          class = enable_if_t<view<_View>>,
          class = enable_if_t<input_range<_View>>,
          class = enable_if_t<is_object_v<_Pred>>,
          class = enable_if_t<indirect_unary_predicate<const _Pred, iterator_t<_View>>>>
#  endif // _CCCL_STD_VER <= 2017
class drop_while_view : public view_interface<drop_while_view<_View, _Pred>>
{
  _CCCL_NO_UNIQUE_ADDRESS _View __base_ = _View();
  _CCCL_NO_UNIQUE_ADDRESS __movable_box<_Pred> __pred_;

  static constexpr bool _UseCache = forward_range<_View>;
  using _Cache                    = _If<_UseCache, __non_propagating_cache<iterator_t<_View>>, __empty_cache>;
  _CCCL_NO_UNIQUE_ADDRESS _Cache __cached_begin_ = _Cache();

public:
#  if _CCCL_STD_VER >= 2020
  _CCCL_HIDE_FROM_ABI drop_while_view()
    requires default_initializable<_View> && default_initializable<_Pred>
  = default;
#  else // ^^^ C++20 ^^^ / vvv C++17 vvv
  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
  _LIBCUDACXX_REQUIRES(default_initializable<_View2>&& default_initializable<_Pred>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr drop_while_view() noexcept(
    is_nothrow_default_constructible_v<_View2> && is_nothrow_default_constructible_v<_Pred>)
      : view_interface<drop_while_view<_View, _Pred>>()
  {}
#  endif // _CCCL_STD_VER <= 2017

  _LIBCUDACXX_HIDE_FROM_ABI constexpr drop_while_view(_View __base, _Pred __pred)
      : view_interface<drop_while_view<_View, _Pred>>()
      , __base_(_CUDA_VSTD::move(__base))
      , __pred_(in_place, _CUDA_VSTD::move(__pred))
  {}

  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
  _LIBCUDACXX_REQUIRES(copy_constructible<_View2>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr _View base() const&
  {
    return __base_;
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr _View base() &&
  {
    return _CUDA_VSTD::move(__base_);
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr const _Pred& pred() const
  {
    return *__pred_;
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr auto begin()
  {
    _CCCL_ASSERT(__pred_.__has_value(),
                 "drop_while_view needs to have a non-empty predicate before calling begin() -- did a previous "
                 "assignment to this drop_while_view fail?");
    if constexpr (_UseCache)
    {
      if (!__cached_begin_.__has_value())
      {
        __cached_begin_.__emplace(_CUDA_VRANGES::find_if_not(__base_, _CUDA_VSTD::cref(*__pred_)));
      }
      return *__cached_begin_;
    }
    else
    {
      return _CUDA_VRANGES::find_if_not(__base_, _CUDA_VSTD::cref(*__pred_));
    }
    _CCCL_UNREACHABLE();
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr auto end()
  {
    return _CUDA_VRANGES::end(__base_);
  }
};

template <class _Range, class _Pred>
_CCCL_HOST_DEVICE drop_while_view(_Range&&, _Pred) -> drop_while_view<_CUDA_VIEWS::all_t<_Range>, _Pred>;

_LIBCUDACXX_END_NAMESPACE_RANGES_ABI

template <class _View, class _Pred>
_CCCL_INLINE_VAR constexpr bool enable_borrowed_range<drop_while_view<_View, _Pred>> = enable_borrowed_range<_View>;

_LIBCUDACXX_END_NAMESPACE_RANGES

_LIBCUDACXX_BEGIN_NAMESPACE_VIEWS
_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__drop_while)

struct __fn
{
  template <class _Range, class _Pred>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr auto operator()(_Range&& __range, _Pred&& __pred) const
    noexcept(noexcept(/**/ drop_while_view(_CUDA_VSTD::forward<_Range>(__range), _CUDA_VSTD::forward<_Pred>(__pred))))
      -> drop_while_view<all_t<_Range>, remove_cvref_t<_Pred>>
  {
    return /*-----------*/ drop_while_view(_CUDA_VSTD::forward<_Range>(__range), _CUDA_VSTD::forward<_Pred>(__pred));
  }

  _LIBCUDACXX_TEMPLATE(class _Pred)
  _LIBCUDACXX_REQUIRES(constructible_from<decay_t<_Pred>, _Pred>)
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr auto operator()(_Pred&& __pred) const
    noexcept(is_nothrow_constructible_v<decay_t<_Pred>, _Pred>)
  {
    return __range_adaptor_closure_t(_CUDA_VSTD::__bind_back(*this, _CUDA_VSTD::forward<_Pred>(__pred)));
  }
};
_LIBCUDACXX_END_NAMESPACE_CPO

inline namespace __cpo
{
_CCCL_GLOBAL_CONSTANT auto drop_while = __drop_while::__fn{};
} // namespace __cpo

_LIBCUDACXX_END_NAMESPACE_VIEWS

_CCCL_DIAG_POP

#endif // _CCCL_STD_VER >= 2017 && !defined(_CCCL_COMPILER_MSVC_2017)

#endif // _LIBCUDACXX___RANGES_DROP_WHILE_VIEW_H
