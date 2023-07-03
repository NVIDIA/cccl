// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//
#ifndef _LIBCUDACXX___RANGES_REF_VIEW_H
#define _LIBCUDACXX___RANGES_REF_VIEW_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__concepts/convertible_to.h"
#include "../__concepts/different_from.h"
#include "../__iterator/concepts.h"
#include "../__iterator/incrementable_traits.h"
#include "../__iterator/iterator_traits.h"
#include "../__memory/addressof.h"
#include "../__ranges/access.h"
#include "../__ranges/concepts.h"
#include "../__ranges/data.h"
#include "../__ranges/empty.h"
#include "../__ranges/enable_borrowed_range.h"
#include "../__ranges/size.h"
#include "../__ranges/view_interface.h"
#include "../__type_traits/enable_if.h"
#include "../__type_traits/is_object.h"
#include "../__utility/forward.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES

#if _LIBCUDACXX_STD_VER > 14

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES_ABI

template <class _Range>
struct __conversion_tester {
    _LIBCUDACXX_INLINE_VISIBILITY static void __fun(_Range&);
    static void __fun(_Range&&) = delete;
};

#if _LIBCUDACXX_STD_VER > 17
template <class _Tp, class _Range>
concept __convertible_to_lvalue = requires { __conversion_tester<_Range>::__fun(declval<_Tp>()); };

template<range _Range>
  requires is_object_v<_Range>
#else
template <class _Tp, class _Range, class = void>
_LIBCUDACXX_INLINE_VAR constexpr bool __convertible_to_lvalue = false;

template <class _Tp, class _Range>
_LIBCUDACXX_INLINE_VAR constexpr bool
  __convertible_to_lvalue<_Tp, _Range, void_t<decltype(__conversion_tester<_Range>::__fun(declval<_Tp>()))>> = true;

template<class _Range, enable_if_t<range<_Range>, int> = 0,
                       enable_if_t<is_object_v<_Range>, int> = 0>
#endif // _LIBCUDACXX_STD_VER > 11
class ref_view : public view_interface<ref_view<_Range>> {
  _Range *__range_;

public:
_LIBCUDACXX_TEMPLATE(class _Tp)
  (requires __different_from<_Tp, ref_view> _LIBCUDACXX_AND
            convertible_to<_Tp, _Range&> _LIBCUDACXX_AND
            __convertible_to_lvalue<_Tp, _Range>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr ref_view(_Tp&& __t)
    : view_interface<ref_view<_Range>>()
    , __range_(_CUDA_VSTD::addressof(static_cast<_Range&>(_CUDA_VSTD::forward<_Tp>(__t))))
  {}

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr _Range& base() const { return *__range_; }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr iterator_t<_Range> begin() const { return _CUDA_VRANGES::begin(*__range_); }
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr sentinel_t<_Range> end() const { return _CUDA_VRANGES::end(*__range_); }

  _LIBCUDACXX_TEMPLATE(class _Range2 = _Range)
    (requires invocable<_CUDA_VRANGES::__empty::__fn, const _Range2&>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr bool empty() const
  { return _CUDA_VRANGES::empty(*__range_); }

  _LIBCUDACXX_TEMPLATE(class _Range2 = _Range)
    (requires sized_range<_Range2>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr auto size() const
  { return _CUDA_VRANGES::size(*__range_); }

  _LIBCUDACXX_TEMPLATE(class _Range2 = _Range)
    (requires contiguous_range<_Range2>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr auto data() const
  { return _CUDA_VRANGES::data(*__range_); }
};

template<class _Range>
ref_view(_Range&) -> ref_view<_Range>;

_LIBCUDACXX_END_NAMESPACE_RANGES_ABI

template<class _Tp>
inline constexpr bool enable_borrowed_range<ref_view<_Tp>> = true;

#endif // _LIBCUDACXX_STD_VER > 14

_LIBCUDACXX_END_NAMESPACE_RANGES

#endif // _LIBCUDACXX___RANGES_REF_VIEW_H
