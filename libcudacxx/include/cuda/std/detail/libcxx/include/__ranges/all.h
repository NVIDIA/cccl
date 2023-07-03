// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//
#ifndef _LIBCUDACXX___RANGES_ALL_H
#define _LIBCUDACXX___RANGES_ALL_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__iterator/concepts.h"
#include "../__iterator/iterator_traits.h"
#include "../__ranges/access.h"
#include "../__ranges/concepts.h"
#include "../__ranges/owning_view.h"
#include "../__ranges/range_adaptor.h"
#include "../__ranges/ref_view.h"
#include "../__type_traits/decay.h"
#include "../__utility/auto_cast.h"
#include "../__utility/declval.h"
#include "../__utility/forward.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_VIEWS

#if _LIBCUDACXX_STD_VER > 14

_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__all)
#if _LIBCUDACXX_STD_VER > 17
  template<class _Tp>
  concept __to_ref_view = (!_CUDA_VRANGES::view<decay_t<_Tp>>) &&
                          requires (_Tp&& __t) { _CUDA_VRANGES::ref_view{_CUDA_VSTD::forward<_Tp>(__t)}; };

  template<class _Tp>
  concept __to_owning_view = (!_CUDA_VRANGES::view<decay_t<_Tp>> && !__to_ref_view<_Tp>) &&
                          requires (_Tp&& __t) { _CUDA_VRANGES::owning_view{_CUDA_VSTD::forward<_Tp>(__t)}; };
#else
  template<class _Tp>
  _LIBCUDACXX_CONCEPT_FRAGMENT(
    __can_ref_view_,
    requires(_Tp&& __t)(
      (_CUDA_VRANGES::ref_view{_CUDA_VSTD::forward<_Tp>(__t)})
    ));

  template<class _Tp>
  _LIBCUDACXX_CONCEPT __can_ref_view = _LIBCUDACXX_FRAGMENT(__can_ref_view_, _Tp);

  template<class _Tp>
  _LIBCUDACXX_CONCEPT_FRAGMENT(
    __to_ref_view_,
    requires()(
      requires(!_CUDA_VRANGES::view<decay_t<_Tp>>),
      requires(__can_ref_view<_Tp>)
    ));

  template<class _Tp>
  _LIBCUDACXX_CONCEPT __to_ref_view = _LIBCUDACXX_FRAGMENT(__to_ref_view_, _Tp);

  template<class _Tp>
  _LIBCUDACXX_CONCEPT_FRAGMENT(
    __to_owning_view_,
    requires(_Tp&& __t)(
      requires(!_CUDA_VRANGES::view<decay_t<_Tp>>),
      requires(!__can_ref_view<_Tp>),
      (_CUDA_VRANGES::owning_view{_CUDA_VSTD::forward<_Tp>(__t)})
    ));

  template<class _Tp>
  _LIBCUDACXX_CONCEPT __to_owning_view = _LIBCUDACXX_FRAGMENT(__to_owning_view_, _Tp);
#endif

  struct __fn : __range_adaptor_closure<__fn> {
    _LIBCUDACXX_TEMPLATE(class _Tp)
      (requires _CUDA_VRANGES::view<decay_t<_Tp>>)
    _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr auto operator()(_Tp&& __t) const
      noexcept(noexcept(_LIBCUDACXX_AUTO_CAST(_CUDA_VSTD::forward<_Tp>(__t))))
            -> decltype(_LIBCUDACXX_AUTO_CAST(_CUDA_VSTD::forward<_Tp>(__t))) {
      return            _LIBCUDACXX_AUTO_CAST(_CUDA_VSTD::forward<_Tp>(__t));
    }

    _LIBCUDACXX_TEMPLATE(class _Tp)
      (requires __to_ref_view<_Tp>)
    _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr auto operator()(_Tp&& __t) const
      noexcept(noexcept(_CUDA_VRANGES::ref_view{_CUDA_VSTD::forward<_Tp>(__t)}))
    {
      return _CUDA_VRANGES::ref_view{_CUDA_VSTD::forward<_Tp>(__t)};
    }

    _LIBCUDACXX_TEMPLATE(class _Tp)
      (requires __to_owning_view<_Tp>)
    _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr auto operator()(_Tp&& __t) const
      noexcept(noexcept(_CUDA_VRANGES::owning_view{_CUDA_VSTD::forward<_Tp>(__t)}))
    {
      return _CUDA_VRANGES::owning_view{_CUDA_VSTD::forward<_Tp>(__t)};
    }
  };
_LIBCUDACXX_END_NAMESPACE_CPO

inline namespace __cpo {
  _LIBCUDACXX_CPO_ACCESSIBILITY auto all = __all::__fn{};
} // namespace __cpo

#if _LIBCUDACXX_STD_VER > 17
template<_CUDA_VRANGES::viewable_range _Range>
using all_t = decltype(_CUDA_VIEWS::all(declval<_Range>()));
#else
template<class _Range>
using all_t = enable_if_t<_CUDA_VRANGES::viewable_range<_Range>, decltype(_CUDA_VIEWS::all(declval<_Range>()))>;
#endif

#endif // _LIBCUDACXX_STD_VER > 14

_LIBCUDACXX_END_NAMESPACE_VIEWS

#endif // _LIBCUDACXX___RANGES_ALL_H
