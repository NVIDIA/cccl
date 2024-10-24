// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//
#ifndef _LIBCUDACXX___RANGES_ALL_H
#define _LIBCUDACXX___RANGES_ALL_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__ranges/access.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__ranges/owning_view.h>
#include <cuda/std/__ranges/range_adaptor.h>
#include <cuda/std/__ranges/ref_view.h>
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__utility/auto_cast.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/forward.h>

_LIBCUDACXX_BEGIN_NAMESPACE_VIEWS

#if _CCCL_STD_VER >= 2017 && !defined(_CCCL_COMPILER_MSVC_2017)

_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__all)
#  if _CCCL_STD_VER >= 2020
template <class _Tp>
concept __to_ref_view = (!_CUDA_VRANGES::view<decay_t<_Tp>>) && requires(_Tp&& __t) {
  _CUDA_VRANGES::ref_view{_CUDA_VSTD::forward<_Tp>(__t)};
};

template <class _Tp>
concept __to_owning_view = (!_CUDA_VRANGES::view<decay_t<_Tp>> && !__to_ref_view<_Tp>) && requires(_Tp&& __t) {
  _CUDA_VRANGES::owning_view{_CUDA_VSTD::forward<_Tp>(__t)};
};
#  else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class _Tp>
_LIBCUDACXX_CONCEPT_FRAGMENT(__can_ref_view_,
                             requires(_Tp&& __t)((_CUDA_VRANGES::ref_view{_CUDA_VSTD::forward<_Tp>(__t)})));

template <class _Tp>
_LIBCUDACXX_CONCEPT __can_ref_view = _LIBCUDACXX_FRAGMENT(__can_ref_view_, _Tp);

template <class _Tp>
_LIBCUDACXX_CONCEPT_FRAGMENT(__to_ref_view_,
                             requires()(requires(!_CUDA_VRANGES::view<decay_t<_Tp>>), requires(__can_ref_view<_Tp>)));

template <class _Tp>
_LIBCUDACXX_CONCEPT __to_ref_view = _LIBCUDACXX_FRAGMENT(__to_ref_view_, _Tp);

template <class _Tp>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  __to_owning_view_,
  requires(_Tp&& __t)(requires(!_CUDA_VRANGES::view<decay_t<_Tp>>),
                      requires(!__can_ref_view<_Tp>),
                      (_CUDA_VRANGES::owning_view{_CUDA_VSTD::forward<_Tp>(__t)})));

template <class _Tp>
_LIBCUDACXX_CONCEPT __to_owning_view = _LIBCUDACXX_FRAGMENT(__to_owning_view_, _Tp);
#  endif // _CCCL_STD_VER <= 2017

struct __fn : __range_adaptor_closure<__fn>
{
  _LIBCUDACXX_TEMPLATE(class _Tp)
  _LIBCUDACXX_REQUIRES(_CUDA_VRANGES::view<decay_t<_Tp>>)
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr auto operator()(_Tp&& __t) const
    noexcept(noexcept(_LIBCUDACXX_AUTO_CAST(_CUDA_VSTD::forward<_Tp>(__t))))
      -> decltype(_LIBCUDACXX_AUTO_CAST(_CUDA_VSTD::forward<_Tp>(__t)))
  {
    return _LIBCUDACXX_AUTO_CAST(_CUDA_VSTD::forward<_Tp>(__t));
  }

  _LIBCUDACXX_TEMPLATE(class _Tp)
  _LIBCUDACXX_REQUIRES(__to_ref_view<_Tp>)
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr auto operator()(_Tp&& __t) const
    noexcept(noexcept(_CUDA_VRANGES::ref_view{_CUDA_VSTD::forward<_Tp>(__t)}))
  {
    return _CUDA_VRANGES::ref_view{_CUDA_VSTD::forward<_Tp>(__t)};
  }

  _LIBCUDACXX_TEMPLATE(class _Tp)
  _LIBCUDACXX_REQUIRES(__to_owning_view<_Tp>)
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr auto operator()(_Tp&& __t) const
    noexcept(noexcept(_CUDA_VRANGES::owning_view{_CUDA_VSTD::forward<_Tp>(__t)}))
  {
    return _CUDA_VRANGES::owning_view{_CUDA_VSTD::forward<_Tp>(__t)};
  }
};
_LIBCUDACXX_END_NAMESPACE_CPO

inline namespace __cpo
{
_CCCL_GLOBAL_CONSTANT auto all = __all::__fn{};
} // namespace __cpo

#  if _CCCL_STD_VER >= 2020
template <_CUDA_VRANGES::viewable_range _Range>
using all_t = decltype(_CUDA_VIEWS::all(declval<_Range>()));
#  else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class _Range>
using all_t = enable_if_t<_CUDA_VRANGES::viewable_range<_Range>, decltype(_CUDA_VIEWS::all(declval<_Range>()))>;
#  endif // _CCCL_STD_VER <= 2017

#endif // _CCCL_STD_VER >= 2017 && !_CCCL_COMPILER_MSVC_2017

_LIBCUDACXX_END_NAMESPACE_VIEWS

#endif // _LIBCUDACXX___RANGES_ALL_H
