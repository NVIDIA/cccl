// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//
#ifndef _LIBCUDACXX___RANGES_REF_VIEW_H
#define _LIBCUDACXX___RANGES_REF_VIEW_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/convertible_to.h>
#include <cuda/std/__concepts/different_from.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/incrementable_traits.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__ranges/access.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__ranges/data.h>
#include <cuda/std/__ranges/empty.h>
#include <cuda/std/__ranges/enable_borrowed_range.h>
#include <cuda/std/__ranges/size.h>
#include <cuda/std/__ranges/view_interface.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_object.h>
#include <cuda/std/__utility/forward.h>

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES

#if _CCCL_STD_VER >= 2017 && !defined(_CCCL_COMPILER_MSVC_2017)

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES_ABI

template <class _Range>
struct __conversion_tester
{
  _LIBCUDACXX_HIDE_FROM_ABI static void __fun(_Range&);
  static void __fun(_Range&&) = delete;
};

#  if _CCCL_STD_VER >= 2020
template <class _Tp, class _Range>
concept __convertible_to_lvalue = requires { __conversion_tester<_Range>::__fun(declval<_Tp>()); };

template <range _Range>
  requires is_object_v<_Range>
#  else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class _Tp, class _Range, class = void>
_CCCL_INLINE_VAR constexpr bool __convertible_to_lvalue = false;

template <class _Tp, class _Range>
_CCCL_INLINE_VAR constexpr bool
  __convertible_to_lvalue<_Tp, _Range, void_t<decltype(__conversion_tester<_Range>::__fun(declval<_Tp>()))>> = true;

template <class _Range, enable_if_t<range<_Range>, int> = 0, enable_if_t<is_object_v<_Range>, int> = 0>
#  endif // _CCCL_STD_VER <= 2017
class ref_view : public view_interface<ref_view<_Range>>
{
  _Range* __range_;

public:
  _LIBCUDACXX_TEMPLATE(class _Tp)
  _LIBCUDACXX_REQUIRES(__different_from<_Tp, ref_view> _LIBCUDACXX_AND convertible_to<_Tp, _Range&> _LIBCUDACXX_AND
                         __convertible_to_lvalue<_Tp, _Range>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr ref_view(_Tp&& __t)
      : view_interface<ref_view<_Range>>()
      , __range_(_CUDA_VSTD::addressof(static_cast<_Range&>(_CUDA_VSTD::forward<_Tp>(__t))))
  {}

  _LIBCUDACXX_HIDE_FROM_ABI constexpr _Range& base() const
  {
    return *__range_;
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr iterator_t<_Range> begin() const
  {
    return _CUDA_VRANGES::begin(*__range_);
  }
  _LIBCUDACXX_HIDE_FROM_ABI constexpr sentinel_t<_Range> end() const
  {
    return _CUDA_VRANGES::end(*__range_);
  }

  _LIBCUDACXX_TEMPLATE(class _Range2 = _Range)
  _LIBCUDACXX_REQUIRES(invocable<_CUDA_VRANGES::__empty::__fn, const _Range2&>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr bool empty() const
  {
    return _CUDA_VRANGES::empty(*__range_);
  }

  _LIBCUDACXX_TEMPLATE(class _Range2 = _Range)
  _LIBCUDACXX_REQUIRES(sized_range<_Range2>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr auto size() const
  {
    return _CUDA_VRANGES::size(*__range_);
  }

  _LIBCUDACXX_TEMPLATE(class _Range2 = _Range)
  _LIBCUDACXX_REQUIRES(contiguous_range<_Range2>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr auto data() const
  {
    return _CUDA_VRANGES::data(*__range_);
  }
};

template <class _Range>
_CCCL_HOST_DEVICE ref_view(_Range&) -> ref_view<_Range>;

_LIBCUDACXX_END_NAMESPACE_RANGES_ABI

template <class _Tp>
_CCCL_INLINE_VAR constexpr bool enable_borrowed_range<ref_view<_Tp>> = true;

#endif // _CCCL_STD_VER >= 2017 && !_CCCL_COMPILER_MSVC_2017

_LIBCUDACXX_END_NAMESPACE_RANGES

#endif // _LIBCUDACXX___RANGES_REF_VIEW_H
