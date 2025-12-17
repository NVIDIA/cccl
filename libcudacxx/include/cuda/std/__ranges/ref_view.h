// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//
#ifndef _CUDA_STD___RANGES_REF_VIEW_H
#define _CUDA_STD___RANGES_REF_VIEW_H

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

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_RANGES

template <class _Range>
struct __conversion_tester
{
  _CCCL_API inline static void __fun(_Range&);
  static void __fun(_Range&&) = delete;
};

template <class _Tp, class _Range>
_CCCL_CONCEPT __convertible_to_lvalue =
  _CCCL_REQUIRES_EXPR((_Tp, _Range))((__conversion_tester<_Range>::__fun(declval<_Tp>())));

#if _CCCL_HAS_CONCEPTS()

template <range _Range>
  requires is_object_v<_Range>
#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv
template <class _Range, enable_if_t<range<_Range>, int> = 0, enable_if_t<is_object_v<_Range>, int> = 0>
#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^
class ref_view : public view_interface<ref_view<_Range>>
{
  _Range* __range_;

public:
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__different_from<_Tp, ref_view> _CCCL_AND convertible_to<_Tp, _Range&> _CCCL_AND
                   __convertible_to_lvalue<_Tp, _Range>)
  _CCCL_API constexpr ref_view(_Tp&& __t)
      : view_interface<ref_view<_Range>>()
      , __range_(::cuda::std::addressof(static_cast<_Range&>(::cuda::std::forward<_Tp>(__t))))
  {}

  _CCCL_API constexpr _Range& base() const
  {
    return *__range_;
  }

  _CCCL_API constexpr iterator_t<_Range> begin() const
  {
    return ::cuda::std::ranges::begin(*__range_);
  }
  _CCCL_API constexpr sentinel_t<_Range> end() const
  {
    return ::cuda::std::ranges::end(*__range_);
  }

  _CCCL_TEMPLATE(class _Range2 = _Range)
  _CCCL_REQUIRES(invocable<::cuda::std::ranges::__empty::__fn, const _Range2&>)
  _CCCL_API constexpr bool empty() const
  {
    return ::cuda::std::ranges::empty(*__range_);
  }

  _CCCL_TEMPLATE(class _Range2 = _Range)
  _CCCL_REQUIRES(sized_range<_Range2>)
  _CCCL_API constexpr auto size() const
  {
    return ::cuda::std::ranges::size(*__range_);
  }

  _CCCL_TEMPLATE(class _Range2 = _Range)
  _CCCL_REQUIRES(contiguous_range<_Range2>)
  _CCCL_API constexpr auto data() const
  {
    return ::cuda::std::ranges::data(*__range_);
  }
};

template <class _Range>
_CCCL_HOST_DEVICE ref_view(_Range&) -> ref_view<_Range>;

template <class _Tp>
inline constexpr bool enable_borrowed_range<ref_view<_Tp>> = true;

_CCCL_END_NAMESPACE_CUDA_STD_RANGES

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___RANGES_REF_VIEW_H
