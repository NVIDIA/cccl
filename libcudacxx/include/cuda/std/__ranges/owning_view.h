// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//
#ifndef _LIBCUDACXX___RANGES_OWNING_VIEW_H
#define _LIBCUDACXX___RANGES_OWNING_VIEW_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/constructible.h>
#include <cuda/std/__concepts/movable.h>
#include <cuda/std/__ranges/access.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__ranges/data.h>
#include <cuda/std/__ranges/empty.h>
#include <cuda/std/__ranges/enable_borrowed_range.h>
#include <cuda/std/__ranges/size.h>
#include <cuda/std/__ranges/view_interface.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_nothrow_default_constructible.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/move.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES

#if _CCCL_HAS_CONCEPTS()
template <range _Rp>
  requires movable<_Rp> && (!__is_std_initializer_list<remove_cvref_t<_Rp>>)
#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv
template <class _Rp,
          enable_if_t<range<_Rp>, int>                                      = 0,
          enable_if_t<movable<_Rp>, int>                                    = 0,
          enable_if_t<!__is_std_initializer_list<remove_cvref_t<_Rp>>, int> = 0>
#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^
class owning_view : public view_interface<owning_view<_Rp>>
{
  _Rp __r_ = _Rp();

public:
#if _CCCL_HAS_CONCEPTS()
  _CCCL_HIDE_FROM_ABI owning_view()
    requires default_initializable<_Rp>
  = default;
#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv
  _CCCL_TEMPLATE(class _Range = _Rp)
  _CCCL_REQUIRES(default_initializable<_Range>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr owning_view() noexcept(is_nothrow_default_constructible_v<_Range>)
      : view_interface<owning_view<_Rp>>()
  {}
#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^
  _LIBCUDACXX_HIDE_FROM_ABI constexpr owning_view(_Rp&& __r) noexcept(is_nothrow_move_constructible_v<_Rp>)
      : view_interface<owning_view<_Rp>>()
      , __r_(_CUDA_VSTD::move(__r))
  {}

  _CCCL_HIDE_FROM_ABI owning_view(owning_view&&)            = default;
  _CCCL_HIDE_FROM_ABI owning_view& operator=(owning_view&&) = default;

  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr _Rp& base() & noexcept
  {
    return __r_;
  }
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr const _Rp& base() const& noexcept
  {
    return __r_;
  }
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr _Rp&& base() && noexcept
  {
    return _CUDA_VSTD::move(__r_);
  }
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr const _Rp&& base() const&& noexcept
  {
    return _CUDA_VSTD::move(__r_);
  }

  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr iterator_t<_Rp> begin()
  {
    return _CUDA_VRANGES::begin(__r_);
  }
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr sentinel_t<_Rp> end()
  {
    return _CUDA_VRANGES::end(__r_);
  }

  _CCCL_TEMPLATE(class _Range = _Rp)
  _CCCL_REQUIRES(range<const _Range>)
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr auto begin() const
  {
    return _CUDA_VRANGES::begin(__r_);
  }
  _CCCL_TEMPLATE(class _Range = _Rp)
  _CCCL_REQUIRES(range<const _Range>)
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr auto end() const
  {
    return _CUDA_VRANGES::end(__r_);
  }

  _CCCL_TEMPLATE(class _Range = _Rp)
  _CCCL_REQUIRES(invocable<_CUDA_VRANGES::__empty::__fn, _Range&>)
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr bool empty()
  {
    return _CUDA_VRANGES::empty(__r_);
  }
  _CCCL_TEMPLATE(class _Range = _Rp)
  _CCCL_REQUIRES(invocable<_CUDA_VRANGES::__empty::__fn, const _Range&>)
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr bool empty() const
  {
    return _CUDA_VRANGES::empty(__r_);
  }

  _CCCL_TEMPLATE(class _Range = _Rp)
  _CCCL_REQUIRES(sized_range<_Range>)
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr auto size()
  {
    return _CUDA_VRANGES::size(__r_);
  }
  _CCCL_TEMPLATE(class _Range = _Rp)
  _CCCL_REQUIRES(sized_range<const _Range>)
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr auto size() const
  {
    return _CUDA_VRANGES::size(__r_);
  }

  _CCCL_TEMPLATE(class _Range = _Rp)
  _CCCL_REQUIRES(contiguous_range<_Range>)
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr auto data()
  {
    return _CUDA_VRANGES::data(__r_);
  }
  _CCCL_TEMPLATE(class _Range = _Rp)
  _CCCL_REQUIRES(contiguous_range<const _Range>)
  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr auto data() const
  {
    return _CUDA_VRANGES::data(__r_);
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(owning_view);

template <class _Rp>
inline constexpr bool enable_borrowed_range<owning_view<_Rp>> = enable_borrowed_range<_Rp>;

_LIBCUDACXX_END_NAMESPACE_RANGES

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___RANGES_OWNING_VIEW_H
