// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//
#ifndef _CUDA_STD___RANGES_OWNING_VIEW_H
#define _CUDA_STD___RANGES_OWNING_VIEW_H

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

_CCCL_BEGIN_NAMESPACE_RANGES

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
  _CCCL_API constexpr owning_view() noexcept(is_nothrow_default_constructible_v<_Range>)
      : view_interface<owning_view<_Rp>>()
  {}
#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^
  _CCCL_API constexpr owning_view(_Rp&& __r) noexcept(is_nothrow_move_constructible_v<_Rp>)
      : view_interface<owning_view<_Rp>>()
      , __r_(::cuda::std::move(__r))
  {}

  _CCCL_HIDE_FROM_ABI owning_view(owning_view&&)            = default;
  _CCCL_HIDE_FROM_ABI owning_view& operator=(owning_view&&) = default;

  [[nodiscard]] _CCCL_API constexpr _Rp& base() & noexcept
  {
    return __r_;
  }
  [[nodiscard]] _CCCL_API constexpr const _Rp& base() const& noexcept
  {
    return __r_;
  }
  [[nodiscard]] _CCCL_API constexpr _Rp&& base() && noexcept
  {
    return ::cuda::std::move(__r_);
  }
  [[nodiscard]] _CCCL_API constexpr const _Rp&& base() const&& noexcept
  {
    return ::cuda::std::move(__r_);
  }

  [[nodiscard]] _CCCL_API constexpr iterator_t<_Rp> begin()
  {
    return ::cuda::std::ranges::begin(__r_);
  }
  [[nodiscard]] _CCCL_API constexpr sentinel_t<_Rp> end()
  {
    return ::cuda::std::ranges::end(__r_);
  }

  _CCCL_TEMPLATE(class _Range = _Rp)
  _CCCL_REQUIRES(range<const _Range>)
  [[nodiscard]] _CCCL_API constexpr auto begin() const
  {
    return ::cuda::std::ranges::begin(__r_);
  }
  _CCCL_TEMPLATE(class _Range = _Rp)
  _CCCL_REQUIRES(range<const _Range>)
  [[nodiscard]] _CCCL_API constexpr auto end() const
  {
    return ::cuda::std::ranges::end(__r_);
  }

  _CCCL_TEMPLATE(class _Range = _Rp)
  _CCCL_REQUIRES(invocable<::cuda::std::ranges::__empty::__fn, _Range&>)
  [[nodiscard]] _CCCL_API constexpr bool empty()
  {
    return ::cuda::std::ranges::empty(__r_);
  }
  _CCCL_TEMPLATE(class _Range = _Rp)
  _CCCL_REQUIRES(invocable<::cuda::std::ranges::__empty::__fn, const _Range&>)
  [[nodiscard]] _CCCL_API constexpr bool empty() const
  {
    return ::cuda::std::ranges::empty(__r_);
  }

  _CCCL_TEMPLATE(class _Range = _Rp)
  _CCCL_REQUIRES(sized_range<_Range>)
  [[nodiscard]] _CCCL_API constexpr auto size()
  {
    return ::cuda::std::ranges::size(__r_);
  }
  _CCCL_TEMPLATE(class _Range = _Rp)
  _CCCL_REQUIRES(sized_range<const _Range>)
  [[nodiscard]] _CCCL_API constexpr auto size() const
  {
    return ::cuda::std::ranges::size(__r_);
  }

  _CCCL_TEMPLATE(class _Range = _Rp)
  _CCCL_REQUIRES(contiguous_range<_Range>)
  [[nodiscard]] _CCCL_API constexpr auto data()
  {
    return ::cuda::std::ranges::data(__r_);
  }
  _CCCL_TEMPLATE(class _Range = _Rp)
  _CCCL_REQUIRES(contiguous_range<const _Range>)
  [[nodiscard]] _CCCL_API constexpr auto data() const
  {
    return ::cuda::std::ranges::data(__r_);
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(owning_view);

template <class _Rp>
inline constexpr bool enable_borrowed_range<owning_view<_Rp>> = enable_borrowed_range<_Rp>;

_CCCL_END_NAMESPACE_RANGES

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___RANGES_OWNING_VIEW_H
