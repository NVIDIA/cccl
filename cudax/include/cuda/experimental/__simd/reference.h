// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD_EXPERIMENTAL___SIMD_REFERENCE_H
#define _CUDA_STD_EXPERIMENTAL___SIMD_REFERENCE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/experimental/__simd/config.h>
#include <cuda/std/experimental/__simd/utility.h>

#if _LIBCUDACXX_EXPERIMENTAL_SIMD_ENABLED

#  include <cuda/std/__cstddef/size_t.h>
#  include <cuda/std/__type_traits/enable_if.h>
#  include <cuda/std/__type_traits/is_assignable.h>
#  include <cuda/std/__type_traits/is_same.h>
#  include <cuda/std/__utility/declval.h>
#  include <cuda/std/__utility/forward.h>
#  include <cuda/std/__utility/move.h>
#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

namespace experimental
{
inline namespace parallelism_v2
{

template <class _Tp, class _Storage, class _Vp>
class __simd_reference
{
  template <class, class>
  friend class simd;
  template <class, class>
  friend class simd_mask;

  _Storage& __s_;
  size_t __idx_;

  _CCCL_HIDE_FROM_ABI __simd_reference(_Storage& __s, size_t __idx)
      : __s_(__s)
      , __idx_(__idx)
  {}

  _CCCL_HIDE_FROM_ABI _Vp __get() const noexcept
  {
    return __s_.__get(__idx_);
  }

  _CCCL_HIDE_FROM_ABI void __set(_Vp __v)
  {
    if constexpr (is_same_v<_Vp, bool>)
    {
      __s_.__set(__idx_, experimental::__set_all_bits<_Tp>(__v));
    }
    else
    {
      __s_.__set(__idx_, __v);
    }
  }

public:
  using value_type = _Vp;

  __simd_reference()                        = delete;
  __simd_reference(const __simd_reference&) = delete;

  _CCCL_HIDE_FROM_ABI operator value_type() const noexcept
  {
    return __get();
  }

  template <class _Up, enable_if_t<is_assignable_v<value_type&, _Up&&>, int> = 0>
  _CCCL_HIDE_FROM_ABI __simd_reference operator=(_Up&& __v) && noexcept
  {
    __set(static_cast<value_type>(::cuda::std::forward<_Up>(__v)));
    return {__s_, __idx_};
  }

  template <class _Tp1, class _Storage1, class _Vp1>
  friend _CCCL_HIDE_FROM_ABI void swap(
    __simd_reference<_Tp1, _Storage1, _Vp1>&& __a,
    __simd_reference<_Tp1, _Storage1, _Vp1>&& __b) noexcept;

  template <class _Tp1, class _Storage1, class _Vp1>
  friend _CCCL_HIDE_FROM_ABI void swap(_Vp1& __a, __simd_reference<_Tp1, _Storage1, _Vp1>&& __b) noexcept;

  template <class _Tp1, class _Storage1, class _Vp1>
  friend _CCCL_HIDE_FROM_ABI void swap(__simd_reference<_Tp1, _Storage1, _Vp1>&& __a, _Vp1& __b) noexcept;

  template <class _Up, class = decltype(::cuda::std::declval<value_type&>() += ::cuda::std::declval<_Up>())>
  _CCCL_HIDE_FROM_ABI __simd_reference operator+=(_Up&& __v) && noexcept
  {
    __set(__get() + static_cast<value_type>(::cuda::std::forward<_Up>(__v)));
    return {__s_, __idx_};
  }

  template <class _Up, class = decltype(::cuda::std::declval<value_type&>() -= ::cuda::std::declval<_Up>())>
  _CCCL_HIDE_FROM_ABI __simd_reference operator-=(_Up&& __v) && noexcept
  {
    __set(__get() - static_cast<value_type>(::cuda::std::forward<_Up>(__v)));
    return {__s_, __idx_};
  }

  template <class _Up, class = decltype(::cuda::std::declval<value_type&>() *= ::cuda::std::declval<_Up>())>
  _CCCL_HIDE_FROM_ABI __simd_reference operator*=(_Up&& __v) && noexcept
  {
    __set(__get() * static_cast<value_type>(::cuda::std::forward<_Up>(__v)));
    return {__s_, __idx_};
  }

  template <class _Up, class = decltype(::cuda::std::declval<value_type&>() /= ::cuda::std::declval<_Up>())>
  _CCCL_HIDE_FROM_ABI __simd_reference operator/=(_Up&& __v) && noexcept
  {
    __set(__get() / static_cast<value_type>(::cuda::std::forward<_Up>(__v)));
    return {__s_, __idx_};
  }

  template <class _Up, class = decltype(::cuda::std::declval<value_type&>() %= ::cuda::std::declval<_Up>())>
  _CCCL_HIDE_FROM_ABI __simd_reference operator%=(_Up&& __v) && noexcept
  {
    __set(__get() % static_cast<value_type>(::cuda::std::forward<_Up>(__v)));
    return {__s_, __idx_};
  }

  template <class _Up, class = decltype(::cuda::std::declval<value_type&>() &= ::cuda::std::declval<_Up>())>
  _CCCL_HIDE_FROM_ABI __simd_reference operator&=(_Up&& __v) && noexcept
  {
    __set(__get() & static_cast<value_type>(::cuda::std::forward<_Up>(__v)));
    return {__s_, __idx_};
  }

  template <class _Up, class = decltype(::cuda::std::declval<value_type&>() |= ::cuda::std::declval<_Up>())>
  _CCCL_HIDE_FROM_ABI __simd_reference operator|=(_Up&& __v) && noexcept
  {
    __set(__get() | static_cast<value_type>(::cuda::std::forward<_Up>(__v)));
    return {__s_, __idx_};
  }

  template <class _Up, class = decltype(::cuda::std::declval<value_type&>() ^= ::cuda::std::declval<_Up>())>
  _CCCL_HIDE_FROM_ABI __simd_reference operator^=(_Up&& __v) && noexcept
  {
    __set(__get() ^ static_cast<value_type>(::cuda::std::forward<_Up>(__v)));
    return {__s_, __idx_};
  }

  template <class _Up, class = decltype(::cuda::std::declval<value_type&>() <<= ::cuda::std::declval<_Up>())>
  _CCCL_HIDE_FROM_ABI __simd_reference operator<<=(_Up&& __v) && noexcept
  {
    __set(__get() << static_cast<value_type>(::cuda::std::forward<_Up>(__v)));
    return {__s_, __idx_};
  }

  template <class _Up, class = decltype(::cuda::std::declval<value_type&>() >>= ::cuda::std::declval<_Up>())>
  _CCCL_HIDE_FROM_ABI __simd_reference operator>>=(_Up&& __v) && noexcept
  {
    __set(__get() >> static_cast<value_type>(::cuda::std::forward<_Up>(__v)));
    return {__s_, __idx_};
  }

  _CCCL_HIDE_FROM_ABI __simd_reference operator++() && noexcept
  {
    __set(__get() + 1);
    return {__s_, __idx_};
  }

  _CCCL_HIDE_FROM_ABI value_type operator++(int) && noexcept
  {
    auto __r = __get();
    __set(__get() + 1);
    return __r;
  }

  _CCCL_HIDE_FROM_ABI __simd_reference operator--() && noexcept
  {
    __set(__get() - 1);
    return {__s_, __idx_};
  }

  _CCCL_HIDE_FROM_ABI value_type operator--(int) && noexcept
  {
    auto __r = __get();
    __set(__get() - 1);
    return __r;
  }
};

template <class _Tp, class _Storage, class _Vp>
_CCCL_HIDE_FROM_ABI void
swap(__simd_reference<_Tp, _Storage, _Vp>&& __a, __simd_reference<_Tp, _Storage, _Vp>&& __b) noexcept
{
  _Vp __tmp(::cuda::std::move(__a));
  ::cuda::std::move(__a) = ::cuda::std::move(__b);
  ::cuda::std::move(__b) = ::cuda::std::move(__tmp);
}

template <class _Tp, class _Storage, class _Vp>
_CCCL_HIDE_FROM_ABI void swap(_Vp& __a, __simd_reference<_Tp, _Storage, _Vp>&& __b) noexcept
{
  _Vp __tmp(::cuda::std::move(__a));
  __a                    = ::cuda::std::move(__b);
  ::cuda::std::move(__b) = ::cuda::std::move(__tmp);
}

template <class _Tp, class _Storage, class _Vp>
_CCCL_HIDE_FROM_ABI void swap(__simd_reference<_Tp, _Storage, _Vp>&& __a, _Vp& __b) noexcept
{
  _Vp __tmp(::cuda::std::move(__a));
  ::cuda::std::move(__a) = ::cuda::std::move(__b);
  __b                    = ::cuda::std::move(__tmp);
}

} // namespace parallelism_v2
} // namespace experimental

_CCCL_END_NAMESPACE_CUDA_STD

#  include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX_EXPERIMENTAL_SIMD_ENABLED

#endif // _CUDA_STD_EXPERIMENTAL___SIMD_REFERENCE_H

