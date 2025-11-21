//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___SIMD_REFERENCE_H
#define _CUDAX___SIMD_REFERENCE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__type_traits/is_assignable.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/move.h>

#include <cuda/experimental/__simd/utility.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::datapar
{
template <typename _Tp, typename _Storage, typename _Vp>
class __simd_reference
{
  template <typename, typename>
  friend class basic_simd;

  template <typename, typename>
  friend class basic_simd_mask;

  _Storage& __s_;
  ::cuda::std::size_t __idx_;

  _CCCL_API __simd_reference(_Storage& __s, ::cuda::std::size_t __idx)
      : __s_{__s}
      , __idx_{__idx}
  {}

  [[nodiscard]] _CCCL_API constexpr _Vp __get() const noexcept
  {
    return __s_.__get(__idx_);
  }

  _CCCL_API constexpr void __set(_Vp __v) noexcept
  {
    if constexpr (::cuda::std::is_same_v<_Vp, bool>)
    {
      __s_.__set(__idx_, ::cuda::experimental::datapar::__mask_bits_from_bool<_Storage>(__v));
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

  _CCCL_API constexpr operator value_type() const noexcept
  {
    return __get();
  }

  _CCCL_TEMPLATE(typename _Up)
  _CCCL_REQUIRES(::cuda::std::is_assignable_v<value_type&, _Up&&>)
  _CCCL_API __simd_reference operator=(_Up&& __v) && noexcept
  {
    __set(static_cast<value_type>(::cuda::std::forward<_Up>(__v)));
    return {__s_, __idx_};
  }

  template <typename _Tp1, typename _Storage1, typename _Vp1>
  friend _CCCL_API void swap(__simd_reference<_Tp1, _Storage1, _Vp1>&& __a,
                             __simd_reference<_Tp1, _Storage1, _Vp1>&& __b) noexcept;

  template <typename _Tp1, typename _Storage1, typename _Vp1>
  friend _CCCL_API void swap(_Vp1& __a, __simd_reference<_Tp1, _Storage1, _Vp1>&& __b) noexcept;

  template <typename _Tp1, typename _Storage1, typename _Vp1>
  friend _CCCL_API void swap(__simd_reference<_Tp1, _Storage1, _Vp1>&& __a, _Vp1& __b) noexcept;

  template <typename _Up, typename = decltype(::cuda::std::declval<value_type&>() += ::cuda::std::declval<_Up>())>
  _CCCL_API __simd_reference operator+=(_Up&& __v) && noexcept
  {
    __set(__get() + static_cast<value_type>(::cuda::std::forward<_Up>(__v)));
    return {__s_, __idx_};
  }

  template <typename _Up, typename = decltype(::cuda::std::declval<value_type&>() -= ::cuda::std::declval<_Up>())>
  _CCCL_API __simd_reference operator-=(_Up&& __v) && noexcept
  {
    __set(__get() - static_cast<value_type>(::cuda::std::forward<_Up>(__v)));
    return {__s_, __idx_};
  }

  template <typename _Up, typename = decltype(::cuda::std::declval<value_type&>() *= ::cuda::std::declval<_Up>())>
  _CCCL_API __simd_reference operator*=(_Up&& __v) && noexcept
  {
    __set(__get() * static_cast<value_type>(::cuda::std::forward<_Up>(__v)));
    return {__s_, __idx_};
  }

  template <typename _Up, typename = decltype(::cuda::std::declval<value_type&>() /= ::cuda::std::declval<_Up>())>
  _CCCL_API __simd_reference operator/=(_Up&& __v) && noexcept
  {
    __set(__get() / static_cast<value_type>(::cuda::std::forward<_Up>(__v)));
    return {__s_, __idx_};
  }

  template <typename _Up, typename = decltype(::cuda::std::declval<value_type&>() %= ::cuda::std::declval<_Up>())>
  _CCCL_API __simd_reference operator%=(_Up&& __v) && noexcept
  {
    __set(__get() % static_cast<value_type>(::cuda::std::forward<_Up>(__v)));
    return {__s_, __idx_};
  }

  template <typename _Up, typename = decltype(::cuda::std::declval<value_type&>() &= ::cuda::std::declval<_Up>())>
  _CCCL_API __simd_reference operator&=(_Up&& __v) && noexcept
  {
    __set(__get() & static_cast<value_type>(::cuda::std::forward<_Up>(__v)));
    return {__s_, __idx_};
  }

  template <typename _Up, typename = decltype(::cuda::std::declval<value_type&>() |= ::cuda::std::declval<_Up>())>
  _CCCL_API __simd_reference operator|=(_Up&& __v) && noexcept
  {
    __set(__get() | static_cast<value_type>(::cuda::std::forward<_Up>(__v)));
    return {__s_, __idx_};
  }

  template <typename _Up, typename = decltype(::cuda::std::declval<value_type&>() ^= ::cuda::std::declval<_Up>())>
  _CCCL_API __simd_reference operator^=(_Up&& __v) && noexcept
  {
    __set(__get() ^ static_cast<value_type>(::cuda::std::forward<_Up>(__v)));
    return {__s_, __idx_};
  }

  template <typename _Up, typename = decltype(::cuda::std::declval<value_type&>() <<= ::cuda::std::declval<_Up>())>
  _CCCL_API __simd_reference operator<<=(_Up&& __v) && noexcept
  {
    __set(__get() << static_cast<value_type>(::cuda::std::forward<_Up>(__v)));
    return {__s_, __idx_};
  }

  template <typename _Up, typename = decltype(::cuda::std::declval<value_type&>() >>= ::cuda::std::declval<_Up>())>
  _CCCL_API __simd_reference operator>>=(_Up&& __v) && noexcept
  {
    __set(__get() >> static_cast<value_type>(::cuda::std::forward<_Up>(__v)));
    return {__s_, __idx_};
  }

  _CCCL_API constexpr __simd_reference operator++() && noexcept
  {
    __set(__get() + 1);
    return {__s_, __idx_};
  }

  _CCCL_API constexpr value_type operator++(int) && noexcept
  {
    auto __r = __get();
    __set(__get() + 1);
    return __r;
  }

  _CCCL_API constexpr __simd_reference operator--() && noexcept
  {
    __set(__get() - 1);
    return {__s_, __idx_};
  }

  _CCCL_API constexpr value_type operator--(int) && noexcept
  {
    auto __r = __get();
    __set(__get() - 1);
    return __r;
  }
};

template <typename _Tp, typename _Storage, typename _Vp>
_CCCL_API void swap(__simd_reference<_Tp, _Storage, _Vp>&& __a, __simd_reference<_Tp, _Storage, _Vp>&& __b) noexcept
{
  _Vp __tmp(::cuda::std::move(__a));
  ::cuda::std::move(__a) = ::cuda::std::move(__b);
  ::cuda::std::move(__b) = ::cuda::std::move(__tmp);
}

template <typename _Tp, typename _Storage, typename _Vp>
_CCCL_API void swap(_Vp& __a, __simd_reference<_Tp, _Storage, _Vp>&& __b) noexcept
{
  _Vp __tmp(::cuda::std::move(__a));
  __a                    = ::cuda::std::move(__b);
  ::cuda::std::move(__b) = ::cuda::std::move(__tmp);
}

template <typename _Tp, typename _Storage, typename _Vp>
_CCCL_API void swap(__simd_reference<_Tp, _Storage, _Vp>&& __a, _Vp& __b) noexcept
{
  _Vp __tmp(::cuda::std::move(__a));
  ::cuda::std::move(__a) = ::cuda::std::move(__b);
  __b                    = ::cuda::std::move(__tmp);
}
} // namespace cuda::experimental::datapar

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___SIMD_REFERENCE_H
