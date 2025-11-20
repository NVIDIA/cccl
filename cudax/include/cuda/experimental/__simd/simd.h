//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___SIMD_SIMD_H
#define _CUDAX___SIMD_SIMD_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/forward.h>

#include <cuda/experimental/__simd/declaration.h>
#include <cuda/experimental/__simd/reference.h>
#include <cuda/experimental/__simd/traits.h>
#include <cuda/experimental/__simd/utility.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::datapar
{
template <typename _Tp, int _Np>
class simd : public __simd_operations<_Tp, simd_abi::fixed_size<_Np>>
{
  using _Impl    = __simd_operations<_Tp, simd_abi::fixed_size<_Np>>;
  using _Storage = typename _Impl::_SimdStorage;

  _Storage __s_;

public:
  using value_type = _Tp;
  using reference  = __simd_reference<_Tp, _Storage, value_type>;
  using mask_type  = simd_mask<_Tp, _Np>;
  using abi_type   = simd_abi::fixed_size<_Np>;

  [[nodiscard]] _CCCL_API static constexpr ::cuda::std::size_t size() noexcept
  {
    return simd_size_v<value_type, abi_type>;
  }

  _CCCL_API simd() noexcept = default;

  struct __storage_tag_t
  {};
  static constexpr __storage_tag_t __storage_tag{};

  _CCCL_API explicit operator _Storage() const
  {
    return __s_;
  }

  _CCCL_API explicit simd(const _Storage& __s, __storage_tag_t)
      : __s_(__s)
  {}

  _CCCL_TEMPLATE(typename _Up)
  _CCCL_REQUIRES(__can_broadcast_v<value_type, ::cuda::std::remove_cvref_t<_Up>>)
  _CCCL_API simd(_Up&& __v) noexcept
      : __s_(_Impl::__broadcast(static_cast<value_type>(::cuda::std::forward<_Up>(__v))))
  {}

  _CCCL_TEMPLATE(typename _Up)
  _CCCL_REQUIRES(!::cuda::std::is_same_v<_Up, _Tp> && ::cuda::std::is_same_v<abi_type, simd_abi::fixed_size<size()>>
                 && __is_non_narrowing_convertible_v<_Up, value_type>)
  _CCCL_API simd(const simd<_Up, size()>& __v) noexcept
  {
    for (::cuda::std::size_t __i = 0; __i < size(); __i++)
    {
      (*this)[__i] = static_cast<value_type>(__v[__i]);
    }
  }

  _CCCL_TEMPLATE(typename _Generator)
  _CCCL_REQUIRES(__can_generate_v<value_type, _Generator, size()>)
  _CCCL_API explicit simd(_Generator&& __g) noexcept
      : __s_(_Impl::__generate(::cuda::std::forward<_Generator>(__g)))
  {}

  _CCCL_TEMPLATE(typename _Up, typename _Flags)
  _CCCL_REQUIRES(__is_vectorizable_v<_Up>&& is_simd_flag_type_v<_Flags>)
  _CCCL_API simd(const _Up* __mem, _Flags) noexcept
  {
    _Impl::__load(__s_, _Flags::template __apply<simd>(__mem));
  }

  _CCCL_TEMPLATE(typename _Up, typename _Flags)
  _CCCL_REQUIRES(__is_vectorizable_v<_Up>&& is_simd_flag_type_v<_Flags>)
  _CCCL_API void copy_from(const _Up* __mem, _Flags) noexcept
  {
    _Impl::__load(__s_, _Flags::template __apply<simd>(__mem));
  }

  _CCCL_TEMPLATE(typename _Up, typename _Flags)
  _CCCL_REQUIRES(__is_vectorizable_v<_Up>&& is_simd_flag_type_v<_Flags>)
  _CCCL_API void copy_to(_Up* __mem, _Flags) const noexcept
  {
    _Impl::__store(__s_, _Flags::template __apply<simd>(__mem));
  }

  _CCCL_API reference operator[](::cuda::std::size_t __i) noexcept
  {
    return reference(__s_, __i);
  }

  _CCCL_API value_type operator[](::cuda::std::size_t __i) const noexcept
  {
    return __s_.__get(__i);
  }

  _CCCL_API simd& operator++() noexcept
  {
    _Impl::__increment(__s_);
    return *this;
  }

  _CCCL_API simd operator++(int) noexcept
  {
    const simd __r = *this;
    _Impl::__increment(__s_);
    return __r;
  }

  _CCCL_API simd& operator--() noexcept
  {
    _Impl::__decrement(__s_);
    return *this;
  }

  _CCCL_API simd operator--(int) noexcept
  {
    const simd __r = *this;
    _Impl::__decrement(__s_);
    return __r;
  }

  [[nodiscard]] _CCCL_API simd operator+() const noexcept
  {
    return *this;
  }

  [[nodiscard]] _CCCL_API simd operator-() const noexcept
  {
    return {_Impl::__unary_minus(__s_), __storage_tag};
  }

  _CCCL_API constexpr friend simd& operator+=(simd& __lhs, const simd& __rhs) noexcept
  {
    return __lhs = {__lhs + __rhs, __storage_tag};
  }

  _CCCL_API constexpr friend simd& operator-=(simd& __lhs, const simd& __rhs) noexcept
  {
    return __lhs = {__lhs - __rhs, __storage_tag};
  }

  _CCCL_API constexpr friend simd& operator*=(simd& __lhs, const simd& __rhs) noexcept
  {
    return __lhs = {__lhs * __rhs, __storage_tag};
  }

  _CCCL_API constexpr friend simd& operator/=(simd& __lhs, const simd& __rhs) noexcept
  {
    return __lhs = {__lhs / __rhs, __storage_tag};
  }

  [[nodiscard]] _CCCL_API constexpr friend simd operator+(const simd& __lhs, const simd& __rhs) noexcept
  {
    return {_Impl::__plus(__lhs.__s_, __rhs.__s_), __storage_tag};
  }

  [[nodiscard]] _CCCL_API constexpr friend simd operator-(const simd& __lhs, const simd& __rhs) noexcept
  {
    return {_Impl::__minus(__lhs.__s_, __rhs.__s_), __storage_tag};
  }

  [[nodiscard]] _CCCL_API constexpr friend simd operator*(const simd& __lhs, const simd& __rhs) noexcept
  {
    return {_Impl::__multiplies(__lhs.__s_, __rhs.__s_), __storage_tag};
  }

  [[nodiscard]] _CCCL_API constexpr friend simd operator/(const simd& __lhs, const simd& __rhs) noexcept
  {
    return {_Impl::__divides(__lhs.__s_, __rhs.__s_), __storage_tag};
  }

  [[nodiscard]] _CCCL_API constexpr friend mask_type operator==(const simd& __lhs, const simd& __rhs) noexcept
  {
    return {_Impl::__equal_to(__lhs.__s_, __rhs.__s_), __storage_tag};
  }

  [[nodiscard]] _CCCL_API constexpr friend mask_type operator!=(const simd& __lhs, const simd& __rhs)
  {
    return {_Impl::__not_equal_to(__lhs.__s_, __rhs.__s_), __storage_tag};
  }

  [[nodiscard]] _CCCL_API constexpr friend mask_type operator<(const simd& __lhs, const simd& __rhs) noexcept
  {
    return {_Impl::__less(__lhs.__s_, __rhs.__s_), __storage_tag};
  }

  [[nodiscard]] _CCCL_API constexpr friend mask_type operator<=(const simd& __lhs, const simd& __rhs) noexcept
  {
    return {_Impl::__less_equal(__lhs.__s_, __rhs.__s_), __storage_tag};
  }

  [[nodiscard]] _CCCL_API constexpr friend mask_type operator>(const simd& __lhs, const simd& __rhs) noexcept
  {
    return {_Impl::__less(__lhs.__s_, __rhs.__s_), __storage_tag};
  }

  [[nodiscard]] _CCCL_API constexpr friend mask_type operator>=(const simd& __lhs, const simd& __rhs) noexcept
  {
    return {_Impl::__less_equal(__lhs.__s_, __rhs.__s_), __storage_tag};
  }
};

template <typename _Tp, typename _Abi>
inline constexpr bool is_simd_v<basic_simd<_Tp, _Abi>> = true;

template <typename _Tp>
using native_simd = simd<_Tp, simd_abi::native<_Tp>>;

template <typename _Tp, int _Np>
using fixed_size_simd = simd<_Tp, _Np>;
} // namespace cuda::experimental::datapar

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___SIMD_SIMD_H
