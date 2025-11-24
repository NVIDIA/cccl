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

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_cvref.h>

#include <cuda/experimental/__simd/declaration.h>
#include <cuda/experimental/__simd/fixed_size_impl.h>
#include <cuda/experimental/__simd/reference.h>
#include <cuda/experimental/__simd/scalar_impl.h>
#include <cuda/experimental/__simd/simd_mask.h>
#include <cuda/experimental/__simd/traits.h>
#include <cuda/experimental/__simd/utility.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::datapar
{
template <typename _Tp, typename _Abi>
class basic_simd : public __simd_operations<_Tp, _Abi>
{
  static_assert(is_abi_tag_v<_Abi>, "basic_simd requires a valid ABI tag");

  using _Impl    = __simd_operations<_Tp, _Abi>;
  using _Storage = typename _Impl::_SimdStorage;

  _Storage __s_;

  template <typename _Up>
  static constexpr bool __is_value_preserving_broadcast =
    (__is_vectorizable_v<::cuda::std::remove_cvref_t<_Up>> && __is_non_narrowing_convertible_v<_Up, _Tp>)
    || (!__is_vectorizable_v<::cuda::std::remove_cvref_t<_Up>> && ::cuda::std::is_convertible_v<_Up, _Tp>);

public:
  using value_type = _Tp;
  using reference  = __simd_reference<_Storage, value_type>;
  using abi_type   = _Abi;
  using mask_type  = basic_simd_mask<value_type, abi_type>;

  _CCCL_TEMPLATE(typename _Up, typename _Ap)
  _CCCL_REQUIRES(::cuda::std::is_same_v<_Up, value_type> _CCCL_AND ::cuda::std::is_same_v<_Ap, abi_type>)
  _CCCL_API explicit operator basic_simd_mask<_Up, _Ap>() const noexcept
  {
    basic_simd_mask<_Up, _Ap> __result;
    for (::cuda::std::size_t __i = 0; __i < size(); ++__i)
    {
      __result[__i] = static_cast<bool>((*this)[__i]);
    }
    return __result;
  }

  [[nodiscard]] _CCCL_API static constexpr ::cuda::std::size_t size() noexcept
  {
    return simd_size_v<value_type, abi_type>;
  }

  _CCCL_HIDE_FROM_ABI basic_simd() noexcept = default;

  struct __storage_tag_t
  {};
  static constexpr __storage_tag_t __storage_tag{};

  _CCCL_API explicit operator _Storage() const
  {
    return __s_;
  }

  _CCCL_API explicit basic_simd(const _Storage& __s, __storage_tag_t)
      : __s_{__s}
  {}

  _CCCL_TEMPLATE(typename _Up)
  _CCCL_REQUIRES(__can_broadcast_v<value_type, ::cuda::std::remove_cvref_t<_Up>>&& __is_value_preserving_broadcast<_Up>)
  _CCCL_API constexpr basic_simd(_Up&& __v) noexcept
      : __s_{_Impl::__broadcast(static_cast<value_type>(__v))}
  {}

  _CCCL_TEMPLATE(typename _Up)
  _CCCL_REQUIRES((__can_broadcast_v<value_type, ::cuda::std::remove_cvref_t<_Up>>
                  && !__is_value_preserving_broadcast<_Up>) )
  _CCCL_API constexpr explicit basic_simd(_Up&& __v) noexcept
      : __s_{_Impl::__broadcast(static_cast<value_type>(__v))}
  {}

  _CCCL_TEMPLATE(typename _Up)
  _CCCL_REQUIRES((!::cuda::std::is_same_v<_Up, _Tp> && __is_non_narrowing_convertible_v<_Up, value_type>) )
  _CCCL_API basic_simd(const basic_simd<_Up, abi_type>& __v) noexcept
  {
    for (::cuda::std::size_t __i = 0; __i < size(); __i++)
    {
      (*this)[__i] = static_cast<value_type>(__v[__i]);
    }
  }

  _CCCL_TEMPLATE(typename _Up)
  _CCCL_REQUIRES((!::cuda::std::is_same_v<_Up, _Tp> && !__is_non_narrowing_convertible_v<_Up, value_type>
                  && ::cuda::std::is_convertible_v<_Up, value_type>) )
  _CCCL_API explicit basic_simd(const basic_simd<_Up, abi_type>& __v) noexcept
  {
    for (::cuda::std::size_t __i = 0; __i < size(); __i++)
    {
      (*this)[__i] = static_cast<value_type>(__v[__i]);
    }
  }

  _CCCL_TEMPLATE(typename _Generator)
  _CCCL_REQUIRES(__can_generate_v<value_type, _Generator, size()>)
  _CCCL_API explicit basic_simd(_Generator&& __g)
      : __s_(_Impl::__generate(__g))
  {}

  _CCCL_TEMPLATE(typename _Up, typename _Flags = element_aligned_tag)
  _CCCL_REQUIRES(__is_vectorizable_v<_Up> _CCCL_AND is_simd_flag_type_v<_Flags>)
  _CCCL_API explicit basic_simd(const _Up* __mem, _Flags = {}) noexcept
  {
    _Impl::__load(__s_, _Flags::template __apply<basic_simd>(__mem));
  }

  _CCCL_TEMPLATE(typename _Up, typename _Flags = element_aligned_tag)
  _CCCL_REQUIRES(__is_vectorizable_v<_Up> _CCCL_AND is_simd_flag_type_v<_Flags>)
  _CCCL_API void copy_from(const _Up* __mem, _Flags = {}) noexcept
  {
    _Impl::__load(__s_, _Flags::template __apply<basic_simd>(__mem));
  }

  _CCCL_TEMPLATE(typename _Up, typename _Flags = element_aligned_tag)
  _CCCL_REQUIRES(__is_vectorizable_v<_Up> _CCCL_AND is_simd_flag_type_v<_Flags>)
  _CCCL_API void copy_to(_Up* __mem, _Flags = {}) const noexcept
  {
    _Impl::__store(__s_, _Flags::template __apply<basic_simd>(__mem));
  }

  _CCCL_API reference operator[](::cuda::std::size_t __i) noexcept
  {
    return reference(__s_, __i);
  }

  _CCCL_API value_type operator[](::cuda::std::size_t __i) const noexcept
  {
    return __s_.__get(__i);
  }

  _CCCL_API basic_simd& operator++() noexcept
  {
    _Impl::__increment(__s_);
    return *this;
  }

  _CCCL_API basic_simd operator++(int) noexcept
  {
    const basic_simd __r = *this;
    _Impl::__increment(__s_);
    return __r;
  }

  _CCCL_API basic_simd& operator--() noexcept
  {
    _Impl::__decrement(__s_);
    return *this;
  }

  _CCCL_API basic_simd operator--(int) noexcept
  {
    const basic_simd __r = *this;
    _Impl::__decrement(__s_);
    return __r;
  }

  [[nodiscard]] _CCCL_API basic_simd operator+() const noexcept
  {
    return *this;
  }

  [[nodiscard]] _CCCL_API basic_simd operator-() const noexcept
  {
    return basic_simd{_Impl::__unary_minus(__s_), __storage_tag};
  }

  _CCCL_API constexpr friend basic_simd& operator+=(basic_simd& __lhs, const basic_simd& __rhs) noexcept
  {
    return __lhs = __lhs + __rhs;
  }

  _CCCL_API constexpr friend basic_simd& operator-=(basic_simd& __lhs, const basic_simd& __rhs) noexcept
  {
    return __lhs = __lhs - __rhs;
  }

  _CCCL_API constexpr friend basic_simd& operator*=(basic_simd& __lhs, const basic_simd& __rhs) noexcept
  {
    return __lhs = __lhs * __rhs;
  }

  _CCCL_API constexpr friend basic_simd& operator/=(basic_simd& __lhs, const basic_simd& __rhs) noexcept
  {
    return __lhs = __lhs / __rhs;
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(::cuda::std::is_integral_v<_Tp>)
  _CCCL_API constexpr friend basic_simd& operator%=(basic_simd& __lhs, const basic_simd& __rhs) noexcept
  {
    return __lhs = __lhs % __rhs;
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(::cuda::std::is_integral_v<_Tp>)
  _CCCL_API constexpr friend basic_simd& operator&=(basic_simd& __lhs, const basic_simd& __rhs) noexcept
  {
    return __lhs = __lhs & __rhs;
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(::cuda::std::is_integral_v<_Tp>)
  _CCCL_API constexpr friend basic_simd& operator|=(basic_simd& __lhs, const basic_simd& __rhs) noexcept
  {
    return __lhs = __lhs | __rhs;
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(::cuda::std::is_integral_v<_Tp>)
  _CCCL_API constexpr friend basic_simd& operator^=(basic_simd& __lhs, const basic_simd& __rhs) noexcept
  {
    return __lhs = __lhs ^ __rhs;
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(::cuda::std::is_integral_v<_Tp>)
  _CCCL_API constexpr friend basic_simd& operator<<=(basic_simd& __lhs, const basic_simd& __rhs) noexcept
  {
    return __lhs = __lhs << __rhs;
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(::cuda::std::is_integral_v<_Tp>)
  _CCCL_API constexpr friend basic_simd& operator>>=(basic_simd& __lhs, const basic_simd& __rhs) noexcept
  {
    return __lhs = __lhs >> __rhs;
  }

  [[nodiscard]] _CCCL_API constexpr friend basic_simd
  operator+(const basic_simd& __lhs, const basic_simd& __rhs) noexcept
  {
    return basic_simd{_Impl::__plus(__lhs.__s_, __rhs.__s_), __storage_tag};
  }

  _CCCL_TEMPLATE(typename _Up)
  _CCCL_REQUIRES(__can_broadcast_v<value_type, _Up>)
  [[nodiscard]] _CCCL_API constexpr friend basic_simd operator+(const basic_simd& __lhs, _Up&& __rhs) noexcept
  {
    return __lhs + basic_simd(static_cast<value_type>(__rhs));
  }

  _CCCL_TEMPLATE(typename _Up)
  _CCCL_REQUIRES(__can_broadcast_v<value_type, _Up>)
  [[nodiscard]] _CCCL_API constexpr friend basic_simd operator+(_Up&& __lhs, const basic_simd& __rhs) noexcept
  {
    return basic_simd(static_cast<value_type>(__lhs)) + __rhs;
  }

  [[nodiscard]] _CCCL_API constexpr friend basic_simd
  operator-(const basic_simd& __lhs, const basic_simd& __rhs) noexcept
  {
    return basic_simd{_Impl::__minus(__lhs.__s_, __rhs.__s_), __storage_tag};
  }

  _CCCL_TEMPLATE(typename _Up)
  _CCCL_REQUIRES(__can_broadcast_v<value_type, _Up>)
  [[nodiscard]] _CCCL_API constexpr friend basic_simd operator-(const basic_simd& __lhs, _Up&& __rhs) noexcept
  {
    return __lhs - basic_simd(static_cast<value_type>(__rhs));
  }

  _CCCL_TEMPLATE(typename _Up)
  _CCCL_REQUIRES(__can_broadcast_v<value_type, _Up>)
  [[nodiscard]] _CCCL_API constexpr friend basic_simd operator-(_Up&& __lhs, const basic_simd& __rhs) noexcept
  {
    return basic_simd(static_cast<value_type>(__lhs)) - __rhs;
  }

  [[nodiscard]] _CCCL_API constexpr friend basic_simd
  operator*(const basic_simd& __lhs, const basic_simd& __rhs) noexcept
  {
    return basic_simd{_Impl::__multiplies(__lhs.__s_, __rhs.__s_), __storage_tag};
  }

  _CCCL_TEMPLATE(typename _Up)
  _CCCL_REQUIRES(__can_broadcast_v<value_type, _Up>)
  [[nodiscard]] _CCCL_API constexpr friend basic_simd operator*(const basic_simd& __lhs, _Up&& __rhs) noexcept
  {
    return __lhs * basic_simd(static_cast<value_type>(__rhs));
  }

  _CCCL_TEMPLATE(typename _Up)
  _CCCL_REQUIRES(__can_broadcast_v<value_type, _Up>)
  [[nodiscard]] _CCCL_API constexpr friend basic_simd operator*(_Up&& __lhs, const basic_simd& __rhs) noexcept
  {
    return basic_simd(static_cast<value_type>(__lhs)) * __rhs;
  }

  [[nodiscard]] _CCCL_API constexpr friend basic_simd
  operator/(const basic_simd& __lhs, const basic_simd& __rhs) noexcept
  {
    return basic_simd{_Impl::__divides(__lhs.__s_, __rhs.__s_), __storage_tag};
  }

  _CCCL_TEMPLATE(typename _Up)
  _CCCL_REQUIRES(__can_broadcast_v<value_type, _Up>)
  [[nodiscard]] _CCCL_API constexpr friend basic_simd operator/(const basic_simd& __lhs, _Up&& __rhs) noexcept
  {
    return __lhs / basic_simd(static_cast<value_type>(__rhs));
  }

  _CCCL_TEMPLATE(typename _Up)
  _CCCL_REQUIRES(__can_broadcast_v<value_type, _Up>)
  [[nodiscard]] _CCCL_API constexpr friend basic_simd operator/(_Up&& __lhs, const basic_simd& __rhs) noexcept
  {
    return basic_simd(static_cast<value_type>(__lhs)) / __rhs;
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(::cuda::std::is_integral_v<_Tp>)
  [[nodiscard]] _CCCL_API constexpr friend basic_simd
  operator%(const basic_simd& __lhs, const basic_simd& __rhs) noexcept
  {
    return basic_simd{_Impl::__modulo(__lhs.__s_, __rhs.__s_), __storage_tag};
  }

  _CCCL_TEMPLATE(typename _Up)
  _CCCL_REQUIRES(::cuda::std::is_integral_v<_Tp> _CCCL_AND __can_broadcast_v<value_type, _Up>)
  [[nodiscard]] _CCCL_API constexpr friend basic_simd operator%(const basic_simd& __lhs, _Up&& __rhs) noexcept
  {
    return __lhs % basic_simd(static_cast<value_type>(__rhs));
  }

  _CCCL_TEMPLATE(typename _Up)
  _CCCL_REQUIRES(::cuda::std::is_integral_v<_Tp> _CCCL_AND __can_broadcast_v<value_type, _Up>)
  [[nodiscard]] _CCCL_API constexpr friend basic_simd operator%(_Up&& __lhs, const basic_simd& __rhs) noexcept
  {
    return basic_simd(static_cast<value_type>(__lhs)) % __rhs;
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(::cuda::std::is_integral_v<_Tp>)
  [[nodiscard]] _CCCL_API constexpr friend basic_simd
  operator&(const basic_simd& __lhs, const basic_simd& __rhs) noexcept
  {
    return basic_simd{_Impl::__bitwise_and(__lhs.__s_, __rhs.__s_), __storage_tag};
  }

  _CCCL_TEMPLATE(typename _Up)
  _CCCL_REQUIRES(::cuda::std::is_integral_v<_Tp> _CCCL_AND __can_broadcast_v<value_type, _Up>)
  [[nodiscard]] _CCCL_API constexpr friend basic_simd operator&(const basic_simd& __lhs, _Up&& __rhs) noexcept
  {
    return __lhs & basic_simd(static_cast<value_type>(__rhs));
  }

  _CCCL_TEMPLATE(typename _Up)
  _CCCL_REQUIRES(::cuda::std::is_integral_v<_Tp> _CCCL_AND __can_broadcast_v<value_type, _Up>)
  [[nodiscard]] _CCCL_API constexpr friend basic_simd operator&(_Up&& __lhs, const basic_simd& __rhs) noexcept
  {
    return basic_simd(static_cast<value_type>(__lhs)) & __rhs;
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(::cuda::std::is_integral_v<_Tp>)
  [[nodiscard]] _CCCL_API constexpr friend basic_simd
  operator|(const basic_simd& __lhs, const basic_simd& __rhs) noexcept
  {
    return basic_simd{_Impl::__bitwise_or(__lhs.__s_, __rhs.__s_), __storage_tag};
  }

  _CCCL_TEMPLATE(typename _Up)
  _CCCL_REQUIRES(::cuda::std::is_integral_v<_Tp> _CCCL_AND __can_broadcast_v<value_type, _Up>)
  [[nodiscard]] _CCCL_API constexpr friend basic_simd operator|(const basic_simd& __lhs, _Up&& __rhs) noexcept
  {
    return __lhs | basic_simd(static_cast<value_type>(__rhs));
  }

  _CCCL_TEMPLATE(typename _Up)
  _CCCL_REQUIRES(::cuda::std::is_integral_v<_Tp> _CCCL_AND __can_broadcast_v<value_type, _Up>)
  [[nodiscard]] _CCCL_API constexpr friend basic_simd operator|(_Up&& __lhs, const basic_simd& __rhs) noexcept
  {
    return basic_simd(static_cast<value_type>(__lhs)) | __rhs;
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(::cuda::std::is_integral_v<_Tp>)
  [[nodiscard]] _CCCL_API constexpr friend basic_simd
  operator^(const basic_simd& __lhs, const basic_simd& __rhs) noexcept
  {
    return basic_simd{_Impl::__bitwise_xor(__lhs.__s_, __rhs.__s_), __storage_tag};
  }

  _CCCL_TEMPLATE(typename _Up)
  _CCCL_REQUIRES(::cuda::std::is_integral_v<_Tp> _CCCL_AND __can_broadcast_v<value_type, _Up>)
  [[nodiscard]] _CCCL_API constexpr friend basic_simd operator^(const basic_simd& __lhs, _Up&& __rhs) noexcept
  {
    return __lhs ^ basic_simd(static_cast<value_type>(__rhs));
  }

  _CCCL_TEMPLATE(typename _Up)
  _CCCL_REQUIRES(::cuda::std::is_integral_v<_Tp> _CCCL_AND __can_broadcast_v<value_type, _Up>)
  [[nodiscard]] _CCCL_API constexpr friend basic_simd operator^(_Up&& __lhs, const basic_simd& __rhs) noexcept
  {
    return basic_simd(static_cast<value_type>(__lhs)) ^ __rhs;
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(::cuda::std::is_integral_v<_Tp>)
  [[nodiscard]] _CCCL_API constexpr friend basic_simd
  operator<<(const basic_simd& __lhs, const basic_simd& __rhs) noexcept
  {
    return basic_simd{_Impl::__shift_left(__lhs.__s_, __rhs.__s_), __storage_tag};
  }

  _CCCL_TEMPLATE(typename _Up)
  _CCCL_REQUIRES(::cuda::std::is_integral_v<_Tp> _CCCL_AND __can_broadcast_v<value_type, _Up>)
  [[nodiscard]] _CCCL_API constexpr friend basic_simd operator<<(const basic_simd& __lhs, _Up&& __rhs) noexcept
  {
    return __lhs << basic_simd(static_cast<value_type>(__rhs));
  }

  _CCCL_TEMPLATE(typename _Up)
  _CCCL_REQUIRES(::cuda::std::is_integral_v<_Tp> _CCCL_AND __can_broadcast_v<value_type, _Up>)
  [[nodiscard]] _CCCL_API constexpr friend basic_simd operator<<(_Up&& __lhs, const basic_simd& __rhs) noexcept
  {
    return basic_simd(static_cast<value_type>(__lhs)) << __rhs;
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(::cuda::std::is_integral_v<_Tp>)
  [[nodiscard]] _CCCL_API constexpr friend basic_simd
  operator>>(const basic_simd& __lhs, const basic_simd& __rhs) noexcept
  {
    return basic_simd{_Impl::__shift_right(__lhs.__s_, __rhs.__s_), __storage_tag};
  }

  _CCCL_TEMPLATE(typename _Up)
  _CCCL_REQUIRES(::cuda::std::is_integral_v<_Tp> _CCCL_AND __can_broadcast_v<value_type, _Up>)
  [[nodiscard]] _CCCL_API constexpr friend basic_simd operator>>(const basic_simd& __lhs, _Up&& __rhs) noexcept
  {
    return __lhs >> basic_simd(static_cast<value_type>(__rhs));
  }

  _CCCL_TEMPLATE(typename _Up)
  _CCCL_REQUIRES(::cuda::std::is_integral_v<_Tp> _CCCL_AND __can_broadcast_v<value_type, _Up>)
  [[nodiscard]] _CCCL_API constexpr friend basic_simd operator>>(_Up&& __lhs, const basic_simd& __rhs) noexcept
  {
    return basic_simd(static_cast<value_type>(__lhs)) >> __rhs;
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(::cuda::std::is_integral_v<_Tp>)
  [[nodiscard]] _CCCL_API constexpr basic_simd operator~() const noexcept
  {
    return basic_simd{_Impl::__bitwise_not(__s_), __storage_tag};
  }

  [[nodiscard]] _CCCL_API constexpr friend mask_type
  operator==(const basic_simd& __lhs, const basic_simd& __rhs) noexcept
  {
    return mask_type{_Impl::__equal_to(__lhs.__s_, __rhs.__s_), mask_type::__storage_tag};
  }

  _CCCL_TEMPLATE(typename _Up)
  _CCCL_REQUIRES(__can_broadcast_v<value_type, _Up>)
  [[nodiscard]] _CCCL_API constexpr friend mask_type operator==(const basic_simd& __lhs, _Up&& __rhs) noexcept
  {
    return __lhs == basic_simd(static_cast<value_type>(__rhs));
  }

  _CCCL_TEMPLATE(typename _Up)
  _CCCL_REQUIRES(__can_broadcast_v<value_type, _Up>)
  [[nodiscard]] _CCCL_API constexpr friend mask_type operator==(_Up&& __lhs, const basic_simd& __rhs) noexcept
  {
    return basic_simd(static_cast<value_type>(__lhs)) == __rhs;
  }

  [[nodiscard]] _CCCL_API constexpr friend mask_type
  operator!=(const basic_simd& __lhs, const basic_simd& __rhs) noexcept
  {
    return mask_type{_Impl::__not_equal_to(__lhs.__s_, __rhs.__s_), __storage_tag};
  }

  _CCCL_TEMPLATE(typename _Up)
  _CCCL_REQUIRES(__can_broadcast_v<value_type, _Up>)
  [[nodiscard]] _CCCL_API constexpr friend mask_type operator!=(const basic_simd& __lhs, _Up&& __rhs) noexcept
  {
    return __lhs != basic_simd(static_cast<value_type>(__rhs));
  }

  _CCCL_TEMPLATE(typename _Up)
  _CCCL_REQUIRES(__can_broadcast_v<value_type, _Up>)
  [[nodiscard]] _CCCL_API constexpr friend mask_type operator!=(_Up&& __lhs, const basic_simd& __rhs) noexcept
  {
    return basic_simd(static_cast<value_type>(__lhs)) != __rhs;
  }

  [[nodiscard]] _CCCL_API constexpr friend mask_type operator<(const basic_simd& __lhs, const basic_simd& __rhs) noexcept
  {
    return mask_type{_Impl::__less(__lhs.__s_, __rhs.__s_), mask_type::__storage_tag};
  }

  _CCCL_TEMPLATE(typename _Up)
  _CCCL_REQUIRES(__can_broadcast_v<value_type, _Up>)
  [[nodiscard]] _CCCL_API constexpr friend mask_type operator<(const basic_simd& __lhs, _Up&& __rhs) noexcept
  {
    return __lhs < basic_simd(static_cast<value_type>(__rhs));
  }

  _CCCL_TEMPLATE(typename _Up)
  _CCCL_REQUIRES(__can_broadcast_v<value_type, _Up>)
  [[nodiscard]] _CCCL_API constexpr friend mask_type operator<(_Up&& __lhs, const basic_simd& __rhs) noexcept
  {
    return basic_simd(static_cast<value_type>(__lhs)) < __rhs;
  }

  [[nodiscard]] _CCCL_API constexpr friend mask_type
  operator<=(const basic_simd& __lhs, const basic_simd& __rhs) noexcept
  {
    return mask_type{_Impl::__less_equal(__lhs.__s_, __rhs.__s_), mask_type::__storage_tag};
  }

  _CCCL_TEMPLATE(typename _Up)
  _CCCL_REQUIRES(__can_broadcast_v<value_type, _Up>)
  [[nodiscard]] _CCCL_API constexpr friend mask_type operator<=(const basic_simd& __lhs, _Up&& __rhs) noexcept
  {
    return __lhs <= basic_simd(static_cast<value_type>(__rhs));
  }

  _CCCL_TEMPLATE(typename _Up)
  _CCCL_REQUIRES(__can_broadcast_v<value_type, _Up>)
  [[nodiscard]] _CCCL_API constexpr friend mask_type operator<=(_Up&& __lhs, const basic_simd& __rhs) noexcept
  {
    return basic_simd(static_cast<value_type>(__lhs)) <= __rhs;
  }

  [[nodiscard]] _CCCL_API constexpr friend mask_type operator>(const basic_simd& __lhs, const basic_simd& __rhs) noexcept
  {
    return mask_type{_Impl::__greater(__lhs.__s_, __rhs.__s_), mask_type::__storage_tag};
  }

  _CCCL_TEMPLATE(typename _Up)
  _CCCL_REQUIRES(__can_broadcast_v<value_type, _Up>)
  [[nodiscard]] _CCCL_API constexpr friend mask_type operator>(const basic_simd& __lhs, _Up&& __rhs) noexcept
  {
    return __lhs > basic_simd(static_cast<value_type>(__rhs));
  }

  _CCCL_TEMPLATE(typename _Up)
  _CCCL_REQUIRES(__can_broadcast_v<value_type, _Up>)
  [[nodiscard]] _CCCL_API constexpr friend mask_type operator>(_Up&& __lhs, const basic_simd& __rhs) noexcept
  {
    return basic_simd(static_cast<value_type>(__lhs)) > __rhs;
  }

  [[nodiscard]] _CCCL_API constexpr friend mask_type
  operator>=(const basic_simd& __lhs, const basic_simd& __rhs) noexcept
  {
    return mask_type{_Impl::__greater_equal(__lhs.__s_, __rhs.__s_), mask_type::__storage_tag};
  }

  _CCCL_TEMPLATE(typename _Up)
  _CCCL_REQUIRES(__can_broadcast_v<value_type, _Up>)
  [[nodiscard]] _CCCL_API constexpr friend mask_type operator>=(const basic_simd& __lhs, _Up&& __rhs) noexcept
  {
    return __lhs >= basic_simd(static_cast<value_type>(__rhs));
  }

  _CCCL_TEMPLATE(typename _Up)
  _CCCL_REQUIRES(__can_broadcast_v<value_type, _Up>)
  [[nodiscard]] _CCCL_API constexpr friend mask_type operator>=(_Up&& __lhs, const basic_simd& __rhs) noexcept
  {
    return basic_simd(static_cast<value_type>(__lhs)) >= __rhs;
  }
};

template <typename _Tp, int _Np>
using fixed_size_simd = simd<_Tp, _Np>;
} // namespace cuda::experimental::datapar

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___SIMD_SIMD_H
