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

#include <cuda/experimental/__simd/concepts.h>
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
// P1928R15: basic_vec is the primary SIMD vector type (renamed from basic_simd)
template <typename _Tp, typename _Abi>
class basic_vec : public __simd_operations<_Tp, _Abi>
{
  static_assert(is_abi_tag_v<_Abi>, "basic_vec requires a valid ABI tag");

  using _Impl    = __simd_operations<_Tp, _Abi>;
  using _Storage = typename _Impl::_SimdStorage;

  _Storage __s_;

  template <typename _Up>
  static constexpr bool __is_value_preserving_broadcast =
    (__is_vectorizable_v<::cuda::std::remove_cvref_t<_Up>> && __is_non_narrowing_convertible_v<_Up, _Tp>)
    || (!__is_vectorizable_v<::cuda::std::remove_cvref_t<_Up>> && ::cuda::std::is_convertible_v<_Up, _Tp>);

  struct __storage_tag_t
  {};
  static constexpr __storage_tag_t __storage_tag{};

public:
  using value_type = _Tp;
  using reference  = __simd_reference<_Storage, value_type>;
  using mask_type  = basic_mask<sizeof(value_type), _Abi>;
  using abi_type   = _Abi;

  // TODO: add iterators
  // using iterator = simd-iterator<basic_vec>;
  // using const_iterator = simd-iterator<const basic_vec>;

  // constexpr iterator begin() noexcept { return {*this, 0}; }
  // constexpr const_iterator begin() const noexcept { return {*this, 0}; }
  // constexpr const_iterator cbegin() const noexcept { return {*this, 0}; }
  // constexpr default_sentinel_t end() const noexcept { return {}; }
  // constexpr default_sentinel_t cend() const noexcept { return {}; }

  static constexpr ::cuda::std::integral_constant<__simd_size_type, __simd_size_v<value_type, abi_type>> size{};

  _CCCL_HIDE_FROM_ABI basic_vec() noexcept = default;

  // [simd.ctor], basic_vec constructors
  // TODO: fix constraints

  _CCCL_TEMPLATE(typename _Up)
  _CCCL_REQUIRES(
    (__can_broadcast_v<value_type, ::cuda::std::remove_cvref_t<_Up>>) _CCCL_AND(__is_value_preserving_broadcast<_Up>))
  _CCCL_API constexpr basic_vec(_Up&& __v) noexcept
      : __s_{_Impl::__broadcast(static_cast<value_type>(__v))}
  {}

  _CCCL_TEMPLATE(typename _Up)
  _CCCL_REQUIRES((__can_broadcast_v<value_type, ::cuda::std::remove_cvref_t<_Up>>) //
                 _CCCL_AND(!__is_value_preserving_broadcast<_Up>))
  _CCCL_API constexpr explicit basic_vec(_Up&& __v) noexcept
      : __s_{_Impl::__broadcast(static_cast<value_type>(__v))}
  {}

  _CCCL_TEMPLATE(typename _Up)
  _CCCL_REQUIRES((!::cuda::std::is_same_v<_Up, _Tp>) _CCCL_AND(__is_non_narrowing_convertible_v<_Up, value_type>))
  _CCCL_API constexpr explicit basic_vec(const basic_vec<_Up, abi_type>& __v) noexcept
  {
    for (__simd_size_type __i = 0; __i < size; __i++)
    {
      (*this)[__i] = static_cast<value_type>(__v[__i]);
    }
  }

  _CCCL_TEMPLATE(typename _Up)
  _CCCL_REQUIRES((!::cuda::std::is_same_v<_Up, _Tp>) _CCCL_AND(!__is_non_narrowing_convertible_v<_Up, value_type>)
                   _CCCL_AND(::cuda::std::is_convertible_v<_Up, value_type>))
  _CCCL_API constexpr explicit basic_vec(const basic_vec<_Up, abi_type>& __v) noexcept
  {
    for (__simd_size_type __i = 0; __i < size; __i++)
    {
      (*this)[__i] = static_cast<value_type>(__v[__i]);
    }
  }

  _CCCL_TEMPLATE(typename _Generator)
  _CCCL_REQUIRES(__can_generate_v<value_type, _Generator, size>)
  _CCCL_API constexpr explicit basic_vec(_Generator&& __g)
      : __s_(_Impl::__generate(__g))
  {}

  // TODO: add constructors
  // template <class R, class... Flags>
  // constexpr basic_vec(R&& range, flags<Flags...> = {});

  // template <class R, class... Flags>
  // constexpr basic_vec(R&& range, const mask_type& mask, flags<Flags...> = {});

  // constexpr basic_vec(const real - type & reals, const real - type& imags = {}) noexcept;

  // [simd.subscr], basic_vec subscript operators
  _CCCL_API value_type operator[](__simd_size_type __i) const noexcept
  {
    return __s_.__get(__i);
  }

  // TODO: add operator[]
  // template<simd-integral I>
  // constexpr resize_t<I::size(), basic_vec> operator[](const I& indices) const;

  // TODO: [simd.complex.access], basic_vec complex accessors
  // constexpr real-type real() const noexcept;
  // constexpr real-type imag() const noexcept;
  // constexpr void real(const real-type& v) noexcept;
  // constexpr void imag(const real-type& v) noexcept;

  // [simd.unary], basic_vec unary operators
  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(__has_pre_increment<_Up>)
  _CCCL_API basic_vec& operator++() noexcept
  {
    _Impl::__increment(__s_);
    return *this;
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(__has_post_increment<_Up>)
  _CCCL_API basic_vec operator++(int) noexcept
  {
    const basic_vec __r = *this;
    _Impl::__increment(__s_);
    return __r;
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(__has_pre_decrement<_Up>)
  _CCCL_API basic_vec& operator--() noexcept
  {
    _Impl::__decrement(__s_);
    return *this;
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(__has_post_decrement<_Up>)
  _CCCL_API basic_vec operator--(int) noexcept
  {
    const basic_vec __r = *this;
    _Impl::__decrement(__s_);
    return __r;
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(__has_negate<_Up>)
  [[nodiscard]] _CCCL_API mask_type operator!() const noexcept
  {
    return mask_type{_Impl::__negate(__s_), mask_type::__storage_tag};
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(__has_bitwise_not<_Up>)
  [[nodiscard]] _CCCL_API constexpr basic_vec operator~() const noexcept
  {
    return basic_vec{_Impl::__bitwise_not(__s_), __storage_tag};
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(__has_plus<_Up>)
  [[nodiscard]] _CCCL_API basic_vec operator+() const noexcept
  {
    return *this;
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(__has_unary_minus<_Up>)
  [[nodiscard]] _CCCL_API basic_vec operator-() const noexcept
  {
    return basic_vec{_Impl::__unary_minus(__s_), __storage_tag};
  }

  // [simd.binary], basic_vec binary operators

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(__has_plus<_Up>)
  [[nodiscard]] _CCCL_API friend constexpr basic_vec operator+(const basic_vec& __lhs, const basic_vec& __rhs) noexcept
  {
    return basic_vec{_Impl::__plus(__lhs.__s_, __rhs.__s_), __storage_tag};
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(__has_minus<_Up>)
  [[nodiscard]] _CCCL_API friend constexpr basic_vec operator-(const basic_vec& __lhs, const basic_vec& __rhs) noexcept
  {
    return basic_vec{_Impl::__minus(__lhs.__s_, __rhs.__s_), __storage_tag};
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(__has_multiplies<_Up>)
  [[nodiscard]]
  _CCCL_API friend constexpr basic_vec operator*(const basic_vec& __lhs, const basic_vec& __rhs) noexcept
  {
    return basic_vec{_Impl::__multiplies(__lhs.__s_, __rhs.__s_), __storage_tag};
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(__has_divides<_Up>)
  [[nodiscard]] _CCCL_API friend constexpr basic_vec operator/(const basic_vec& __lhs, const basic_vec& __rhs) noexcept
  {
    return basic_vec{_Impl::__divides(__lhs.__s_, __rhs.__s_), __storage_tag};
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(__has_modulo<_Up>)
  [[nodiscard]] _CCCL_API friend constexpr basic_vec operator%(const basic_vec& __lhs, const basic_vec& __rhs) noexcept
  {
    return basic_vec{_Impl::__modulo(__lhs.__s_, __rhs.__s_), __storage_tag};
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(__has_bitwise_and<_Up>)
  [[nodiscard]] _CCCL_API friend constexpr basic_vec operator&(const basic_vec& __lhs, const basic_vec& __rhs) noexcept
  {
    return basic_vec{_Impl::__bitwise_and(__lhs.__s_, __rhs.__s_), __storage_tag};
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(__has_bitwise_or<_Up>)
  [[nodiscard]] _CCCL_API friend constexpr basic_vec operator|(const basic_vec& __lhs, const basic_vec& __rhs) noexcept
  {
    return basic_vec{_Impl::__bitwise_or(__lhs.__s_, __rhs.__s_), __storage_tag};
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(__has_bitwise_xor<_Up>)
  [[nodiscard]] _CCCL_API friend constexpr basic_vec operator^(const basic_vec& __lhs, const basic_vec& __rhs) noexcept
  {
    return basic_vec{_Impl::__bitwise_xor(__lhs.__s_, __rhs.__s_), __storage_tag};
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(__has_shift_left<_Up>)
  [[nodiscard]] _CCCL_API friend constexpr basic_vec operator<<(const basic_vec& __lhs, const basic_vec& __rhs) noexcept
  {
    return basic_vec{_Impl::__shift_left(__lhs.__s_, __rhs.__s_), __storage_tag};
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(__has_shift_right<_Up>)
  [[nodiscard]] _CCCL_API friend constexpr basic_vec operator>>(const basic_vec& __lhs, const basic_vec& __rhs) noexcept
  {
    return basic_vec{_Impl::__shift_right(__lhs.__s_, __rhs.__s_), __storage_tag};
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(__has_shift_left_size<_Up>)
  [[nodiscard]] _CCCL_API friend constexpr basic_vec operator<<(const basic_vec& __lhs, __simd_size_type __n) noexcept
  {
    return basic_vec{_Impl::__shift_left(__lhs.__s_, basic_vec{__n}), __storage_tag};
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(__has_shift_right_size<_Up>)
  [[nodiscard]] _CCCL_API friend constexpr basic_vec operator>>(const basic_vec& __lhs, __simd_size_type __n) noexcept
  {
    return basic_vec{_Impl::__shift_right(__lhs.__s_, basic_vec{__n}), __storage_tag};
  }

  // [simd.cassign], basic_vec compound assignment

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(__has_plus<_Up>)
  _CCCL_API friend constexpr basic_vec& operator+=(basic_vec& __lhs, const basic_vec& __rhs) noexcept
  {
    return __lhs = __lhs + __rhs;
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(__has_minus<_Up>)
  _CCCL_API friend constexpr basic_vec& operator-=(basic_vec& __lhs, const basic_vec& __rhs) noexcept
  {
    return __lhs = __lhs - __rhs;
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(__has_multiplies<_Up>)
  _CCCL_API friend constexpr basic_vec& operator*=(basic_vec& __lhs, const basic_vec& __rhs) noexcept
  {
    return __lhs = __lhs * __rhs;
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(__has_divides<_Up>)
  _CCCL_API friend constexpr basic_vec& operator/=(basic_vec& __lhs, const basic_vec& __rhs) noexcept
  {
    return __lhs = __lhs / __rhs;
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(__has_modulo<_Up>)
  _CCCL_API friend constexpr basic_vec& operator%=(basic_vec& __lhs, const basic_vec& __rhs) noexcept
  {
    return __lhs = __lhs % __rhs;
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(__has_bitwise_and<_Up>)
  _CCCL_API friend constexpr basic_vec& operator&=(basic_vec& __lhs, const basic_vec& __rhs) noexcept
  {
    return __lhs = __lhs & __rhs;
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(__has_bitwise_or<_Up>)
  _CCCL_API friend constexpr basic_vec& operator|=(basic_vec& __lhs, const basic_vec& __rhs) noexcept
  {
    return __lhs = __lhs | __rhs;
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(__has_bitwise_xor<_Up>)
  _CCCL_API friend constexpr basic_vec& operator^=(basic_vec& __lhs, const basic_vec& __rhs) noexcept
  {
    return __lhs = __lhs ^ __rhs;
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(__has_shift_left<_Up>)
  _CCCL_API friend constexpr basic_vec& operator<<=(basic_vec& __lhs, const basic_vec& __rhs) noexcept
  {
    return __lhs = __lhs << __rhs;
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(__has_shift_right<_Up>)
  _CCCL_API friend constexpr basic_vec& operator>>=(basic_vec& __lhs, const basic_vec& __rhs) noexcept
  {
    return __lhs = __lhs >> __rhs;
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(__has_shift_left_size<_Up>)
  _CCCL_API friend constexpr basic_vec& operator<<=(basic_vec& __lhs, __simd_size_type __n) noexcept
  {
    return __lhs = __lhs << __n;
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(__has_shift_right_size<_Up>)
  _CCCL_API friend constexpr basic_vec& operator>>=(basic_vec& __lhs, __simd_size_type __n) noexcept
  {
    return __lhs = __lhs >> __n;
  }

  // [simd.comparison], basic_vec compare operators

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(__has_equal_to<_Up>)
  [[nodiscard]] _CCCL_API friend constexpr mask_type operator==(const basic_vec& __lhs, const basic_vec& __rhs) noexcept
  {
    return mask_type{_Impl::__equal_to(__lhs.__s_, __rhs.__s_), mask_type::__storage_tag};
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(__has_not_equal_to<_Up>)
  [[nodiscard]] _CCCL_API friend constexpr mask_type operator!=(const basic_vec& __lhs, const basic_vec& __rhs) noexcept
  {
    return mask_type{_Impl::__not_equal_to(__lhs.__s_, __rhs.__s_), __storage_tag};
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(__has_greater_equal<_Up>)
  [[nodiscard]] _CCCL_API friend constexpr mask_type operator>=(const basic_vec& __lhs, const basic_vec& __rhs) noexcept
  {
    return mask_type{_Impl::__greater_equal(__lhs.__s_, __rhs.__s_), mask_type::__storage_tag};
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(__has_less_equal<_Up>)
  [[nodiscard]] _CCCL_API friend constexpr mask_type operator<=(const basic_vec& __lhs, const basic_vec& __rhs) noexcept
  {
    return mask_type{_Impl::__less_equal(__lhs.__s_, __rhs.__s_), mask_type::__storage_tag};
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(__has_greater<_Up>)
  [[nodiscard]] _CCCL_API friend constexpr mask_type operator>(const basic_vec& __lhs, const basic_vec& __rhs) noexcept
  {
    return mask_type{_Impl::__greater(__lhs.__s_, __rhs.__s_), mask_type::__storage_tag};
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(__has_less<_Up>)
  [[nodiscard]] _CCCL_API friend constexpr mask_type operator<(const basic_vec& __lhs, const basic_vec& __rhs) noexcept
  {
    return mask_type{_Impl::__less(__lhs.__s_, __rhs.__s_), mask_type::__storage_tag};
  }

  //  _CCCL_TEMPLATE(typename _Up, typename _Flags = element_aligned_tag)
  //  _CCCL_REQUIRES(__is_vectorizable_v<_Up> _CCCL_AND is_simd_flag_type_v<_Flags>)
  //  _CCCL_API void copy_from(const _Up* __mem, _Flags = {}) noexcept
  //  {
  //    _Impl::__load(__s_, _Flags::template __apply<basic_vec>(__mem));
  //  }
  //
  //  _CCCL_TEMPLATE(typename _Up, typename _Flags = element_aligned_tag)
  //  _CCCL_REQUIRES(__is_vectorizable_v<_Up> _CCCL_AND is_simd_flag_type_v<_Flags>)
  //  _CCCL_API void copy_to(_Up* __mem, _Flags = {}) const noexcept
  //  {
  //    _Impl::__store(__s_, _Flags::template __apply<basic_vec>(__mem));
  //  }
};

// TODO: deduction guides
// template<class R, class... Ts>
// basic_vec(R&& r, Ts...) -> ...;

// template<size_t Bytes, class Abi>
// basic_vec(basic_mask<Bytes, Abi>) -> ...;
} // namespace cuda::experimental::datapar

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___SIMD_SIMD_H
