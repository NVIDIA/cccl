//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___SIMD_BASIC_VEC_H
#define _CUDA_STD___SIMD_BASIC_VEC_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__utility/in_range.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__fwd/simd.h>
#include <cuda/std/__iterator/default_sentinel.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__ranges/data.h>
#include <cuda/std/__simd/basic_mask.h>
#include <cuda/std/__simd/concepts.h>
#include <cuda/std/__simd/flag.h>
#include <cuda/std/__simd/iterator.h>
#include <cuda/std/__simd/specializations/fixed_size_vec.h>
#include <cuda/std/__simd/utility.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/operations.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_SIMD

// If basic_vec<T, Abi> is disabled, the specialization has a deleted default constructor, deleted destructor, deleted
// copy constructor, and deleted copy assignment. In addition only the value_type, abi_type, and mask_type members are
// present.
template <typename _Tp, typename _Abi, typename>
class basic_vec
{
public:
  using value_type = _Tp;
  using abi_type   = _Abi;
  using mask_type  = basic_mask<sizeof(_Tp), _Abi>;

  _CCCL_HIDE_FROM_ABI basic_vec()                            = delete;
  _CCCL_HIDE_FROM_ABI ~basic_vec()                           = delete;
  _CCCL_HIDE_FROM_ABI basic_vec(const basic_vec&)            = delete;
  _CCCL_HIDE_FROM_ABI basic_vec& operator=(const basic_vec&) = delete;
};

// basic_vec<T, Abi> is enabled when T is a vectorizable type and there exists N in [1, 64] derived from
// deduce-abi-t<T, N>
template <typename _Tp, typename _Abi>
class basic_vec<_Tp, _Abi, enable_if_t<__is_vectorizable_v<_Tp> && __is_enabled_abi_v<_Abi>>>
    : public __simd_operations<_Tp, _Abi>
{
public:
  using value_type = _Tp;
  using mask_type  = basic_mask<sizeof(value_type), _Abi>;

private:
  template <size_t, typename, typename>
  friend class basic_mask;

  using _Impl    = __simd_operations<_Tp, _Abi>;
  using _Storage = typename _Impl::_SimdStorage;

  _Storage __s_{};

  struct __storage_tag_t
  {};
  static constexpr __storage_tag_t __storage_tag{};

  _CCCL_API constexpr basic_vec(const _Storage __s, __storage_tag_t) noexcept
      : __s_{__s}
  {}

  // Friend comparison operators (e.g. operator==) cannot access basic_mask's private constructor directly (friendship
  // is not transitive). This function is required to access the private constructor of basic_mask.
  _CCCL_API static constexpr mask_type __make_mask(const typename mask_type::_Storage __s) noexcept
  {
    return mask_type{__s, mask_type::__storage_tag};
  }

  // operator[] is const only. We need this function to set values
  _CCCL_API constexpr void __set(const __simd_size_type __i, const value_type __v) noexcept
  {
    __s_.__set(__i, __v);
  }

public:
  using abi_type = _Abi;

  using iterator       = __simd_iterator<basic_vec>;
  using const_iterator = __simd_iterator<const basic_vec>;

  [[nodiscard]] _CCCL_API constexpr iterator begin() noexcept
  {
    return {*this, 0};
  }

  [[nodiscard]] _CCCL_API constexpr const_iterator begin() const noexcept
  {
    return {*this, 0};
  }

  [[nodiscard]] _CCCL_API constexpr const_iterator cbegin() const noexcept
  {
    return {*this, 0};
  }

  [[nodiscard]] _CCCL_API constexpr default_sentinel_t end() const noexcept
  {
    return {};
  }

  [[nodiscard]] _CCCL_API constexpr default_sentinel_t cend() const noexcept
  {
    return {};
  }

  static constexpr integral_constant<__simd_size_type, __simd_size_v<value_type, abi_type>> size{};

  static constexpr auto __usize = size_t{size};
  static constexpr auto __size  = __simd_size_type{size};

  _CCCL_HIDE_FROM_ABI basic_vec() noexcept = default;

  // [simd.ctor], basic_vec constructors

  // [simd.ctor] value broadcast constructor (explicit overload)
  _CCCL_TEMPLATE(typename _Up)
  _CCCL_REQUIRES((__explicitly_convertible_to<_Up, value_type>) _CCCL_AND(!__is_value_ctor_implicit<_Up, value_type>))
  _CCCL_API constexpr explicit basic_vec(_Up&& __v) noexcept
      : __s_{_Impl::__broadcast(static_cast<value_type>(__v))}
  {}

  // [simd.ctor] value broadcast constructor (implicit overload)
  _CCCL_TEMPLATE(typename _Up)
  _CCCL_REQUIRES((__explicitly_convertible_to<_Up, value_type>) _CCCL_AND(__is_value_ctor_implicit<_Up, value_type>))
  _CCCL_API constexpr basic_vec(_Up&& __v) noexcept
      : __s_{_Impl::__broadcast(static_cast<value_type>(__v))}
  {}

  // [simd.ctor] converting constructor from basic_vec<U, UAbi> (explicit overload)
  _CCCL_TEMPLATE(typename _Up, typename _UAbi)
  _CCCL_REQUIRES((__simd_size_v<_Up, _UAbi> == __size) _CCCL_AND(__explicitly_convertible_to<_Up, value_type>)
                   _CCCL_AND(__is_vec_ctor_explicit<_Up, value_type>))
  _CCCL_API constexpr explicit basic_vec(const basic_vec<_Up, _UAbi>& __v) noexcept
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (__simd_size_type __i = 0; __i < __size; ++__i)
    {
      __s_.__set(__i, static_cast<value_type>(__v[__i]));
    }
  }

  // [simd.ctor] converting constructor from basic_vec<U, UAbi> (implicit overload)
  _CCCL_TEMPLATE(typename _Up, typename _UAbi)
  _CCCL_REQUIRES((__simd_size_v<_Up, _UAbi> == __size) _CCCL_AND(__explicitly_convertible_to<_Up, value_type>)
                   _CCCL_AND(!__is_vec_ctor_explicit<_Up, value_type>))
  _CCCL_API constexpr basic_vec(const basic_vec<_Up, _UAbi>& __v) noexcept
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (__simd_size_type __i = 0; __i < __size; ++__i)
    {
      __s_.__set(__i, static_cast<value_type>(__v[__i]));
    }
  }

  // [simd.ctor] generator constructor
  _CCCL_TEMPLATE(typename _Generator)
  _CCCL_REQUIRES(__can_generate_v<value_type, _Generator, __size>)
  _CCCL_API constexpr explicit basic_vec(_Generator&& __g)
      : __s_{_Impl::__generate(__g)}
  {}

  // [simd.ctor] range constructor

  template <typename _Range>
  static constexpr bool __is_compatible_range = __is_compatible_range_v<value_type, __size, _Range>;

  // [simd.ctor] range constructor
  _CCCL_TEMPLATE(typename _Range, typename... _Flags)
  _CCCL_REQUIRES(__is_compatible_range<_Range>)
  _CCCL_API constexpr basic_vec(_Range&& __range, flags<_Flags...> = {})
  {
    static_assert(__has_convert_flag_v<_Flags...> || __is_value_preserving_v<ranges::range_value_t<_Range>, value_type>,
                  "Conversion from range_value_t<R> to value_type is not value-preserving; use flag_convert");
    const auto __data = ::cuda::std::ranges::data(__range);
    __assert_load_store_alignment<basic_vec, ranges::range_value_t<_Range>, _Flags...>(__data);
    _CCCL_PRAGMA_UNROLL_FULL()
    for (__simd_size_type __i = 0; __i < __size; ++__i)
    {
      __s_.__set(__i, static_cast<value_type>(__data[__i]));
    }
  }

  // [simd.ctor] masked range constructor
  _CCCL_TEMPLATE(typename _Range, typename... _Flags)
  _CCCL_REQUIRES(__is_compatible_range<_Range>)
  _CCCL_API constexpr basic_vec(_Range&& __range, const mask_type& __mask, flags<_Flags...> = {})
  {
    static_assert(__has_convert_flag_v<_Flags...> || __is_value_preserving_v<ranges::range_value_t<_Range>, value_type>,
                  "Conversion from range_value_t<R> to value_type is not value-preserving; use flag_convert");
    const auto __data = ranges::data(__range);
    __assert_load_store_alignment<basic_vec, ranges::range_value_t<_Range>, _Flags...>(__data);
    _CCCL_PRAGMA_UNROLL_FULL()
    for (__simd_size_type __i = 0; __i < __size; ++__i)
    {
      __s_.__set(__i, __mask[__i] ? static_cast<value_type>(__data[__i]) : value_type());
    }
  }

  // TODO(fbusato): add complex constructor
  // constexpr basic_vec(const real-type& __reals, const real-type& __imags = {}) noexcept;

  // [simd.subscr], basic_vec subscript operators

  [[nodiscard]] _CCCL_API constexpr value_type operator[](const __simd_size_type __i) const noexcept
  {
    _CCCL_ASSERT(::cuda::in_range(__i, __simd_size_type{0}, __size), "Index is out of bounds");
    return __s_.__get(__i);
  }

  // TODO(fbusato): subscript with integral indices, requires permute()
  // template<simd-integral _Idx>
  //   constexpr resize_t<_Idx::size(), basic_vec> operator[](const _Idx& __indices) const;

  // TODO(fbusato): [simd.complex.access], basic_vec complex accessors
  // constexpr real-type real() const noexcept;
  // constexpr real-type imag() const noexcept;
  // constexpr void real(const real-type& __v) noexcept;
  // constexpr void imag(const real-type& __v) noexcept;

  // [simd.unary], basic_vec unary operators

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(__has_pre_increment<_Up>)
  _CCCL_API constexpr basic_vec& operator++() noexcept
  {
    _Impl::__increment(__s_);
    return *this;
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(__has_post_increment<_Up>)
  [[nodiscard]] _CCCL_API constexpr basic_vec operator++(int) noexcept
  {
    const basic_vec __r = *this;
    _Impl::__increment(__s_);
    return __r;
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(__has_pre_decrement<_Up>)
  _CCCL_API constexpr basic_vec& operator--() noexcept
  {
    _Impl::__decrement(__s_);
    return *this;
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(__has_post_decrement<_Up>)
  [[nodiscard]] _CCCL_API constexpr basic_vec operator--(int) noexcept
  {
    const basic_vec __r = *this;
    _Impl::__decrement(__s_);
    return __r;
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(__has_negate<_Up>)
  [[nodiscard]] _CCCL_API constexpr mask_type operator!() const noexcept
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
  _CCCL_REQUIRES(__has_unary_plus<_Up>)
  [[nodiscard]] _CCCL_API constexpr basic_vec operator+() const noexcept
  {
    return *this;
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(__has_unary_minus<_Up>)
  [[nodiscard]] _CCCL_API constexpr basic_vec operator-() const noexcept
  {
    return basic_vec{_Impl::__unary_minus(__s_), __storage_tag};
  }

  // [simd.binary], basic_vec binary operators

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(__has_binary_plus<_Up>)
  [[nodiscard]] _CCCL_API friend constexpr basic_vec operator+(const basic_vec& __lhs, const basic_vec& __rhs) noexcept
  {
    return basic_vec{_Impl::__plus(__lhs.__s_, __rhs.__s_), __storage_tag};
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(__has_binary_minus<_Up>)
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
  [[nodiscard]] _CCCL_API friend constexpr basic_vec
  operator<<(const basic_vec& __lhs, const __simd_size_type __n) noexcept
  {
    return __lhs << basic_vec{__n};
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(__has_shift_right_size<_Up>)
  [[nodiscard]] _CCCL_API friend constexpr basic_vec
  operator>>(const basic_vec& __lhs, const __simd_size_type __n) noexcept
  {
    return __lhs >> basic_vec{__n};
  }

  // [simd.cassign], basic_vec compound assignment

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(__has_binary_plus<_Up>)
  _CCCL_API friend constexpr basic_vec& operator+=(basic_vec& __lhs, const basic_vec& __rhs) noexcept
  {
    return __lhs = __lhs + __rhs;
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(__has_binary_minus<_Up>)
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
  _CCCL_API friend constexpr basic_vec& operator<<=(basic_vec& __lhs, const __simd_size_type __n) noexcept
  {
    return __lhs = __lhs << __n;
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(__has_shift_right_size<_Up>)
  _CCCL_API friend constexpr basic_vec& operator>>=(basic_vec& __lhs, const __simd_size_type __n) noexcept
  {
    return __lhs = __lhs >> __n;
  }

  // [simd.comparison], basic_vec compare operators

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(__has_equal_to<_Up>)
  [[nodiscard]] _CCCL_API friend constexpr mask_type operator==(const basic_vec& __lhs, const basic_vec& __rhs) noexcept
  {
    return __make_mask(_Impl::__equal_to(__lhs.__s_, __rhs.__s_));
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(__has_not_equal_to<_Up>)
  [[nodiscard]] _CCCL_API friend constexpr mask_type operator!=(const basic_vec& __lhs, const basic_vec& __rhs) noexcept
  {
    return __make_mask(_Impl::__not_equal_to(__lhs.__s_, __rhs.__s_));
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(__has_greater_equal<_Up>)
  [[nodiscard]] _CCCL_API friend constexpr mask_type operator>=(const basic_vec& __lhs, const basic_vec& __rhs) noexcept
  {
    return __make_mask(_Impl::__greater_equal(__lhs.__s_, __rhs.__s_));
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(__has_less_equal<_Up>)
  [[nodiscard]] _CCCL_API friend constexpr mask_type operator<=(const basic_vec& __lhs, const basic_vec& __rhs) noexcept
  {
    return __make_mask(_Impl::__less_equal(__lhs.__s_, __rhs.__s_));
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(__has_greater<_Up>)
  [[nodiscard]] _CCCL_API friend constexpr mask_type operator>(const basic_vec& __lhs, const basic_vec& __rhs) noexcept
  {
    return __make_mask(_Impl::__greater(__lhs.__s_, __rhs.__s_));
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(__has_less<_Up>)
  [[nodiscard]] _CCCL_API friend constexpr mask_type operator<(const basic_vec& __lhs, const basic_vec& __rhs) noexcept
  {
    return __make_mask(_Impl::__less(__lhs.__s_, __rhs.__s_));
  }

  // TODO(fbusato): [simd.cond], basic_vec exposition-only conditional operators
  // friend constexpr basic_vec __simd_select_impl(
  //   const mask_type&, const basic_vec&, const basic_vec&) noexcept;
};

// [simd.ctor] deduction guide from contiguous sized range
// Deduces vec<range_value_t<R>, static_cast<simd-size-type>(ranges::size(r))>
//    * it is not possible to use the alias "vec" for the deduction guide
//    * "vec" is defined as basic_vec<_Tp, __deduce_abi_t<_Tp, _Np>>
//    * where _Np is __simd_size_v<_Tp, __static_range_size_v<_Range>>
_CCCL_TEMPLATE(typename _Range, typename... _Ts)
_CCCL_REQUIRES(
  ranges::contiguous_range<_Range> _CCCL_AND ranges::sized_range<_Range> _CCCL_AND __has_static_size<_Range>)
_CCCL_HOST_DEVICE basic_vec(_Range&&, _Ts...)
  -> basic_vec<ranges::range_value_t<_Range>,
               __deduce_abi_t<ranges::range_value_t<_Range>, __static_range_size_v<_Range>>>;

// [simd.ctor] deduction guide from basic_mask
// basic_vec<__integer_from<Bytes>, Abi> is equivalent to decltype(+k):
//   * k has type basic_mask<_Bytes, _Abi>
//   * +k calls basic_mask::operator+()
//   * the return type is basic_vec<__integer_from<_Bp>, _Abi>
// The deduced type is equivalent to decltype(+k), i.e. basic_vec<__integer_from<Bytes>, Abi>
_CCCL_TEMPLATE(size_t _Bytes, typename _Abi)
_CCCL_REQUIRES(__has_unary_plus<basic_mask<_Bytes, _Abi>>)
_CCCL_HOST_DEVICE basic_vec(basic_mask<_Bytes, _Abi>) -> basic_vec<__integer_from<_Bytes>, _Abi>;

_CCCL_END_NAMESPACE_CUDA_STD_SIMD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___SIMD_BASIC_VEC_H
