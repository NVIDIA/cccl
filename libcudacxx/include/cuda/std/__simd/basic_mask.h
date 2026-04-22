//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___SIMD_BASIC_MASK_H
#define _CUDA_STD___SIMD_BASIC_MASK_H

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
#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__fwd/simd.h>
#include <cuda/std/__simd/specializations/fixed_size_mask.h>
#include <cuda/std/__simd/utility.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_unsigned.h>
#include <cuda/std/__type_traits/num_bits.h>
#include <cuda/std/bitset>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_SIMD

template <size_t _Bytes>
inline constexpr bool __is_vectorizable_byte_size_v =
  (_Bytes == 1 || _Bytes == 2 || _Bytes == 4 || _Bytes == 8
#if _CCCL_HAS_INT128()
   || _Bytes == 16
#endif // _CCCL_HAS_INT128()
  );

// If basic_mask<Bytes, Abi> is disabled, the specialization has a deleted default constructor, deleted destructor,
// deleted copy constructor, and deleted copy assignment. In addition only the value_type and abi_type members are
// present.
template <size_t _Bytes, typename _Abi, typename>
class basic_mask
{
public:
  using value_type = bool;
  using abi_type   = _Abi;

  _CCCL_HIDE_FROM_ABI basic_mask()                             = delete;
  _CCCL_HIDE_FROM_ABI ~basic_mask()                            = delete;
  _CCCL_HIDE_FROM_ABI basic_mask(const basic_mask&)            = delete;
  _CCCL_HIDE_FROM_ABI basic_mask& operator=(const basic_mask&) = delete;
};

// basic_mask<Bytes, Abi> is enabled when there exists a vectorizable type T with sizeof(T) == Bytes and N in [1, 64]
// derived from deduce-abi-t<T, N>
template <size_t _Bytes, typename _Abi>
class basic_mask<_Bytes, _Abi, enable_if_t<__is_vectorizable_byte_size_v<_Bytes> && __is_enabled_abi_v<_Abi>>>
    : public __mask_operations<_Bytes, _Abi>
{
  template <typename, typename, typename>
  friend class basic_vec;

  using _Impl    = __mask_operations<_Bytes, _Abi>;
  using _Storage = typename _Impl::_MaskStorage;

  _Storage __s_;

  struct __storage_tag_t
  {};
  static constexpr __storage_tag_t __storage_tag{};

  _CCCL_API constexpr basic_mask(const _Storage __v, __storage_tag_t) noexcept
      : __s_{__v}
  {}

public:
  using value_type = bool;
  using abi_type   = _Abi;

  // TODO(fbusato): add simd-iterator
  // using iterator       = simd-iterator<basic_mask>;
  // using const_iterator = simd-iterator<const basic_mask>;

  // constexpr iterator begin() noexcept { return {*this, 0}; }
  // constexpr const_iterator begin() const noexcept { return {*this, 0}; }
  // constexpr const_iterator cbegin() const noexcept { return {*this, 0}; }
  // constexpr default_sentinel_t end() const noexcept { return {}; }
  // constexpr default_sentinel_t cend() const noexcept { return {}; }

  static constexpr integral_constant<__simd_size_type, __simd_size_v<__integer_from<_Bytes>, _Abi>> size{};

  static constexpr auto __usize = size_t{size};
  static constexpr auto __size  = __simd_size_type{size};

  _CCCL_HIDE_FROM_ABI constexpr basic_mask() noexcept = default;

  // [simd.mask.ctor], basic_mask constructors

  _CCCL_TEMPLATE(typename _Up)
  _CCCL_REQUIRES(same_as<_Up, value_type>)
  _CCCL_API constexpr explicit basic_mask(const _Up __v) noexcept
      : __s_{_Impl::__broadcast(__v)}
  {}

  _CCCL_TEMPLATE(size_t _UBytes, typename _UAbi)
  _CCCL_REQUIRES((__simd_size_v<__integer_from<_UBytes>, _UAbi> == __size))
  _CCCL_API constexpr explicit basic_mask(const basic_mask<_UBytes, _UAbi>& __x) noexcept
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (__simd_size_type __i = 0; __i < __size; ++__i)
    {
      __s_.__set(__i, __x[__i]);
    }
  }

  _CCCL_TEMPLATE(typename _Generator)
  _CCCL_REQUIRES(__can_generate_v<bool, _Generator, __size>)
  _CCCL_API constexpr explicit basic_mask(_Generator&& __g)
      : __s_{_Impl::__generate(__g)}
  {}

  _CCCL_TEMPLATE(typename _Tp)
  _CCCL_REQUIRES(same_as<_Tp, bitset<__usize>>)
  _CCCL_API constexpr basic_mask(const _Tp& __b) noexcept
      : __s_{_Impl::__broadcast(false)}
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (__simd_size_type __i = 0; __i < __size; ++__i)
    {
      __s_.__set(__i, static_cast<bool>(__b[__i]));
    }
  }

  _CCCL_TEMPLATE(typename _Tp)
  _CCCL_REQUIRES(is_integral_v<_Tp> _CCCL_AND is_unsigned_v<_Tp> _CCCL_AND(!is_same_v<_Tp, value_type>))
  _CCCL_API constexpr explicit basic_mask(const _Tp __val) noexcept
      : __s_{_Impl::__broadcast(false)}
  {
    constexpr auto __num_bits = __simd_size_type{__num_bits_v<_Tp>};
    constexpr auto __m        = __size < __num_bits ? __size : __num_bits;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (__simd_size_type __i = 0; __i < __m; ++__i)
    {
      __s_.__set(__i, static_cast<bool>((__val >> __i) & _Tp{1}));
    }
  }

  // [simd.mask.subscr], basic_mask subscript operators

  [[nodiscard]] _CCCL_API constexpr value_type operator[](const __simd_size_type __i) const noexcept
  {
    _CCCL_ASSERT(::cuda::in_range(__i, __simd_size_type{0}, __size), "Index is out of bounds");
    return static_cast<bool>(__s_.__get(__i));
  }

  // TODO(fbusato): subscript with integral indices, requires permute()
  // template<simd-integral I>
  // constexpr resize_t<I::size(), basic_mask> operator[](const I& indices) const;

  // [simd.mask.unary], basic_mask unary operators

  [[nodiscard]] _CCCL_API constexpr basic_mask operator!() const noexcept
  {
    return {_Impl::__bitwise_not(__s_), __storage_tag};
  }

  using __unary_return_t = basic_vec<__integer_from<_Bytes>, _Abi>;

  [[nodiscard]] _CCCL_API constexpr __unary_return_t operator+() const noexcept
  {
    return static_cast<__unary_return_t>(*this);
  }

  [[nodiscard]] _CCCL_API constexpr __unary_return_t operator-() const noexcept
  {
    return -static_cast<__unary_return_t>(*this);
  }

  [[nodiscard]] _CCCL_API constexpr __unary_return_t operator~() const noexcept
  {
    return ~static_cast<__unary_return_t>(*this);
  }

  // [simd.mask.conv], basic_mask conversions

  _CCCL_TEMPLATE(typename _Up, typename _Ap)
  _CCCL_REQUIRES((sizeof(_Up) != _Bytes && __simd_size_v<_Up, _Ap> == __size))
  _CCCL_API constexpr explicit operator basic_vec<_Up, _Ap>() const noexcept
  {
    basic_vec<_Up, _Ap> __result{};
    _CCCL_PRAGMA_UNROLL_FULL()
    for (__simd_size_type __i = 0; __i < __size; ++__i)
    {
      __result.__s_.__set(__i, static_cast<_Up>((*this)[__i]));
    }
    return __result;
  }

  _CCCL_TEMPLATE(typename _Up, typename _Ap)
  _CCCL_REQUIRES((sizeof(_Up) == _Bytes && __simd_size_v<_Up, _Ap> == __size))
  _CCCL_API constexpr operator basic_vec<_Up, _Ap>() const noexcept
  {
    basic_vec<_Up, _Ap> __result{};
    _CCCL_PRAGMA_UNROLL_FULL()
    for (__simd_size_type __i = 0; __i < __size; ++__i)
    {
      __result.__s_.__set(__i, static_cast<_Up>((*this)[__i]));
    }
    return __result;
  }

  [[nodiscard]] _CCCL_API constexpr bitset<__usize> to_bitset() const noexcept
  {
    bitset<__usize> __result{};
    _CCCL_PRAGMA_UNROLL_FULL()
    for (__simd_size_type __i = 0; __i < __size; ++__i)
    {
      __result.set(__i, (*this)[__i]);
    }
    return __result;
  }

  [[nodiscard]] _CCCL_API constexpr unsigned long long to_ullong() const
  {
    constexpr __simd_size_type __nbits = __num_bits_v<unsigned long long>;
    if constexpr (__size > __nbits)
    {
      for (auto __i = __nbits; __i < __size; ++__i)
      {
        _CCCL_ASSERT(!(*this)[__i], "Bit above unsigned long long width is set");
      }
    }
    return to_bitset().to_ullong();
  }

  // [simd.mask.binary], basic_mask binary operators

  [[nodiscard]] _CCCL_API friend constexpr basic_mask
  operator&&(const basic_mask& __lhs, const basic_mask& __rhs) noexcept
  {
    return {_Impl::__logic_and(__lhs.__s_, __rhs.__s_), __storage_tag};
  }

  [[nodiscard]] _CCCL_API friend constexpr basic_mask
  operator||(const basic_mask& __lhs, const basic_mask& __rhs) noexcept
  {
    return {_Impl::__logic_or(__lhs.__s_, __rhs.__s_), __storage_tag};
  }

  [[nodiscard]] _CCCL_API friend constexpr basic_mask
  operator&(const basic_mask& __lhs, const basic_mask& __rhs) noexcept
  {
    return {_Impl::__bitwise_and(__lhs.__s_, __rhs.__s_), __storage_tag};
  }

  [[nodiscard]] _CCCL_API friend constexpr basic_mask
  operator|(const basic_mask& __lhs, const basic_mask& __rhs) noexcept
  {
    return {_Impl::__bitwise_or(__lhs.__s_, __rhs.__s_), __storage_tag};
  }

  [[nodiscard]] _CCCL_API friend constexpr basic_mask
  operator^(const basic_mask& __lhs, const basic_mask& __rhs) noexcept
  {
    return {_Impl::__bitwise_xor(__lhs.__s_, __rhs.__s_), __storage_tag};
  }

  // [simd.mask.cassign], basic_mask compound assignment

  _CCCL_API friend constexpr basic_mask& operator&=(basic_mask& __lhs, const basic_mask& __rhs) noexcept
  {
    return __lhs = __lhs & __rhs;
  }

  _CCCL_API friend constexpr basic_mask& operator|=(basic_mask& __lhs, const basic_mask& __rhs) noexcept
  {
    return __lhs = __lhs | __rhs;
  }

  _CCCL_API friend constexpr basic_mask& operator^=(basic_mask& __lhs, const basic_mask& __rhs) noexcept
  {
    return __lhs = __lhs ^ __rhs;
  }

  // [simd.mask.comparison], basic_mask comparisons (element-wise)

  [[nodiscard]] _CCCL_API friend constexpr basic_mask
  operator==(const basic_mask& __lhs, const basic_mask& __rhs) noexcept
  {
    return !(__lhs ^ __rhs);
  }

  [[nodiscard]] _CCCL_API friend constexpr basic_mask
  operator!=(const basic_mask& __lhs, const basic_mask& __rhs) noexcept
  {
    return __lhs ^ __rhs;
  }

  [[nodiscard]] _CCCL_API friend constexpr basic_mask
  operator>=(const basic_mask& __lhs, const basic_mask& __rhs) noexcept
  {
    return __lhs || !__rhs;
  }

  [[nodiscard]] _CCCL_API friend constexpr basic_mask
  operator<=(const basic_mask& __lhs, const basic_mask& __rhs) noexcept
  {
    return !__lhs || __rhs;
  }

  [[nodiscard]] _CCCL_API friend constexpr basic_mask
  operator>(const basic_mask& __lhs, const basic_mask& __rhs) noexcept
  {
    return __lhs && !__rhs;
  }

  [[nodiscard]] _CCCL_API friend constexpr basic_mask
  operator<(const basic_mask& __lhs, const basic_mask& __rhs) noexcept
  {
    return !__lhs && __rhs;
  }

  // TODO(fbusato): [simd.mask.cond], basic_mask exposition only conditional operators
  // friend constexpr basic_mask __simd_select_impl(
  //   const basic_mask&, const basic_mask&, const basic_mask&) noexcept;
  // friend constexpr basic_mask __simd_select_impl(
  //   const basic_mask&, same_as<bool> auto, same_as<bool> auto) noexcept;
  // template<class T0, class T1>
  //   friend constexpr vec<see below, size()> __simd_select_impl(
  //     const basic_mask&, const T0&, const T1&) noexcept;
};
_CCCL_END_NAMESPACE_CUDA_STD_SIMD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___SIMD_BASIC_MASK_H
