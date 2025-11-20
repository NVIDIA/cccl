//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___SIMD_SIMD_MASK_H
#define _CUDAX___SIMD_SIMD_MASK_H

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
#include <cuda/std/__type_traits/is_same.h>

#include <cuda/experimental/__simd/declaration.h>
#include <cuda/experimental/__simd/reference.h>
#include <cuda/experimental/__simd/traits.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::datapar
{
template <typename _Tp, int _Np>
class simd_mask : public __mask_operations<_Tp, simd_abi::fixed_size<_Np>>
{
  using _Impl    = __mask_operations<_Tp, simd_abi::fixed_size<_Np>>;
  using _Storage = typename _Impl::_MaskStorage;

  _Storage __s_;

public:
  using value_type = bool;
  using reference  = __simd_reference<_Tp, _Storage, bool>;
  using abi_type   = simd_abi::fixed_size<_Np>;

  [[nodiscard]] _CCCL_API static constexpr ::cuda::std::size_t size() noexcept
  {
    return _Np;
  }

  _CCCL_API simd_mask() noexcept = default;

  struct __storage_tag_t
  {};
  static constexpr __storage_tag_t __storage_tag{};

  _CCCL_API explicit operator _Storage() const noexcept
  {
    return __s_;
  }

  _CCCL_API explicit simd_mask(const _Storage& __s, __storage_tag_t)
      : __s_(__s)
  {}

  _CCCL_TEMPLATE(typename _Up)
  _CCCL_REQUIRES(::cuda::std::is_same_v<_Up, bool>)
  _CCCL_API simd_mask(_Up __v) noexcept
      : __s_(_Impl::__broadcast(__v))
  {}

  _CCCL_API reference operator[](::cuda::std::size_t __i) noexcept
  {
    return reference(__s_, __i);
  }

  _CCCL_API value_type operator[](::cuda::std::size_t __i) const noexcept
  {
    return static_cast<bool>(__s_.__get(__i));
  }

  // Bitwise operations
  [[nodiscard]] _CCCL_API constexpr friend simd_mask operator&(const simd_mask& __lhs, const simd_mask& __rhs) noexcept
  {
    return {_Impl::__bitwise_and(__lhs.__s_, __rhs.__s_), __storage_tag};
  }

  [[nodiscard]] _CCCL_API constexpr friend simd_mask operator|(const simd_mask& __lhs, const simd_mask& __rhs) noexcept
  {
    return {_Impl::__bitwise_or(__lhs.__s_, __rhs.__s_), __storage_tag};
  }

  [[nodiscard]] _CCCL_API constexpr friend simd_mask operator^(const simd_mask& __lhs, const simd_mask& __rhs) noexcept
  {
    return {_Impl::__bitwise_xor(__lhs.__s_, __rhs.__s_), __storage_tag};
  }

  [[nodiscard]] _CCCL_API constexpr simd_mask operator!() const noexcept
  {
    return {_Impl::__bitwise_not(__s_), __storage_tag};
  }

  _CCCL_API simd_mask& operator&=(const simd_mask& __rhs) noexcept
  {
    return *this = *this & __rhs;
  }

  _CCCL_API simd_mask& operator|=(const simd_mask& __rhs) noexcept
  {
    return *this = *this | __rhs;
  }

  _CCCL_API simd_mask& operator^=(const simd_mask& __rhs) noexcept
  {
    return *this = *this ^ __rhs;
  }

  // Comparison operations
  [[nodiscard]] _CCCL_API constexpr friend bool operator==(const simd_mask& __lhs, const simd_mask& __rhs) noexcept
  {
    return _Impl::__equal_to(__lhs.__s_, __rhs.__s_);
  }

#if _CCCL_STD_VER < 2020
  [[nodiscard]] _CCCL_API constexpr friend bool operator!=(const simd_mask& __lhs, const simd_mask& __rhs) noexcept
  {
    return !(__lhs == __rhs);
  }
#endif // _CCCL_STD_VER < 2020

  [[nodiscard]] _CCCL_API constexpr bool all() const noexcept
  {
    return _Impl::__all(__s_);
  }

  [[nodiscard]] _CCCL_API constexpr bool any() const noexcept
  {
    return _Impl::__any(__s_);
  }

  [[nodiscard]] _CCCL_API constexpr bool none() const noexcept
  {
    return !any();
  }

  [[nodiscard]] _CCCL_API constexpr int count() const noexcept
  {
    return _Impl::__count(__s_);
  }
};

template <typename _Tp, typename _Abi>
inline constexpr bool is_simd_v<basic_simd_mask<_Tp, _Abi>> = true;

template <typename _Tp, int _Np>
using fixed_size_simd_mask = simd_mask<_Tp, _Np>;
} // namespace cuda::experimental::datapar

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___SIMD_SIMD_MASK_H
