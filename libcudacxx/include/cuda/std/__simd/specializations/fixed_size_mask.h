//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___SIMD_SPECIALIZATIONS_FIXED_SIZE_MASK_H
#define _CUDA_STD___SIMD_SPECIALIZATIONS_FIXED_SIZE_MASK_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__utility/in_range.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__fwd/simd.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__utility/integer_sequence.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_SIMD

// Bool-per-element mask storage for fixed_size ABI
template <size_t _Bytes, __simd_size_type _Np>
struct __mask_storage<_Bytes, __fixed_size<_Np>>
{
  static constexpr size_t __element_bytes = _Bytes;

  bool __data[_Np]{};

  _CCCL_HIDE_FROM_ABI constexpr __mask_storage() noexcept                                 = default;
  _CCCL_HIDE_FROM_ABI constexpr __mask_storage(const __mask_storage&) noexcept            = default;
  _CCCL_HIDE_FROM_ABI constexpr __mask_storage& operator=(const __mask_storage&) noexcept = default;

  [[nodiscard]] _CCCL_API constexpr bool __get(const __simd_size_type __idx) const noexcept
  {
    _CCCL_ASSERT(::cuda::in_range(__idx, __simd_size_type{0}, _Np), "Index is out of bounds");
    return __data[__idx];
  }

  _CCCL_API constexpr void __set(const __simd_size_type __idx, const bool __v) noexcept
  {
    _CCCL_ASSERT(::cuda::in_range(__idx, __simd_size_type{0}, _Np), "Index is out of bounds");
    __data[__idx] = __v;
  }
};

// Mask operations for fixed_size ABI with bool-per-element storage
template <size_t _Bytes, __simd_size_type _Np>
struct __mask_operations<_Bytes, __fixed_size<_Np>>
{
  using _MaskStorage = __mask_storage<_Bytes, __fixed_size<_Np>>;

  [[nodiscard]] _CCCL_API static constexpr _MaskStorage __broadcast(const bool __v) noexcept
  {
    _MaskStorage __result;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (__simd_size_type __i = 0; __i < _Np; ++__i)
    {
      __result.__data[__i] = __v;
    }
    return __result;
  }

  template <typename _Generator, __simd_size_type... _Is>
  [[nodiscard]] _CCCL_API static constexpr _MaskStorage
  __generate_init(_Generator&& __g, integer_sequence<__simd_size_type, _Is...>)
  {
#if _CCCL_STD_VER >= 2020
    _MaskStorage __result;
    ((__result.__data[_Is] = static_cast<bool>(__g(integral_constant<__simd_size_type, _Is>()))), ...);
    return __result;
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
    return _MaskStorage{{ static_cast<bool>(__g(integral_constant<__simd_size_type, _Is>()))... }};
#endif // _CCCL_STD_VER < 2020
  }

  template <typename _Generator>
  [[nodiscard]] _CCCL_API static constexpr _MaskStorage __generate(_Generator&& __g)
  {
    return __generate_init(__g, make_integer_sequence<__simd_size_type, _Np>());
  }

  // Logical operators (for operator&& and operator||)

  [[nodiscard]] _CCCL_API static constexpr _MaskStorage
  __logic_and(const _MaskStorage& __lhs, const _MaskStorage& __rhs) noexcept
  {
    _MaskStorage __result;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (__simd_size_type __i = 0; __i < _Np; ++__i)
    {
      __result.__data[__i] = __lhs.__data[__i] && __rhs.__data[__i];
    }
    return __result;
  }

  [[nodiscard]] _CCCL_API static constexpr _MaskStorage
  __logic_or(const _MaskStorage& __lhs, const _MaskStorage& __rhs) noexcept
  {
    _MaskStorage __result;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (__simd_size_type __i = 0; __i < _Np; ++__i)
    {
      __result.__data[__i] = __lhs.__data[__i] || __rhs.__data[__i];
    }
    return __result;
  }

  // Bitwise operators (for operator&, operator|, operator^)

  [[nodiscard]] _CCCL_API static constexpr _MaskStorage
  __bitwise_and(const _MaskStorage& __lhs, const _MaskStorage& __rhs) noexcept
  {
    _MaskStorage __result;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (__simd_size_type __i = 0; __i < _Np; ++__i)
    {
      __result.__data[__i] = __lhs.__data[__i] && __rhs.__data[__i];
    }
    return __result;
  }

  [[nodiscard]] _CCCL_API static constexpr _MaskStorage
  __bitwise_or(const _MaskStorage& __lhs, const _MaskStorage& __rhs) noexcept
  {
    _MaskStorage __result;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (__simd_size_type __i = 0; __i < _Np; ++__i)
    {
      __result.__data[__i] = __lhs.__data[__i] || __rhs.__data[__i];
    }
    return __result;
  }

  [[nodiscard]] _CCCL_API static constexpr _MaskStorage
  __bitwise_xor(const _MaskStorage& __lhs, const _MaskStorage& __rhs) noexcept
  {
    _MaskStorage __result;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (__simd_size_type __i = 0; __i < _Np; ++__i)
    {
      __result.__data[__i] = __lhs.__data[__i] != __rhs.__data[__i];
    }
    return __result;
  }

  [[nodiscard]] _CCCL_API static constexpr _MaskStorage __bitwise_not(const _MaskStorage& __s) noexcept
  {
    _MaskStorage __result;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (__simd_size_type __i = 0; __i < _Np; ++__i)
    {
      __result.__data[__i] = !__s.__data[__i];
    }
    return __result;
  }
};

_CCCL_END_NAMESPACE_CUDA_STD_SIMD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___SIMD_SPECIALIZATIONS_FIXED_SIZE_MASK_H
