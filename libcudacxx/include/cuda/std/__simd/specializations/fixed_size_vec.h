//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___SIMD_SPECIALIZATIONS_FIXED_SIZE_VEC_H
#define _CUDA_STD___SIMD_SPECIALIZATIONS_FIXED_SIZE_VEC_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__utility/in_range.h>
#include <cuda/std/__fwd/simd.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__utility/integer_sequence.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_SIMD

template <__simd_size_type _Np>
struct __fixed_size
{
  static_assert(_Np > 0, "_Np must be greater than 0");

  static constexpr __simd_size_type __simd_size = _Np;
};

// Element-per-slot simd storage for fixed_size ABI
template <typename _Tp, __simd_size_type _Np>
struct __simd_storage<_Tp, __fixed_size<_Np>>
{
  using value_type = _Tp;

  _Tp __data[_Np]{};

  _CCCL_HIDE_FROM_ABI constexpr __simd_storage()                                 = default;
  _CCCL_HIDE_FROM_ABI constexpr __simd_storage(const __simd_storage&)            = default;
  _CCCL_HIDE_FROM_ABI constexpr __simd_storage& operator=(const __simd_storage&) = default;

  [[nodiscard]] _CCCL_API constexpr _Tp __get(const __simd_size_type __idx) const noexcept
  {
    _CCCL_ASSERT(::cuda::in_range(__idx, __simd_size_type{0}, _Np), "Index is out of bounds");
    return __data[__idx];
  }

  _CCCL_API constexpr void __set(const __simd_size_type __idx, const _Tp __v) noexcept
  {
    _CCCL_ASSERT(::cuda::in_range(__idx, __simd_size_type{0}, _Np), "Index is out of bounds");
    __data[__idx] = __v;
  }
};

// Simd operations for fixed_size ABI
template <typename _Tp, __simd_size_type _Np>
struct __simd_operations<_Tp, __fixed_size<_Np>>
{
  using _SimdStorage = __simd_storage<_Tp, __fixed_size<_Np>>;
  using _MaskStorage = __mask_storage<sizeof(_Tp), __fixed_size<_Np>>;

  [[nodiscard]] _CCCL_API static constexpr _SimdStorage __broadcast(const _Tp __v) noexcept
  {
    _SimdStorage __result;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (__simd_size_type __i = 0; __i < _Np; ++__i)
    {
      __result.__data[__i] = __v;
    }
    return __result;
  }

  template <typename _Generator, __simd_size_type... _Is>
  [[nodiscard]] _CCCL_API static constexpr _SimdStorage
  __generate_init(_Generator&& __g, integer_sequence<__simd_size_type, _Is...>)
  {
#if _CCCL_STD_VER >= 2020
    _SimdStorage __result;
    ((__result.__data[_Is] = __g(integral_constant<__simd_size_type, _Is>())), ...);
    return __result;
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
    return _SimdStorage{{ __g(integral_constant<__simd_size_type, _Is>())... }};
#endif // _CCCL_STD_VER < 2020
  }

  template <typename _Generator>
  [[nodiscard]] _CCCL_API static constexpr _SimdStorage __generate(_Generator&& __g)
  {
    return __generate_init(__g, make_integer_sequence<__simd_size_type, _Np>());
  }

  // Unary operations

  _CCCL_API static constexpr void __increment(_SimdStorage& __s) noexcept
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (__simd_size_type __i = 0; __i < _Np; ++__i)
    {
      ++__s.__data[__i];
    }
  }

  _CCCL_API static constexpr void __decrement(_SimdStorage& __s) noexcept
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (__simd_size_type __i = 0; __i < _Np; ++__i)
    {
      --__s.__data[__i];
    }
  }

  [[nodiscard]] _CCCL_API static constexpr _MaskStorage __negate(const _SimdStorage& __s) noexcept
  {
    _MaskStorage __result;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (__simd_size_type __i = 0; __i < _Np; ++__i)
    {
      __result.__data[__i] = !__s.__data[__i];
    }
    return __result;
  }

  [[nodiscard]] _CCCL_API static constexpr _SimdStorage __bitwise_not(const _SimdStorage& __s) noexcept
  {
    _SimdStorage __result;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (__simd_size_type __i = 0; __i < _Np; ++__i)
    {
      __result.__data[__i] = ~__s.__data[__i];
    }
    return __result;
  }

  _CCCL_DIAG_PUSH
  _CCCL_DIAG_SUPPRESS_MSVC(4146) // unary minus applied to unsigned type
  [[nodiscard]] _CCCL_API static constexpr _SimdStorage __unary_minus(const _SimdStorage& __s) noexcept
  {
    _SimdStorage __result;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (__simd_size_type __i = 0; __i < _Np; ++__i)
    {
      __result.__data[__i] = -__s.__data[__i];
    }
    return __result;
  }
  _CCCL_DIAG_POP

  // Binary arithmetic operations

  [[nodiscard]] _CCCL_API static constexpr _SimdStorage
  __plus(const _SimdStorage& __lhs, const _SimdStorage& __rhs) noexcept
  {
    _SimdStorage __result;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (__simd_size_type __i = 0; __i < _Np; ++__i)
    {
      __result.__data[__i] = (__lhs.__data[__i] + __rhs.__data[__i]);
    }
    return __result;
  }

  [[nodiscard]] _CCCL_API static constexpr _SimdStorage
  __minus(const _SimdStorage& __lhs, const _SimdStorage& __rhs) noexcept
  {
    _SimdStorage __result;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (__simd_size_type __i = 0; __i < _Np; ++__i)
    {
      __result.__data[__i] = (__lhs.__data[__i] - __rhs.__data[__i]);
    }
    return __result;
  }

  [[nodiscard]] _CCCL_API static constexpr _SimdStorage
  __multiplies(const _SimdStorage& __lhs, const _SimdStorage& __rhs) noexcept
  {
    _SimdStorage __result;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (__simd_size_type __i = 0; __i < _Np; ++__i)
    {
      __result.__data[__i] = (__lhs.__data[__i] * __rhs.__data[__i]);
    }
    return __result;
  }

  [[nodiscard]] _CCCL_API static constexpr _SimdStorage
  __divides(const _SimdStorage& __lhs, const _SimdStorage& __rhs) noexcept
  {
    _SimdStorage __result;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (__simd_size_type __i = 0; __i < _Np; ++__i)
    {
      __result.__data[__i] = (__lhs.__data[__i] / __rhs.__data[__i]);
    }
    return __result;
  }

  [[nodiscard]] _CCCL_API static constexpr _SimdStorage
  __modulo(const _SimdStorage& __lhs, const _SimdStorage& __rhs) noexcept
  {
    _SimdStorage __result;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (__simd_size_type __i = 0; __i < _Np; ++__i)
    {
      __result.__data[__i] = (__lhs.__data[__i] % __rhs.__data[__i]);
    }
    return __result;
  }

  // Comparison operations

  [[nodiscard]] _CCCL_API static constexpr _MaskStorage
  __equal_to(const _SimdStorage& __lhs, const _SimdStorage& __rhs) noexcept
  {
    _MaskStorage __result;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (__simd_size_type __i = 0; __i < _Np; ++__i)
    {
      __result.__data[__i] = (__lhs.__data[__i] == __rhs.__data[__i]);
    }
    return __result;
  }

  [[nodiscard]] _CCCL_API static constexpr _MaskStorage
  __not_equal_to(const _SimdStorage& __lhs, const _SimdStorage& __rhs) noexcept
  {
    _MaskStorage __result;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (__simd_size_type __i = 0; __i < _Np; ++__i)
    {
      __result.__data[__i] = (__lhs.__data[__i] != __rhs.__data[__i]);
    }
    return __result;
  }

  [[nodiscard]] _CCCL_API static constexpr _MaskStorage
  __less(const _SimdStorage& __lhs, const _SimdStorage& __rhs) noexcept
  {
    _MaskStorage __result;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (__simd_size_type __i = 0; __i < _Np; ++__i)
    {
      __result.__data[__i] = (__lhs.__data[__i] < __rhs.__data[__i]);
    }
    return __result;
  }

  [[nodiscard]] _CCCL_API static constexpr _MaskStorage
  __less_equal(const _SimdStorage& __lhs, const _SimdStorage& __rhs) noexcept
  {
    _MaskStorage __result;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (__simd_size_type __i = 0; __i < _Np; ++__i)
    {
      __result.__data[__i] = (__lhs.__data[__i] <= __rhs.__data[__i]);
    }
    return __result;
  }

  [[nodiscard]] _CCCL_API static constexpr _MaskStorage
  __greater(const _SimdStorage& __lhs, const _SimdStorage& __rhs) noexcept
  {
    _MaskStorage __result;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (__simd_size_type __i = 0; __i < _Np; ++__i)
    {
      __result.__data[__i] = (__lhs.__data[__i] > __rhs.__data[__i]);
    }
    return __result;
  }

  [[nodiscard]] _CCCL_API static constexpr _MaskStorage
  __greater_equal(const _SimdStorage& __lhs, const _SimdStorage& __rhs) noexcept
  {
    _MaskStorage __result;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (__simd_size_type __i = 0; __i < _Np; ++__i)
    {
      __result.__data[__i] = (__lhs.__data[__i] >= __rhs.__data[__i]);
    }
    return __result;
  }

  // Bitwise and shift operations

  [[nodiscard]] _CCCL_API static constexpr _SimdStorage
  __bitwise_and(const _SimdStorage& __lhs, const _SimdStorage& __rhs) noexcept
  {
    _SimdStorage __result;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (__simd_size_type __i = 0; __i < _Np; ++__i)
    {
      __result.__data[__i] = (__lhs.__data[__i] & __rhs.__data[__i]);
    }
    return __result;
  }

  [[nodiscard]] _CCCL_API static constexpr _SimdStorage
  __bitwise_or(const _SimdStorage& __lhs, const _SimdStorage& __rhs) noexcept
  {
    _SimdStorage __result;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (__simd_size_type __i = 0; __i < _Np; ++__i)
    {
      __result.__data[__i] = (__lhs.__data[__i] | __rhs.__data[__i]);
    }
    return __result;
  }

  [[nodiscard]] _CCCL_API static constexpr _SimdStorage
  __bitwise_xor(const _SimdStorage& __lhs, const _SimdStorage& __rhs) noexcept
  {
    _SimdStorage __result;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (__simd_size_type __i = 0; __i < _Np; ++__i)
    {
      __result.__data[__i] = (__lhs.__data[__i] ^ __rhs.__data[__i]);
    }
    return __result;
  }

  [[nodiscard]] _CCCL_API static constexpr _SimdStorage
  __shift_left(const _SimdStorage& __lhs, const _SimdStorage& __rhs) noexcept
  {
    _SimdStorage __result;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (__simd_size_type __i = 0; __i < _Np; ++__i)
    {
      __result.__data[__i] = (__lhs.__data[__i] << __rhs.__data[__i]);
    }
    return __result;
  }

  [[nodiscard]] _CCCL_API static constexpr _SimdStorage
  __shift_right(const _SimdStorage& __lhs, const _SimdStorage& __rhs) noexcept
  {
    _SimdStorage __result;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (__simd_size_type __i = 0; __i < _Np; ++__i)
    {
      __result.__data[__i] = (__lhs.__data[__i] >> __rhs.__data[__i]);
    }
    return __result;
  }
};
_CCCL_END_NAMESPACE_CUDA_STD_SIMD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___SIMD_SPECIALIZATIONS_FIXED_SIZE_VEC_H
