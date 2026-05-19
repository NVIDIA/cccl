//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___SIMD_SPECIALIZATIONS_FIXED_SIZE_INTEGRAL_VEC_H
#define _CUDA_STD___SIMD_SPECIALIZATIONS_FIXED_SIZE_INTEGRAL_VEC_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// automatic vectorization for small integers is not supported (until CUDA 13.2)
// TODO(fbusato): remove this path once the feature is supported
// TODO(fbusato): extend to other GPU archs in the future

#include <cuda/__cmath/ceil_div.h>
#include <cuda/std/__simd/specializations/fixed_size_vec.h>
#include <cuda/std/__simd/specializations/simd_intrinsics_array.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/array>
#include <cuda/std/cstdint>

#include <nv/target>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_SIMD

template <typename _Tp, __simd_size_type _Np>
inline constexpr bool __is_fixed_size_small_integral_v =
  is_integral_v<_Tp> && sizeof(_Tp) < sizeof(uint32_t) && _Np >= 2;

inline constexpr auto __simd_operations_small_integral = __simd_operations_kind::__fixed_size_integral;

template <typename _Tp, __simd_size_type _Np>
inline constexpr __simd_operations_kind __simd_operations_kind_v<_Tp, __fixed_size<_Np>> =
  __is_fixed_size_small_integral_v<_Tp, _Np> ? __simd_operations_small_integral : __simd_operations_kind::__default;

#define _CCCL_SIMD_FIXED_SIZE_INTEGRAL_BINARY_BITWISE(_NAME, _OP)                           \
  [[nodiscard]] _CCCL_API static constexpr __simd_storage_t _NAME(                          \
    const __simd_storage_t& __lhs, const __simd_storage_t& __rhs) noexcept                  \
  {                                                                                         \
    _CCCL_IF_NOT_CONSTEVAL_DEFAULT                                                          \
    {                                                                                       \
      __unsigned_storage_t __result_u{};                                                    \
      const auto __lhs_u = ::cuda::std::simd::__to_unsigned_storage(__lhs);                 \
      const auto __rhs_u = ::cuda::std::simd::__to_unsigned_storage(__rhs);                 \
      _CCCL_PRAGMA_UNROLL_FULL()                                                            \
      for (__simd_size_type __i = 0; __i < __usize; ++__i)                                  \
      {                                                                                     \
        __result_u[__i] = __lhs_u[__i] _OP __rhs_u[__i];                                    \
      }                                                                                     \
      return ::cuda::std::simd::__copy_from_unsigned_storage<__simd_storage_t>(__result_u); \
    }                                                                                       \
    return __base::_NAME(__lhs, __rhs);                                                     \
  }

// Simd operations for fixed_size ABI with small integral element types.
template <typename _Tp, __simd_size_type _Np>
struct __simd_operations<_Tp, __fixed_size<_Np>, __simd_operations_small_integral> : __fixed_size_operations<_Tp, _Np>
{
  using __base           = __fixed_size_operations<_Tp, _Np>;
  using __simd_storage_t = __simd_storage<_Tp, __fixed_size<_Np>>;

  // all computation is done on uint32_t, so the alignment must be at least the alignment of uint32_t
  static_assert(alignof(__simd_storage_t) >= alignof(uint32_t));

  static constexpr __simd_size_type __ratio = sizeof(uint32_t) / sizeof(_Tp);
  static constexpr __simd_size_type __usize = ::cuda::ceil_div(_Np, __ratio);
  using __unsigned_storage_t                = array<uint32_t, __usize>;

  [[nodiscard]] _CCCL_API static constexpr __simd_storage_t __bitwise_not(const __simd_storage_t& __s) noexcept
  {
    _CCCL_IF_NOT_CONSTEVAL_DEFAULT
    {
      auto __udata = ::cuda::std::simd::__to_unsigned_storage(__s);
      _CCCL_PRAGMA_UNROLL_FULL()
      for (__simd_size_type __i = 0; __i < __usize; ++__i)
      {
        __udata[__i] = ~__udata[__i];
      }
      return ::cuda::std::simd::__copy_from_unsigned_storage<__simd_storage_t>(__udata);
    }
    return __fixed_size_operations<_Tp, _Np>::__bitwise_not(__s);
  }

  _CCCL_SIMD_FIXED_SIZE_INTEGRAL_BINARY_BITWISE(__bitwise_and, &)
  _CCCL_SIMD_FIXED_SIZE_INTEGRAL_BINARY_BITWISE(__bitwise_or, |)
  _CCCL_SIMD_FIXED_SIZE_INTEGRAL_BINARY_BITWISE(__bitwise_xor, ^)

#if _CCCL_CUDA_COMPILATION() && !_CCCL_TILE_COMPILATION()
  // Unary arithmetic operations

  // x++ = x + 1
  _CCCL_API static constexpr void __increment(__simd_storage_t& __s) noexcept
  {
    _CCCL_IF_NOT_CONSTEVAL_DEFAULT
    {
      [[maybe_unused]] constexpr __simd_storage_t __one = __base::__broadcast(1);
      if constexpr (sizeof(_Tp) == 2)
      {
        NV_IF_TARGET(NV_PROVIDES_SM_90, (__s = __plus(__s, __one); return;))
      }
#  if _CCCL_HAS_SIMD_8BIT()
      else if constexpr (sizeof(_Tp) == 1)
      {
        NV_IF_TARGET(NV_HAS_FEATURE_SM_120f, (__s = __plus(__s, __one); return;))
      }
#  endif // _CCCL_HAS_SIMD_8BIT()
    }
    __base::__increment(__s);
  }

  // x-- = x - 1
  _CCCL_API static constexpr void __decrement(__simd_storage_t& __s) noexcept
  {
    _CCCL_IF_NOT_CONSTEVAL_DEFAULT
    {
      [[maybe_unused]] constexpr __simd_storage_t __minus_one = __base::__broadcast(static_cast<_Tp>(-1));
      if constexpr (sizeof(_Tp) == 2)
      {
        NV_IF_TARGET(NV_PROVIDES_SM_90, (__s = __plus(__s, __minus_one); return;))
      }
#  if _CCCL_HAS_SIMD_8BIT()
      else if constexpr (sizeof(_Tp) == 1)
      {
        NV_IF_TARGET(NV_HAS_FEATURE_SM_120f, (__s = __plus(__s, __minus_one); return;))
      }
#  endif // _CCCL_HAS_SIMD_8BIT()
    }
    __base::__decrement(__s);
  }

  // -x = ~x + 1
  [[nodiscard]]
  _CCCL_API static constexpr __simd_storage_t __unary_minus(const __simd_storage_t& __s) noexcept
  {
    _CCCL_IF_NOT_CONSTEVAL_DEFAULT
    {
      [[maybe_unused]] constexpr __simd_storage_t __one = __base::__broadcast(1);
      if constexpr (sizeof(_Tp) == 2)
      {
        NV_IF_TARGET(NV_PROVIDES_SM_90, (return __plus(__bitwise_not(__s), __one);))
      }
#  if _CCCL_HAS_SIMD_8BIT()
      else if constexpr (sizeof(_Tp) == 1)
      {
        NV_IF_TARGET(NV_HAS_FEATURE_SM_120f, (return __plus(__bitwise_not(__s), __one);))
      }
#  endif // _CCCL_HAS_SIMD_8BIT()
    }
    return __base::__unary_minus(__s);
  }

  // Binary arithmetic operations

  [[nodiscard]]
  _CCCL_API static constexpr __simd_storage_t
  __plus(const __simd_storage_t& __lhs, const __simd_storage_t& __rhs) noexcept
  {
    _CCCL_IF_NOT_CONSTEVAL_DEFAULT
    {
      [[maybe_unused]] const auto __lhs_u = ::cuda::std::simd::__to_unsigned_storage(__lhs);
      [[maybe_unused]] const auto __rhs_u = ::cuda::std::simd::__to_unsigned_storage(__rhs);
      if constexpr (sizeof(_Tp) == 2)
      {
        NV_IF_TARGET(NV_PROVIDES_SM_90,
                     (return ::cuda::std::simd::__copy_from_unsigned_storage<__simd_storage_t>(
                               ::cuda::std::simd::__vadd_16bit_x2<_Tp>(__lhs_u, __rhs_u));))
      }
#  if _CCCL_HAS_SIMD_8BIT()
      else if constexpr (sizeof(_Tp) == 1)
      {
        NV_IF_TARGET(NV_HAS_FEATURE_SM_120f,
                     (return ::cuda::std::simd::__copy_from_unsigned_storage<__simd_storage_t>(
                               ::cuda::std::simd::__vadd_8bit_x4<_Tp>(__lhs_u, __rhs_u));))
      }
#  endif // _CCCL_HAS_SIMD_8BIT()
    }
    return __fixed_size_operations<_Tp, _Np>::__plus(__lhs, __rhs);
  }

  [[nodiscard]]
  _CCCL_API static constexpr __simd_storage_t
  __minus(const __simd_storage_t& __lhs, const __simd_storage_t& __rhs) noexcept
  {
    _CCCL_IF_NOT_CONSTEVAL_DEFAULT
    {
      if constexpr (sizeof(_Tp) == 2)
      {
        NV_IF_TARGET(NV_PROVIDES_SM_90, (return __plus(__lhs, __unary_minus(__rhs));))
      }
#  if _CCCL_HAS_SIMD_8BIT()
      else if constexpr (sizeof(_Tp) == 1)
      {
        NV_IF_TARGET(NV_HAS_FEATURE_SM_120f, (return __plus(__lhs, __unary_minus(__rhs));))
      }
#  endif // _CCCL_HAS_SIMD_8BIT()
    }
    return __base::__minus(__lhs, __rhs);
  }
#endif // _CCCL_CUDA_COMPILATION() && !_CCCL_TILE_COMPILATION()
};

#undef _CCCL_SIMD_FIXED_SIZE_INTEGRAL_BINARY_BITWISE

_CCCL_END_NAMESPACE_CUDA_STD_SIMD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___SIMD_SPECIALIZATIONS_FIXED_SIZE_INTEGRAL_VEC_H
