//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___SIMD_ADD_MIN_MAX_H
#define _CUDA___SIMD_ADD_MIN_MAX_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__internal/features.h>
#include <cuda/std/__simd/algorithm.h>
#include <cuda/std/__simd/basic_vec.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/cstdint>
#if _CCCL_HAS_SIMD_ADD_MIN_MAX()
#  include <cuda/__simd/simd_intrinsics_array.h>
#  include <cuda/std/__simd/specializations/simd_intrinsics_array.h>
#endif // _CCCL_HAS_SIMD_ADD_MIN_MAX()

#include <nv/target>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_SIMD

// Depending on the compiler and the gpu architecture, the plain C++ code does not always generate the optimal packed
// 16-bit SASS instructions.

#if _CCCL_HAS_SIMD_ADD_MIN_MAX()

template <typename _Tp>
struct __add_max_operation
{
  template <typename _Storage>
  [[nodiscard]] _CCCL_DEVICE_API constexpr _Storage
  operator()(const _Storage& __a, const _Storage& __b, const _Storage& __c) const noexcept
  {
    const auto __a_u      = ::cuda::std::simd::__to_unsigned_storage(__a);
    const auto __b_u      = ::cuda::std::simd::__to_unsigned_storage(__b);
    const auto __c_u      = ::cuda::std::simd::__to_unsigned_storage(__c);
    const auto __result_u = ::cuda::simd::__viaddmax_16bit_x2<_Tp>(__a_u, __b_u, __c_u);
    return ::cuda::std::simd::__copy_from_unsigned_storage<_Storage>(__result_u);
  }
};

template <typename _Tp>
struct __add_min_operation
{
  template <typename _Storage>
  [[nodiscard]] _CCCL_DEVICE_API constexpr _Storage
  operator()(const _Storage& __a, const _Storage& __b, const _Storage& __c) const noexcept
  {
    const auto __a_u      = ::cuda::std::simd::__to_unsigned_storage(__a);
    const auto __b_u      = ::cuda::std::simd::__to_unsigned_storage(__b);
    const auto __c_u      = ::cuda::std::simd::__to_unsigned_storage(__c);
    const auto __result_u = ::cuda::simd::__viaddmin_16bit_x2<_Tp>(__a_u, __b_u, __c_u);
    return ::cuda::std::simd::__copy_from_unsigned_storage<_Storage>(__result_u);
  }
};

#endif // _CCCL_HAS_SIMD_ADD_MIN_MAX()

//! @brief Performs an element-wise addition followed by a maximum.
//! @param[in] __a The first addend vector.
//! @param[in] __b The second addend vector.
//! @param[in] __c The vector compared with the sum.
//! @return A vector containing max(a[i] + b[i], c[i]) for each element.
_CCCL_TEMPLATE(typename _Tp, typename _Abi)
_CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp>)
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr ::cuda::std::simd::basic_vec<_Tp, _Abi>
add_max(const ::cuda::std::simd::basic_vec<_Tp, _Abi>& __a,
        const ::cuda::std::simd::basic_vec<_Tp, _Abi>& __b,
        const ::cuda::std::simd::basic_vec<_Tp, _Abi>& __c) noexcept
{
#if _CCCL_HAS_SIMD_ADD_MIN_MAX()
  _CCCL_IF_NOT_CONSTEVAL_DEFAULT
  {
    if constexpr (sizeof(_Tp) == sizeof(::cuda::std::int16_t))
    {
      NV_IF_TARGET(NV_IS_DEVICE, (return __simd_add_min_max_impl(__a, __b, __c, __add_max_operation<_Tp>{});)) // ADL
    }
  }
#endif // _CCCL_HAS_SIMD_ADD_MIN_MAX()

  return ::cuda::std::simd::max(__a + __b, __c);
}

//! @brief Performs an element-wise addition followed by a minimum.
//! @param[in] __a The first addend vector.
//! @param[in] __b The second addend vector.
//! @param[in] __c The vector compared with the sum.
//! @return A vector containing min(a[i] + b[i], c[i]) for each element.
_CCCL_TEMPLATE(typename _Tp, typename _Abi)
_CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp>)
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr ::cuda::std::simd::basic_vec<_Tp, _Abi>
add_min(const ::cuda::std::simd::basic_vec<_Tp, _Abi>& __a,
        const ::cuda::std::simd::basic_vec<_Tp, _Abi>& __b,
        const ::cuda::std::simd::basic_vec<_Tp, _Abi>& __c) noexcept
{
#if _CCCL_HAS_SIMD_ADD_MIN_MAX()
  _CCCL_IF_NOT_CONSTEVAL_DEFAULT
  {
    if constexpr (sizeof(_Tp) == sizeof(::cuda::std::int16_t))
    {
      NV_IF_TARGET(NV_IS_DEVICE, (return __simd_add_min_max_impl(__a, __b, __c, __add_min_operation<_Tp>{});)) // ADL
    }
  }
#endif // _CCCL_HAS_SIMD_ADD_MIN_MAX()

  return ::cuda::std::simd::min(__a + __b, __c);
}

_CCCL_END_NAMESPACE_CUDA_SIMD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___SIMD_ADD_MIN_MAX_H
