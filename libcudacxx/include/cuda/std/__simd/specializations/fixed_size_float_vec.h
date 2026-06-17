//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___SIMD_SPECIALIZATIONS_FIXED_SIZE_FLOAT_VEC_H
#define _CUDA_STD___SIMD_SPECIALIZATIONS_FIXED_SIZE_FLOAT_VEC_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_SIMD_F32X2()

#  include <cuda/std/__simd/specializations/fixed_size_vec.h>
#  include <cuda/std/__simd/specializations/fp32x2_intrinsics.h>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_SIMD

template <__simd_size_type _Np>
inline constexpr __simd_operations_kind __simd_operations_kind_v<float, __fixed_size<_Np>> =
  (_Np >= 2) ? __simd_operations_kind::__fixed_size_float : __simd_operations_kind::__default;

// Simd operations for fixed_size ABI with float elements and F32x2 fast paths.
template <__simd_size_type _Np>
struct __simd_operations<float, __fixed_size<_Np>, __simd_operations_kind::__fixed_size_float>
    : __fixed_size_operations<float, _Np>
{
  using __base       = __fixed_size_operations<float, _Np>;
  using _SimdStorage = __simd_storage<float, __fixed_size<_Np>>;

  _CCCL_HOST_DEVICE_API static constexpr void __increment(_SimdStorage& __s) noexcept
  {
    _CCCL_IF_NOT_CONSTEVAL_DEFAULT
    {
      NV_IF_TARGET(NV_IS_EXACTLY_SM_100, ({
                     constexpr _SimdStorage __one = __base::__broadcast(1.0f);
                     __s                          = ::cuda::std::simd::__plus_f32x2(__s, __one);
                     return;
                   }));
    }
    __base::__increment(__s);
  }

  _CCCL_HOST_DEVICE_API static constexpr void __decrement(_SimdStorage& __s) noexcept
  {
    _CCCL_IF_NOT_CONSTEVAL_DEFAULT
    {
      NV_IF_TARGET(NV_IS_EXACTLY_SM_100, ({
                     constexpr _SimdStorage __one = __base::__broadcast(1.0f);
                     __s                          = ::cuda::std::simd::__minus_f32x2(__s, __one);
                     return;
                   }));
    }
    __base::__decrement(__s);
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API static constexpr _SimdStorage __unary_minus(const _SimdStorage& __s) noexcept
  {
    _CCCL_IF_NOT_CONSTEVAL_DEFAULT
    {
      NV_IF_TARGET(NV_IS_EXACTLY_SM_100, ({
                     constexpr _SimdStorage __zero = __base::__broadcast(0.0f);
                     return ::cuda::std::simd::__minus_f32x2(__zero, __s);
                   }))
    }
    return __base::__unary_minus(__s);
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API static constexpr _SimdStorage
  __plus(const _SimdStorage& __lhs, const _SimdStorage& __rhs) noexcept
  {
    _CCCL_IF_NOT_CONSTEVAL_DEFAULT
    {
      NV_IF_TARGET(NV_IS_EXACTLY_SM_100, (return ::cuda::std::simd::__plus_f32x2(__lhs, __rhs);))
    }
    return __base::__plus(__lhs, __rhs);
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API static constexpr _SimdStorage
  __minus(const _SimdStorage& __lhs, const _SimdStorage& __rhs) noexcept
  {
    _CCCL_IF_NOT_CONSTEVAL_DEFAULT
    {
      NV_IF_TARGET(NV_IS_EXACTLY_SM_100, (return ::cuda::std::simd::__minus_f32x2(__lhs, __rhs);))
    }
    return __base::__minus(__lhs, __rhs);
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API static constexpr _SimdStorage
  __multiplies(const _SimdStorage& __lhs, const _SimdStorage& __rhs) noexcept
  {
    _CCCL_IF_NOT_CONSTEVAL_DEFAULT
    {
      NV_IF_TARGET(NV_IS_EXACTLY_SM_100, (return ::cuda::std::simd::__multiplies_f32x2(__lhs, __rhs);))
    }
    return __base::__multiplies(__lhs, __rhs);
  }
};

_CCCL_END_NAMESPACE_CUDA_STD_SIMD

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_SIMD_F32X2()
#endif // _CUDA_STD___SIMD_SPECIALIZATIONS_FIXED_SIZE_FLOAT_VEC_H
