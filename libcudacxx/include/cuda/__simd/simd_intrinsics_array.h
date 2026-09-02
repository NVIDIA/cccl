//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___SIMD_SIMD_INTRINSICS_ARRAY_H
#define _CUDA___SIMD_SIMD_INTRINSICS_ARRAY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_SIMD_SAT()

#  include <cuda/__simd/simd_intrinsics.h>
#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/__simd/specializations/simd_intrinsics_array.h>
#  include <cuda/std/__type_traits/is_unsigned.h>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_SIMD

//----------------------------------------------------------------------------------------------------------------------
// device-only functions

template <typename _Tp, ::cuda::std::size_t _Np>
[[nodiscard]] _CCCL_DEVICE_API constexpr ::cuda::std::simd::__array_u32_t<_Np> __vadd_sat_16bit_x2(
  const ::cuda::std::simd::__array_u32_t<_Np>& __lhs_u, const ::cuda::std::simd::__array_u32_t<_Np>& __rhs_u) noexcept
{
  ::cuda::std::simd::__array_u32_t<_Np> __result_u;
  _CCCL_PRAGMA_UNROLL_FULL()
  for (::cuda::std::size_t __i = 0; __i < _Np; ++__i)
  {
    if constexpr (::cuda::std::is_unsigned_v<_Tp>)
    {
      __result_u[__i] = ::cuda::simd::__vadd_sat_u16x2(__lhs_u[__i], __rhs_u[__i]);
    }
    else
    {
      __result_u[__i] = ::cuda::simd::__vadd_sat_s16x2(__lhs_u[__i], __rhs_u[__i]);
    }
  }
  return __result_u;
}

template <typename _Tp, ::cuda::std::size_t _Np>
[[nodiscard]] _CCCL_DEVICE_API constexpr ::cuda::std::simd::__array_u32_t<_Np> __vadd_sat_8bit_x4(
  const ::cuda::std::simd::__array_u32_t<_Np>& __lhs_u, const ::cuda::std::simd::__array_u32_t<_Np>& __rhs_u) noexcept
{
  ::cuda::std::simd::__array_u32_t<_Np> __result_u;
  _CCCL_PRAGMA_UNROLL_FULL()
  for (::cuda::std::size_t __i = 0; __i < _Np; ++__i)
  {
    if constexpr (::cuda::std::is_unsigned_v<_Tp>)
    {
      __result_u[__i] = ::cuda::simd::__vadd_sat_u8x4(__lhs_u[__i], __rhs_u[__i]);
    }
    else
    {
      __result_u[__i] = ::cuda::simd::__vadd_sat_s8x4(__lhs_u[__i], __rhs_u[__i]);
    }
  }
  return __result_u;
}

_CCCL_END_NAMESPACE_CUDA_SIMD

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_SIMD_SAT()
#endif // _CUDA___SIMD_SIMD_INTRINSICS_ARRAY_H
