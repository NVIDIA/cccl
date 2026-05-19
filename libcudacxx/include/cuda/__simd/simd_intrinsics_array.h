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

#if _CCCL_HAS_SIMD_SAT() || _CCCL_HAS_SIMD_VABSDIFF() || _CCCL_HAS_SIMD_IDOT()

#  include <cuda/__simd/simd_intrinsics.h>
#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/__simd/specializations/simd_intrinsics_array.h>
#  include <cuda/std/__type_traits/is_unsigned.h>
#  include <cuda/std/cstdint>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_SIMD

//----------------------------------------------------------------------------------------------------------------------
// device-only functions

#  if _CCCL_HAS_SIMD_SAT()

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

#  endif // _CCCL_HAS_SIMD_SAT()

#  if _CCCL_HAS_SIMD_VABSDIFF()

template <typename _Tp, ::cuda::std::size_t _Np>
[[nodiscard]] _CCCL_DEVICE_API constexpr ::cuda::std::simd::__array_u32_t<_Np> __vabsdiff_8bit_x4(
  const ::cuda::std::simd::__array_u32_t<_Np>& __lhs_u,
  const ::cuda::std::simd::__array_u32_t<_Np>& __rhs_u,
  const ::cuda::std::simd::__array_u32_t<_Np>& __c_u) noexcept
{
  ::cuda::std::simd::__array_u32_t<_Np> __result_u;
  _CCCL_PRAGMA_UNROLL_FULL()
  for (::cuda::std::size_t __i = 0; __i < _Np; ++__i)
  {
    if constexpr (::cuda::std::is_unsigned_v<_Tp>)
    {
      __result_u[__i] = ::cuda::simd::__vabsdiff_u8x4(__lhs_u[__i], __rhs_u[__i], __c_u[__i]);
    }
    else
    {
      __result_u[__i] = ::cuda::simd::__vabsdiff_s8x4(__lhs_u[__i], __rhs_u[__i], __c_u[__i]);
    }
  }
  return __result_u;
}

#  endif // _CCCL_HAS_SIMD_VABSDIFF()

#  if _CCCL_HAS_SIMD_IDOT()

template <typename _Tp, typename _Up, typename _AccumT, ::cuda::std::size_t _Np>
[[nodiscard]] _CCCL_DEVICE_API _AccumT __dp4a_8bit_x4(
  const ::cuda::std::simd::__array_u32_t<_Np>& __lhs_u,
  const ::cuda::std::simd::__array_u32_t<_Np>& __rhs_u,
  const _AccumT __acc) noexcept
{
  _AccumT __result = __acc;
  _CCCL_PRAGMA_UNROLL_FULL()
  for (::cuda::std::size_t __i = 0; __i < _Np; ++__i)
  {
    if constexpr (::cuda::std::is_unsigned_v<_Tp> && ::cuda::std::is_unsigned_v<_Up>)
    {
      __result = ::cuda::simd::__dp4a_u8x4_u8x4(__lhs_u[__i], __rhs_u[__i], __result);
    }
    else if constexpr (::cuda::std::is_unsigned_v<_Tp>)
    {
      __result = ::cuda::simd::__dp4a_u8x4_s8x4(__lhs_u[__i], __rhs_u[__i], __result);
    }
    else if constexpr (::cuda::std::is_unsigned_v<_Up>)
    {
      __result = ::cuda::simd::__dp4a_s8x4_u8x4(__lhs_u[__i], __rhs_u[__i], __result);
    }
    else
    {
      __result = ::cuda::simd::__dp4a_s8x4_s8x4(__lhs_u[__i], __rhs_u[__i], __result);
    }
  }
  return __result;
}

template <typename _Tp, typename _Up, typename _AccumT, ::cuda::std::size_t _N_16Bit, ::cuda::std::size_t _N_8Bit>
[[nodiscard]] _CCCL_DEVICE_API _AccumT __dp2a_16bit_x2_8bit_x4(
  const ::cuda::std::simd::__array_u32_t<_N_16Bit>& __lhs_u16,
  const ::cuda::std::simd::__array_u32_t<_N_8Bit>& __rhs_u8,
  const _AccumT __acc) noexcept
{
  _AccumT __result = __acc;
  _CCCL_PRAGMA_UNROLL_FULL()
  for (::cuda::std::size_t __i = 0; __i < _N_16Bit; ++__i)
  {
    const auto __rhs_u = __rhs_u8[__i / 2];
    if constexpr (::cuda::std::is_unsigned_v<_Tp> && ::cuda::std::is_unsigned_v<_Up>)
    {
      if (__i % 2 == 0)
      {
        __result = ::cuda::simd::__dp2a_lo_u16x2_u8x4(__lhs_u16[__i], __rhs_u, __result);
      }
      else
      {
        __result = ::cuda::simd::__dp2a_hi_u16x2_u8x4(__lhs_u16[__i], __rhs_u, __result);
      }
    }
    else if constexpr (::cuda::std::is_unsigned_v<_Tp>)
    {
      if (__i % 2 == 0)
      {
        __result = ::cuda::simd::__dp2a_lo_u16x2_s8x4(__lhs_u16[__i], __rhs_u, __result);
      }
      else
      {
        __result = ::cuda::simd::__dp2a_hi_u16x2_s8x4(__lhs_u16[__i], __rhs_u, __result);
      }
    }
    else if constexpr (::cuda::std::is_unsigned_v<_Up>)
    {
      if (__i % 2 == 0)
      {
        __result = ::cuda::simd::__dp2a_lo_s16x2_u8x4(__lhs_u16[__i], __rhs_u, __result);
      }
      else
      {
        __result = ::cuda::simd::__dp2a_hi_s16x2_u8x4(__lhs_u16[__i], __rhs_u, __result);
      }
    }
    else
    {
      if (__i % 2 == 0)
      {
        __result = ::cuda::simd::__dp2a_lo_s16x2_s8x4(__lhs_u16[__i], __rhs_u, __result);
      }
      else
      {
        __result = ::cuda::simd::__dp2a_hi_s16x2_s8x4(__lhs_u16[__i], __rhs_u, __result);
      }
    }
  }
  return __result;
}

#  endif // _CCCL_HAS_SIMD_IDOT()

_CCCL_END_NAMESPACE_CUDA_SIMD

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_SIMD_SAT() || _CCCL_HAS_SIMD_VABSDIFF() || _CCCL_HAS_SIMD_IDOT()
#endif // _CUDA___SIMD_SIMD_INTRINSICS_ARRAY_H
