//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___SIMD_IDOT_H
#define _CUDA___SIMD_IDOT_H

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
#include <cuda/std/__simd/basic_vec.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/is_unsigned.h>
#if _CCCL_HAS_SIMD_IDOT()
#  include <cuda/__simd/simd_intrinsics_array.h>
#  include <cuda/std/__simd/specializations/simd_intrinsics_array.h>
#endif // _CCCL_HAS_SIMD_IDOT()

#include <nv/target>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_SIMD

_CCCL_TEMPLATE(typename _Tp, typename _Up, typename _Abi, typename _AccumT)
_CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp> _CCCL_AND ::cuda::std::__cccl_is_integer_v<_Up>
                 _CCCL_AND ::cuda::std::__cccl_is_integer_v<_AccumT>)
[[nodiscard]] _CCCL_API constexpr _AccumT
idot(const ::cuda::std::simd::basic_vec<_Tp, _Abi>& __lhs,
     const ::cuda::std::simd::basic_vec<_Up, _Abi>& __rhs,
     const _AccumT __acc) noexcept
{
#if _CCCL_HAS_SIMD_IDOT()
  _CCCL_IF_NOT_CONSTEVAL_DEFAULT
  {
    using ::cuda::std::is_signed_v;
    using ::cuda::std::is_unsigned_v;
    constexpr bool __is_unsigned_dot   = is_unsigned_v<_Tp> && is_unsigned_v<_Up> && is_unsigned_v<_AccumT>;
    constexpr bool __is_signed_dot     = (is_signed_v<_Tp> || is_signed_v<_Up>) && is_signed_v<_AccumT>;
    constexpr bool __has_matching_sign = __is_unsigned_dot || __is_signed_dot;

    constexpr bool __is_dp4 = sizeof(_Tp) == 1 && sizeof(_Up) == 1 && sizeof(_AccumT) == 4 && __has_matching_sign;

    constexpr bool __is_dp2_16bitx2_8bitx4 =
      (sizeof(_Tp) == 2 && sizeof(_Up) == 1) && sizeof(_AccumT) == 4 && __has_matching_sign;
    constexpr bool __is_dp2_8bitx4_16bitx2 =
      (sizeof(_Tp) == 1 && sizeof(_Up) == 2) && sizeof(_AccumT) == 4 && __has_matching_sign;

    if constexpr (__is_dp4)
    {
      NV_IF_TARGET(NV_PROVIDES_SM_61, ({
                     const auto __lhs_u = ::cuda::std::simd::__to_unsigned_storage(__lhs.__s_);
                     const auto __rhs_u = ::cuda::std::simd::__to_unsigned_storage(__rhs.__s_);
                     return ::cuda::simd::__dp4a_8bit_x4<_Tp, _Up>(__lhs_u, __rhs_u, __acc);
                   }))
    }
    else if constexpr (__is_dp2_16bitx2_8bitx4)
    {
      NV_IF_TARGET(NV_PROVIDES_SM_61, ({
                     const auto __lhs_u = ::cuda::std::simd::__to_unsigned_storage(__lhs.__s_);
                     const auto __rhs_u = ::cuda::std::simd::__to_unsigned_storage(__rhs.__s_);
                     return ::cuda::simd::__dp2a_16bit_x2_8bit_x4<_Tp, _Up>(__lhs_u, __rhs_u, __acc);
                   }))
    }
    else if constexpr (__is_dp2_8bitx4_16bitx2)
    {
      NV_IF_TARGET(NV_PROVIDES_SM_61, ({
                     const auto __lhs_u = ::cuda::std::simd::__to_unsigned_storage(__lhs.__s_);
                     const auto __rhs_u = ::cuda::std::simd::__to_unsigned_storage(__rhs.__s_);
                     return ::cuda::simd::__dp2a_16bit_x2_8bit_x4<_Up, _Tp>(__rhs_u, __lhs_u, __acc);
                   }))
    }
  }
#endif // _CCCL_HAS_SIMD_IDOT()

  _AccumT __result = __acc;
  _CCCL_PRAGMA_UNROLL_FULL()
  for (::cuda::std::simd::__simd_size_type __i = 0; __i < __lhs.__size; ++__i)
  {
    const auto __lhs_value = static_cast<_AccumT>(__lhs.__s_.__data[__i]);
    const auto __rhs_value = static_cast<_AccumT>(__rhs.__s_.__data[__i]);
    const auto __product   = static_cast<_AccumT>(__lhs_value * __rhs_value);
    __result               = static_cast<_AccumT>(__result + __product);
  }
  return __result;
}

_CCCL_END_NAMESPACE_CUDA_SIMD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___SIMD_IDOT_H
