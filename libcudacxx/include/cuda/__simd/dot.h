//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___SIMD_DOT_H
#define _CUDA___SIMD_DOT_H

#include <cuda/std/detail/__config>

#include <cuda/std/__type_traits/common_type.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__internal/features.h>
#include <cuda/std/__simd/basic_vec.h>
#include <cuda/std/__simd/reductions.h>
#include <cuda/std/__simd/type_traits.h>
#include <cuda/std/__type_traits/common_type.h>
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

#if _CCCL_HAS_SIMD_IDOT()

template <typename _Tp, typename _Up>
struct __dot_operation
{
  template <typename _AccumT, typename _LhsStorage, typename _RhsStorage>
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr _AccumT
  operator()(const _LhsStorage& __lhs, const _RhsStorage& __rhs, const _AccumT __init) const noexcept
  {
    if constexpr (sizeof(_Tp) == 1 && sizeof(_Up) == 1)
    {
      const auto __lhs_u = ::cuda::std::simd::__to_unsigned_storage(__lhs);
      const auto __rhs_u = ::cuda::std::simd::__to_unsigned_storage(__rhs);
      return ::cuda::simd::__dp4a_8bit_x4<_Tp, _Up>(__lhs_u, __rhs_u, __init);
    }
    else if constexpr (sizeof(_Tp) == 2 && sizeof(_Up) == 1)
    {
      const auto __lhs_u = ::cuda::std::simd::__to_unsigned_storage(__lhs);
      const auto __rhs_u = ::cuda::std::simd::__to_unsigned_storage(__rhs);
      return ::cuda::simd::__dp2a_16bit_x2_8bit_x4<_Tp, _Up>(__lhs_u, __rhs_u, __init);
    }
    else
    {
      const auto __lhs_u = ::cuda::std::simd::__to_unsigned_storage(__lhs);
      const auto __rhs_u = ::cuda::std::simd::__to_unsigned_storage(__rhs);
      return ::cuda::simd::__dp2a_16bit_x2_8bit_x4<_Up, _Tp>(__rhs_u, __lhs_u, __init);
    }
  }
};

#endif // _CCCL_HAS_SIMD_IDOT()

//! @brief Computes a dot product and adds it to an accumulator.
//! @param[in] __lhs The left-hand side input vector.
//! @param[in] __rhs The right-hand side input vector.
//! @param[in] __init The initial accumulator value.
//! @return The accumulator plus the dot product of the input vectors.
template <typename _Tp, typename _Up, typename _Abi, typename _AccumT = ::cuda::std::common_type_t<_Tp, _Up>>
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr _AccumT
dot(const ::cuda::std::simd::basic_vec<_Tp, _Abi>& __lhs,
    const ::cuda::std::simd::basic_vec<_Up, _Abi>& __rhs,
    const _AccumT __init = {}) noexcept
{
#if _CCCL_HAS_SIMD_IDOT()
  _CCCL_IF_NOT_CONSTEVAL_DEFAULT
  {
    using ::cuda::std::__cccl_is_integer_v;
    using ::cuda::std::is_signed_v;
    using ::cuda::std::is_unsigned_v;
    constexpr bool __is_integer_dot =
      __cccl_is_integer_v<_Tp> && __cccl_is_integer_v<_Up> && __cccl_is_integer_v<_AccumT>;
    constexpr bool __is_unsigned_dot =
      __is_integer_dot && is_unsigned_v<_Tp> && is_unsigned_v<_Up> && is_unsigned_v<_AccumT>;
    constexpr bool __is_signed_dot = __is_integer_dot && (is_signed_v<_Tp> || is_signed_v<_Up>) && is_signed_v<_AccumT>;
    constexpr bool __has_matching_sign = __is_unsigned_dot || __is_signed_dot;
    constexpr bool __has_dp4a = sizeof(_AccumT) == 4 && __has_matching_sign && sizeof(_Tp) == 1 && sizeof(_Up) == 1;
    constexpr bool __has_dp2a = sizeof(_AccumT) == 4 && __has_matching_sign
                             && ((sizeof(_Tp) == 2 && sizeof(_Up) == 1) || (sizeof(_Tp) == 1 && sizeof(_Up) == 2));

    if constexpr (__has_dp4a || __has_dp2a)
    {
      NV_IF_TARGET(NV_IS_DEVICE, (return __simd_dot_impl(__lhs, __rhs, __init, __dot_operation<_Tp, _Up>{});)) // ADL
    }
  }
#endif // _CCCL_HAS_SIMD_IDOT()

  // nvcc still generates FFMA for floating-point type (FMUL2 with SM100 and similar)
  using __accum_vec = ::cuda::std::simd::rebind_t<_AccumT, ::cuda::std::simd::basic_vec<_Tp, _Abi>>;
  const __accum_vec __lhs_accum{__lhs};
  const __accum_vec __rhs_accum{__rhs};
  return __init + ::cuda::std::simd::reduce(__lhs_accum * __rhs_accum);
}

_CCCL_END_NAMESPACE_CUDA_SIMD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___SIMD_DOT_H
