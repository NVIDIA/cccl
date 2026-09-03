//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___SIMD_SATURATING_ADD_H
#define _CUDA___SIMD_SATURATING_ADD_H

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
#include <cuda/std/__numeric/saturating_add.h>
#include <cuda/std/__simd/basic_vec.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/cstdint>
#if _CCCL_HAS_SIMD_SAT()
#  include <cuda/__simd/simd_intrinsics_array.h>
#  include <cuda/std/__simd/specializations/simd_intrinsics_array.h>
#endif // _CCCL_HAS_SIMD_SAT()

#include <nv/target>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_SIMD

template <typename _Tp, typename _Abi>
struct __saturating_add_operation
{
  template <typename _Storage>
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr _Storage
  operator()(const _Storage& __lhs, const _Storage& __rhs) const noexcept
  {
#if _CCCL_HAS_SIMD_SAT()
    _CCCL_IF_NOT_CONSTEVAL_DEFAULT
    {
      if constexpr (sizeof(_Tp) == sizeof(::cuda::std::uint8_t) || sizeof(_Tp) == sizeof(::cuda::std::uint16_t))
      {
        NV_IF_TARGET(NV_HAS_FEATURE_SM_120f, ({
                       using __unsigned_storage_t = ::cuda::std::simd::__simd_storage_u32_t<_Storage>;
                       const auto __lhs_u         = ::cuda::std::simd::__to_unsigned_storage(__lhs);
                       const auto __rhs_u         = ::cuda::std::simd::__to_unsigned_storage(__rhs);
                       __unsigned_storage_t __result_u{};
                       if constexpr (sizeof(_Tp) == sizeof(::cuda::std::uint16_t))
                       {
                         __result_u = ::cuda::simd::__vadd_sat_16bit_x2<_Tp>(__lhs_u, __rhs_u);
                       }
                       else
                       {
                         __result_u = ::cuda::simd::__vadd_sat_8bit_x4<_Tp>(__lhs_u, __rhs_u);
                       }
                       return ::cuda::std::simd::__copy_from_unsigned_storage<_Storage>(__result_u);
                     }));
      }
    }
#endif // _CCCL_HAS_SIMD_SAT()

    constexpr auto __size = ::cuda::std::simd::basic_vec<_Tp, _Abi>::size();
    _Storage __result{};
    _CCCL_PRAGMA_UNROLL_FULL()
    for (::cuda::std::simd::__simd_size_type __i = 0; __i < __size; ++__i)
    {
      __result.__data[__i] = ::cuda::std::saturating_add(__lhs.__data[__i], __rhs.__data[__i]);
    }
    return __result;
  }
};

//! @brief Performs element-wise saturating addition.
//! @param __lhs The left-hand side input vector.
//! @param __rhs The right-hand side input vector.
//! @return A vector containing the saturated sum of each pair of input elements.
_CCCL_TEMPLATE(typename _Tp, typename _Abi)
_CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp>)
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr ::cuda::std::simd::basic_vec<_Tp, _Abi> saturating_add(
  const ::cuda::std::simd::basic_vec<_Tp, _Abi>& __lhs, const ::cuda::std::simd::basic_vec<_Tp, _Abi>& __rhs) noexcept
{
  return __simd_saturating_add_impl(__lhs, __rhs, __saturating_add_operation<_Tp, _Abi>{}); // ADL
}

_CCCL_END_NAMESPACE_CUDA_SIMD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___SIMD_SATURATING_ADD_H
