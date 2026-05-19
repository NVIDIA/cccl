//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___SIMD_SPECIALIZATIONS_MIN_MAX_OPTIMIZATION_H
#define _CUDA_STD___SIMD_SPECIALIZATIONS_MIN_MAX_OPTIMIZATION_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__concepts/totally_ordered.h>
#include <cuda/std/__simd/basic_vec.h>
#include <cuda/std/__simd/specializations/simd_intrinsics_array.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/cstdint>

#include <nv/target>

#include <cuda/std/__cccl/prologue.h>

// TODO(fbusato): remove the optimized paths once the compiler generates the optimized code (nvbug 6174302)

_CCCL_BEGIN_NAMESPACE_CUDA_STD_SIMD

template <typename _Tp, __simd_size_type _Np>
inline constexpr bool __is_fixed_size_small_integral_v =
  is_integral_v<_Tp> && sizeof(_Tp) < sizeof(uint32_t) && _Np >= 2;

template <typename _Vec>
struct __min_generator
{
  const _Vec& __a;
  const _Vec& __b;

  template <typename _Ip>
  [[nodiscard]] _CCCL_API constexpr typename _Vec::value_type operator()(_Ip) const noexcept
  {
    return ::cuda::std::min(__a[_Ip::value], __b[_Ip::value]);
  }
};

template <typename _Vec>
struct __max_generator
{
  const _Vec& __a;
  const _Vec& __b;

  template <typename _Ip>
  [[nodiscard]] _CCCL_API constexpr typename _Vec::value_type operator()(_Ip) const noexcept
  {
    return ::cuda::std::max(__a[_Ip::value], __b[_Ip::value]);
  }
};

_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _Vec = basic_vec<_Tp, _Abi>)
_CCCL_REQUIRES(totally_ordered<_Tp>)
[[nodiscard]]
_CCCL_API constexpr _Vec min(const basic_vec<_Tp, _Abi>& __lhs, const basic_vec<_Tp, _Abi>& __rhs) noexcept
{
#if !_CCCL_TILE_COMPILATION()
  if constexpr (__is_fixed_size_small_integral_v<_Tp, __simd_size_v<_Tp, _Abi>>)
  {
    _CCCL_IF_NOT_CONSTEVAL
    {
      using __simd_storage_t              = __simd_storage<_Tp, _Abi>;
      [[maybe_unused]] const auto __lhs_u = ::cuda::std::simd::__to_unsigned_storage(__lhs.__s_);
      [[maybe_unused]] const auto __rhs_u = ::cuda::std::simd::__to_unsigned_storage(__rhs.__s_);
      if constexpr (sizeof(_Tp) == 2) // uint16_t/int16_t x 2
      {
        NV_IF_TARGET(NV_PROVIDES_SM_90, ({
                       const auto __result_u = ::cuda::std::simd::__vmin_16bit_x2<_Tp>(__lhs_u, __rhs_u);
                       const auto __result_s =
                         ::cuda::std::simd::__copy_from_unsigned_storage<__simd_storage_t>(__result_u);
                       return _Vec{__result_s, _Vec::__storage_tag};
                     }))
      }
#  if _CCCL_HAS_SIMD_8BIT()
      else if constexpr (sizeof(_Tp) == 1) // int8_t/uint8_t x 4
      {
        NV_IF_TARGET(NV_HAS_FEATURE_SM_120f, ({
                       const auto __result_u = ::cuda::std::simd::__vmin_8bit_x4<_Tp>(__lhs_u, __rhs_u);
                       const auto __result_s =
                         ::cuda::std::simd::__copy_from_unsigned_storage<__simd_storage_t>(__result_u);
                       return _Vec{__result_s, _Vec::__storage_tag};
                     }))
      }
#  endif // _CCCL_HAS_SIMD_8BIT()
    }
  }
#endif // !_CCCL_TILE_COMPILATION()
  return _Vec{__min_generator<_Vec>{__lhs, __rhs}};
}

_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _Vec = basic_vec<_Tp, _Abi>)
_CCCL_REQUIRES(totally_ordered<_Tp>)
[[nodiscard]] _CCCL_API constexpr _Vec max(const basic_vec<_Tp, _Abi>& __lhs, const basic_vec<_Tp, _Abi>& __rhs) noexcept
{
#if !_CCCL_TILE_COMPILATION()
  if constexpr (__is_fixed_size_small_integral_v<_Tp, __simd_size_v<_Tp, _Abi>>)
  {
    _CCCL_IF_NOT_CONSTEVAL
    {
      using __simd_storage_t              = __simd_storage<_Tp, _Abi>;
      [[maybe_unused]] const auto __lhs_u = ::cuda::std::simd::__to_unsigned_storage(__lhs.__s_);
      [[maybe_unused]] const auto __rhs_u = ::cuda::std::simd::__to_unsigned_storage(__rhs.__s_);
      if constexpr (sizeof(_Tp) == 2) // uint16_t/int16_t x 2
      {
        NV_IF_TARGET(NV_PROVIDES_SM_90, ({
                       const auto __result_u = ::cuda::std::simd::__vmax_16bit_x2<_Tp>(__lhs_u, __rhs_u);
                       const auto __result_s =
                         ::cuda::std::simd::__copy_from_unsigned_storage<__simd_storage_t>(__result_u);
                       return _Vec{__result_s, _Vec::__storage_tag};
                     }))
      }
#  if _CCCL_HAS_SIMD_8BIT()
      else if constexpr (sizeof(_Tp) == 1) // int8_t/uint8_t x 4
      {
        NV_IF_TARGET(NV_HAS_FEATURE_SM_120f, ({
                       const auto __result_u = ::cuda::std::simd::__vmax_8bit_x4<_Tp>(__lhs_u, __rhs_u);
                       const auto __result_s =
                         ::cuda::std::simd::__copy_from_unsigned_storage<__simd_storage_t>(__result_u);
                       return _Vec{__result_s, _Vec::__storage_tag};
                     }))
      }
#  endif // _CCCL_HAS_SIMD_8BIT()
    }
  }
#endif // !_CCCL_TILE_COMPILATION()
  return _Vec{__max_generator<_Vec>{__lhs, __rhs}};
}

_CCCL_END_NAMESPACE_CUDA_STD_SIMD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___SIMD_SPECIALIZATIONS_MIN_MAX_OPTIMIZATION_H
