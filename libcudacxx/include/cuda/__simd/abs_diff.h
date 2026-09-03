//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___SIMD_ABS_DIFF_H
#define _CUDA___SIMD_ABS_DIFF_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/abs_diff.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__internal/features.h>
#include <cuda/std/__simd/algorithm.h>
#include <cuda/std/__simd/basic_vec.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/cstdint>
#if _CCCL_HAS_SIMD_VABSDIFF()
#  include <cuda/__simd/simd_intrinsics_array.h>
#  include <cuda/std/__simd/specializations/simd_intrinsics_array.h>
#endif // _CCCL_HAS_SIMD_VABSDIFF()

#include <nv/target>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_SIMD

#if _CCCL_HAS_SIMD_VABSDIFF()

template <typename _Tp>
struct __abs_diff_8bit_operation
{
  template <typename _ResultStorage, typename _Storage>
  [[nodiscard]] _CCCL_DEVICE_API constexpr _ResultStorage
  operator()(const _Storage& __lhs, const _Storage& __rhs) const noexcept
  {
    using __unsigned_storage_t _CCCL_NODEBUG = ::cuda::std::simd::__simd_storage_u32_t<_ResultStorage>;
    constexpr __unsigned_storage_t __c_u{};
    const auto __lhs_u    = ::cuda::std::simd::__to_unsigned_storage(__lhs);
    const auto __rhs_u    = ::cuda::std::simd::__to_unsigned_storage(__rhs);
    const auto __result_u = ::cuda::simd::__vabsdiff_8bit_x4<_Tp>(__lhs_u, __rhs_u, __c_u);
    return ::cuda::std::simd::__copy_from_unsigned_storage<_ResultStorage>(__result_u);
  }
};

#endif // _CCCL_HAS_SIMD_VABSDIFF()

template <typename _Tp, typename _Abi>
struct __abs_diff_32bit_operation
{
  template <typename _ResultStorage, typename _Storage>
  [[nodiscard]] _CCCL_DEVICE_API constexpr _ResultStorage
  operator()(const _Storage& __lhs, const _Storage& __rhs) const noexcept
  {
    constexpr auto __size = ::cuda::std::simd::basic_vec<_Tp, _Abi>::size();
    _ResultStorage __result{};
    _CCCL_PRAGMA_UNROLL_FULL()
    for (::cuda::std::simd::__simd_size_type __i = 0; __i < __size; ++__i)
    {
      __result.__data[__i] = ::cuda::abs_diff(__lhs.__data[__i], __rhs.__data[__i]);
    }
    return __result;
  }
};

//! @brief Performs element-wise absolute difference.
//! @param[in] __lhs The left-hand side input vector.
//! @param[in] __rhs The right-hand side input vector.
//! @return An unsigned vector containing the absolute difference of each pair of input elements.
_CCCL_TEMPLATE(typename _Tp, typename _Abi)
_CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp>)
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr ::cuda::std::simd::basic_vec<::cuda::std::make_unsigned_t<_Tp>, _Abi>
abs_diff(const ::cuda::std::simd::basic_vec<_Tp, _Abi>& __lhs,
         const ::cuda::std::simd::basic_vec<_Tp, _Abi>& __rhs) noexcept
{
  using __result_type _CCCL_NODEBUG = ::cuda::std::simd::basic_vec<::cuda::std::make_unsigned_t<_Tp>, _Abi>;
#if _CCCL_HAS_SIMD_VABSDIFF()
  _CCCL_IF_NOT_CONSTEVAL_DEFAULT
  {
    NV_IF_TARGET(
      NV_IS_DEVICE, ({
        if constexpr (sizeof(_Tp) == sizeof(::cuda::std::uint8_t))
        {
          return __simd_abs_diff_impl(
            __lhs, __rhs, __abs_diff_8bit_operation<_Tp>{}, static_cast<__result_type*>(nullptr)); // ADL
        }
        if constexpr (sizeof(_Tp) == sizeof(::cuda::std::uint32_t))
        {
          return __simd_abs_diff_impl(
            __lhs, __rhs, __abs_diff_32bit_operation<_Tp, _Abi>{}, static_cast<__result_type*>(nullptr)); // ADL
        }
      }))
  }
#endif // _CCCL_HAS_SIMD_VABSDIFF()

  return __result_type{::cuda::std::simd::max(__lhs, __rhs)} - __result_type{::cuda::std::simd::min(__lhs, __rhs)};
}

_CCCL_END_NAMESPACE_CUDA_SIMD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___SIMD_ABS_DIFF_H
