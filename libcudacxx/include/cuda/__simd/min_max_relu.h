//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___SIMD_MIN_MAX_RELU_H
#define _CUDA___SIMD_MIN_MAX_RELU_H

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
#include <cuda/std/__type_traits/is_signed_integer.h>
#include <cuda/std/cstdint>
#if _CCCL_HAS_SIMD_MIN_MAX_RELU()
#  include <cuda/__simd/simd_intrinsics_array.h>
#  include <cuda/std/__simd/specializations/simd_intrinsics_array.h>
#endif // _CCCL_HAS_SIMD_MIN_MAX_RELU()

#include <nv/target>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_SIMD

// Depending on the compiler and the gpu architecture, the plain C++ code (without intrinsics) can generate or not the
// expected SASS instructions. The instrinsic paths are preferred to always generate the optimal code.

#if _CCCL_HAS_SIMD_MIN_MAX_RELU()

template <typename _Tp, typename _Abi>
struct __max_relu_operation
{
  template <typename _Storage>
  [[nodiscard]] _CCCL_DEVICE_API constexpr _Storage
  operator()(const _Storage& __lhs, const _Storage& __rhs) const noexcept
  {
#  if _CCCL_HAS_SIMD_8BIT_PTX()
    if constexpr (sizeof(_Tp) == sizeof(::cuda::std::int8_t))
    {
      const auto __lhs_u    = ::cuda::std::simd::__to_unsigned_storage(__lhs);
      const auto __rhs_u    = ::cuda::std::simd::__to_unsigned_storage(__rhs);
      const auto __result_u = ::cuda::simd::__vmax_relu_8bit_x4(__lhs_u, __rhs_u);
      return ::cuda::std::simd::__copy_from_unsigned_storage<_Storage>(__result_u);
    }
    else
#  endif // _CCCL_HAS_SIMD_8BIT_PTX()
      if constexpr (sizeof(_Tp) == sizeof(::cuda::std::int16_t))
      {
        const auto __lhs_u    = ::cuda::std::simd::__to_unsigned_storage(__lhs);
        const auto __rhs_u    = ::cuda::std::simd::__to_unsigned_storage(__rhs);
        const auto __result_u = ::cuda::simd::__vmax_relu_16bit_x2(__lhs_u, __rhs_u);
        return ::cuda::std::simd::__copy_from_unsigned_storage<_Storage>(__result_u);
      }
      else
      {
        constexpr auto __size = ::cuda::std::simd::basic_vec<_Tp, _Abi>::size();
        _Storage __result{};
        _CCCL_PRAGMA_UNROLL_FULL()
        for (::cuda::std::simd::__simd_size_type __i = 0; __i < __size; ++__i)
        {
          __result.__data[__i] = ::__vimax_s32_relu(__lhs.__data[__i], __rhs.__data[__i]);
        }
        return __result;
      }
  }

  template <typename _Storage>
  [[nodiscard]] _CCCL_DEVICE_API constexpr _Storage
  operator()(const _Storage& __a, const _Storage& __b, const _Storage& __c) const noexcept
  {
    if constexpr (sizeof(_Tp) == sizeof(::cuda::std::int16_t))
    {
      const auto __a_u      = ::cuda::std::simd::__to_unsigned_storage(__a);
      const auto __b_u      = ::cuda::std::simd::__to_unsigned_storage(__b);
      const auto __c_u      = ::cuda::std::simd::__to_unsigned_storage(__c);
      const auto __result_u = ::cuda::simd::__vmax3_relu_16bit_x2(__a_u, __b_u, __c_u);
      return ::cuda::std::simd::__copy_from_unsigned_storage<_Storage>(__result_u);
    }
    else
    {
      constexpr auto __size = ::cuda::std::simd::basic_vec<_Tp, _Abi>::size();
      _Storage __result{};
      _CCCL_PRAGMA_UNROLL_FULL()
      for (::cuda::std::simd::__simd_size_type __i = 0; __i < __size; ++__i)
      {
        __result.__data[__i] = ::__vimax3_s32_relu(__a.__data[__i], __b.__data[__i], __c.__data[__i]);
      }
      return __result;
    }
  }
};

template <typename _Tp, typename _Abi>
struct __min_relu_operation
{
  template <typename _Storage>
  [[nodiscard]] _CCCL_DEVICE_API constexpr _Storage
  operator()(const _Storage& __lhs, const _Storage& __rhs) const noexcept
  {
#  if _CCCL_HAS_SIMD_8BIT_PTX()
    if constexpr (sizeof(_Tp) == sizeof(::cuda::std::int8_t))
    {
      const auto __lhs_u    = ::cuda::std::simd::__to_unsigned_storage(__lhs);
      const auto __rhs_u    = ::cuda::std::simd::__to_unsigned_storage(__rhs);
      const auto __result_u = ::cuda::simd::__vmin_relu_8bit_x4(__lhs_u, __rhs_u);
      return ::cuda::std::simd::__copy_from_unsigned_storage<_Storage>(__result_u);
    }
    else
#  endif // _CCCL_HAS_SIMD_8BIT_PTX()
      if constexpr (sizeof(_Tp) == sizeof(::cuda::std::int16_t))
      {
        const auto __lhs_u    = ::cuda::std::simd::__to_unsigned_storage(__lhs);
        const auto __rhs_u    = ::cuda::std::simd::__to_unsigned_storage(__rhs);
        const auto __result_u = ::cuda::simd::__vmin_relu_16bit_x2(__lhs_u, __rhs_u);
        return ::cuda::std::simd::__copy_from_unsigned_storage<_Storage>(__result_u);
      }
      else
      {
        constexpr auto __size = ::cuda::std::simd::basic_vec<_Tp, _Abi>::size();
        _Storage __result{};
        _CCCL_PRAGMA_UNROLL_FULL()
        for (::cuda::std::simd::__simd_size_type __i = 0; __i < __size; ++__i)
        {
          __result.__data[__i] = ::__vimin_s32_relu(__lhs.__data[__i], __rhs.__data[__i]);
        }
        return __result;
      }
  }

  template <typename _Storage>
  [[nodiscard]] _CCCL_DEVICE_API constexpr _Storage
  operator()(const _Storage& __a, const _Storage& __b, const _Storage& __c) const noexcept
  {
    if constexpr (sizeof(_Tp) == sizeof(::cuda::std::int16_t))
    {
      const auto __a_u      = ::cuda::std::simd::__to_unsigned_storage(__a);
      const auto __b_u      = ::cuda::std::simd::__to_unsigned_storage(__b);
      const auto __c_u      = ::cuda::std::simd::__to_unsigned_storage(__c);
      const auto __result_u = ::cuda::simd::__vmin3_relu_16bit_x2(__a_u, __b_u, __c_u);
      return ::cuda::std::simd::__copy_from_unsigned_storage<_Storage>(__result_u);
    }
    else
    {
      constexpr auto __size = ::cuda::std::simd::basic_vec<_Tp, _Abi>::size();
      _Storage __result{};
      _CCCL_PRAGMA_UNROLL_FULL()
      for (::cuda::std::simd::__simd_size_type __i = 0; __i < __size; ++__i)
      {
        __result.__data[__i] = ::__vimin3_s32_relu(__a.__data[__i], __b.__data[__i], __c.__data[__i]);
      }
      return __result;
    }
  }
};

#endif // _CCCL_HAS_SIMD_MIN_MAX_RELU()

//! @brief Performs an element-wise maximum followed by ReLU.
//! @param[in] __lhs The left-hand side input vector.
//! @param[in] __rhs The right-hand side input vector.
//! @return A vector containing max(lhs[i],rhs[i],0) for each element.
_CCCL_TEMPLATE(typename _Tp, typename _Abi)
_CCCL_REQUIRES(::cuda::std::__cccl_is_signed_integer_v<_Tp>)
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr ::cuda::std::simd::basic_vec<_Tp, _Abi> max_relu(
  const ::cuda::std::simd::basic_vec<_Tp, _Abi>& __lhs, const ::cuda::std::simd::basic_vec<_Tp, _Abi>& __rhs) noexcept
{
#if _CCCL_HAS_SIMD_MIN_MAX_RELU()
  _CCCL_IF_NOT_CONSTEVAL_DEFAULT
  {
#  if _CCCL_HAS_SIMD_8BIT_PTX()
    if constexpr (sizeof(_Tp) == sizeof(::cuda::std::int8_t))
    {
#    if (__cccl_ptx_isa >= 940ULL)
      NV_IF_TARGET(NV_HAS_FEATURE_SM_107f,
                   (return __simd_min_max_relu_impl(__lhs, __rhs, __max_relu_operation<_Tp, _Abi>{});)) // ADL
#    endif // __cccl_ptx_isa >= 940ULL
      NV_IF_TARGET(NV_HAS_FEATURE_SM_120f,
                   (return __simd_min_max_relu_impl(__lhs, __rhs, __max_relu_operation<_Tp, _Abi>{});)) // ADL
    }
    else
#  endif // _CCCL_HAS_SIMD_8BIT_PTX()
      if constexpr (sizeof(_Tp) == sizeof(::cuda::std::int16_t) || sizeof(_Tp) == sizeof(::cuda::std::int32_t))
      {
        NV_IF_TARGET(NV_IS_DEVICE,
                     (return __simd_min_max_relu_impl(__lhs, __rhs, __max_relu_operation<_Tp, _Abi>{});)) // ADL
      }
  }
#endif // _CCCL_HAS_SIMD_MIN_MAX_RELU()

  using __vec_t = ::cuda::std::simd::basic_vec<_Tp, _Abi>;
  return ::cuda::std::simd::max(::cuda::std::simd::max(__lhs, __rhs), __vec_t{_Tp{0}});
}

//! @brief Performs an element-wise minimum followed by ReLU.
//! @param[in] __lhs The left-hand side input vector.
//! @param[in] __rhs The right-hand side input vector.
//! @return A vector containing max(min(lhs[i],rhs[i]),0) for each element.
_CCCL_TEMPLATE(typename _Tp, typename _Abi)
_CCCL_REQUIRES(::cuda::std::__cccl_is_signed_integer_v<_Tp>)
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr ::cuda::std::simd::basic_vec<_Tp, _Abi> min_relu(
  const ::cuda::std::simd::basic_vec<_Tp, _Abi>& __lhs, const ::cuda::std::simd::basic_vec<_Tp, _Abi>& __rhs) noexcept
{
#if _CCCL_HAS_SIMD_MIN_MAX_RELU()
  _CCCL_IF_NOT_CONSTEVAL_DEFAULT
  {
#  if _CCCL_HAS_SIMD_8BIT_PTX()
    if constexpr (sizeof(_Tp) == sizeof(::cuda::std::int8_t))
    {
#    if (__cccl_ptx_isa >= 940ULL)
      NV_IF_TARGET(NV_HAS_FEATURE_SM_107f,
                   (return __simd_min_max_relu_impl(__lhs, __rhs, __min_relu_operation<_Tp, _Abi>{});)) // ADL
#    endif // __cccl_ptx_isa >= 940ULL
      NV_IF_TARGET(NV_HAS_FEATURE_SM_120f,
                   (return __simd_min_max_relu_impl(__lhs, __rhs, __min_relu_operation<_Tp, _Abi>{});)) // ADL
    }
    else
#  endif // _CCCL_HAS_SIMD_8BIT_PTX()
      if constexpr (sizeof(_Tp) == sizeof(::cuda::std::int16_t) || sizeof(_Tp) == sizeof(::cuda::std::int32_t))
      {
        NV_IF_TARGET(NV_IS_DEVICE,
                     (return __simd_min_max_relu_impl(__lhs, __rhs, __min_relu_operation<_Tp, _Abi>{});)) // ADL
      }
  }
#endif // _CCCL_HAS_SIMD_MIN_MAX_RELU()

  using __vec_t = ::cuda::std::simd::basic_vec<_Tp, _Abi>;
  return ::cuda::std::simd::max(::cuda::std::simd::min(__lhs, __rhs), __vec_t{_Tp{0}});
}

//! @brief Performs an element-wise maximum of three vectors followed by ReLU.
//! @param[in] __a The first input vector.
//! @param[in] __b The second input vector.
//! @param[in] __c The third input vector.
//! @return A vector containing max(a[i],b[i],c[i],0) for each element.
_CCCL_TEMPLATE(typename _Tp, typename _Abi)
_CCCL_REQUIRES(::cuda::std::__cccl_is_signed_integer_v<_Tp>)
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr ::cuda::std::simd::basic_vec<_Tp, _Abi>
max_relu(const ::cuda::std::simd::basic_vec<_Tp, _Abi>& __a,
         const ::cuda::std::simd::basic_vec<_Tp, _Abi>& __b,
         const ::cuda::std::simd::basic_vec<_Tp, _Abi>& __c) noexcept
{
#if _CCCL_HAS_SIMD_MIN_MAX_RELU()
  _CCCL_IF_NOT_CONSTEVAL_DEFAULT
  {
#  if _CCCL_HAS_SIMD_8BIT_PTX()
    if constexpr (sizeof(_Tp) == sizeof(::cuda::std::int8_t))
    {
#    if (__cccl_ptx_isa >= 940ULL)
      NV_IF_TARGET(NV_HAS_FEATURE_SM_107f, (return ::cuda::simd::max_relu(::cuda::simd::max_relu(__a, __b), __c);))
#    endif // __cccl_ptx_isa >= 940ULL
      NV_IF_TARGET(NV_HAS_FEATURE_SM_120f, (return ::cuda::simd::max_relu(::cuda::simd::max_relu(__a, __b), __c);))
    }
    else
#  endif // _CCCL_HAS_SIMD_8BIT_PTX()
      if constexpr (sizeof(_Tp) == sizeof(::cuda::std::int16_t) || sizeof(_Tp) == sizeof(::cuda::std::int32_t))
      {
        NV_IF_TARGET(NV_IS_DEVICE,
                     (return __simd_min_max_relu_impl(__a, __b, __c, __max_relu_operation<_Tp, _Abi>{});)) // ADL
      }
  }
#endif // _CCCL_HAS_SIMD_MIN_MAX_RELU()

  using __vec_t = ::cuda::std::simd::basic_vec<_Tp, _Abi>;
  return ::cuda::std::simd::max(::cuda::std::simd::max(::cuda::std::simd::max(__a, __b), __c), __vec_t{_Tp{0}});
}

//! @brief Performs an element-wise minimum of three vectors followed by ReLU.
//! @param[in] __a The first input vector.
//! @param[in] __b The second input vector.
//! @param[in] __c The third input vector.
//! @return A vector containing max(min(a[i],b[i],c[i]),0) for each element.
_CCCL_TEMPLATE(typename _Tp, typename _Abi)
_CCCL_REQUIRES(::cuda::std::__cccl_is_signed_integer_v<_Tp>)
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr ::cuda::std::simd::basic_vec<_Tp, _Abi>
min_relu(const ::cuda::std::simd::basic_vec<_Tp, _Abi>& __a,
         const ::cuda::std::simd::basic_vec<_Tp, _Abi>& __b,
         const ::cuda::std::simd::basic_vec<_Tp, _Abi>& __c) noexcept
{
#if _CCCL_HAS_SIMD_MIN_MAX_RELU()
  _CCCL_IF_NOT_CONSTEVAL_DEFAULT
  {
#  if _CCCL_HAS_SIMD_8BIT_PTX()
    if constexpr (sizeof(_Tp) == sizeof(::cuda::std::int8_t))
    {
#    if (__cccl_ptx_isa >= 940ULL)
      NV_IF_TARGET(NV_HAS_FEATURE_SM_107f, (return ::cuda::simd::min_relu(::cuda::simd::min_relu(__a, __b), __c);))
#    endif // __cccl_ptx_isa >= 940ULL
      NV_IF_TARGET(NV_HAS_FEATURE_SM_120f, (return ::cuda::simd::min_relu(::cuda::simd::min_relu(__a, __b), __c);))
    }
    else
#  endif // _CCCL_HAS_SIMD_8BIT_PTX()
      if constexpr (sizeof(_Tp) == sizeof(::cuda::std::int16_t) || sizeof(_Tp) == sizeof(::cuda::std::int32_t))
      {
        NV_IF_TARGET(NV_IS_DEVICE,
                     (return __simd_min_max_relu_impl(__a, __b, __c, __min_relu_operation<_Tp, _Abi>{});)) // ADL
      }
  }
#endif // _CCCL_HAS_SIMD_MIN_MAX_RELU()

  using __vec_t = ::cuda::std::simd::basic_vec<_Tp, _Abi>;
  return ::cuda::std::simd::max(::cuda::std::simd::min(::cuda::std::simd::min(__a, __b), __c), __vec_t{_Tp{0}});
}

_CCCL_END_NAMESPACE_CUDA_SIMD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___SIMD_MIN_MAX_RELU_H
