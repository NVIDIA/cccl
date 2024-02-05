// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_PTX_HELPER_FUNCTIONS_H_
#define _CUDA_PTX_HELPER_FUNCTIONS_H_

#include "../../cstdint"        // uint32_t
#include "../../cstddef"        // size_t
#include "../../__type_traits/integral_constant.h"

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA_PTX

template <int __n>
using n32_t = _CUDA_VSTD::integral_constant<int, __n>;

/*************************************************************
 *
 * Conversion from generic pointer -> state space "pointer"
 *
 **************************************************************/
inline _LIBCUDACXX_DEVICE _CUDA_VSTD::uint32_t __as_ptr_smem(const void* __ptr)
{
  // Consider adding debug asserts here.
  return static_cast<_CUDA_VSTD::uint32_t>(__cvta_generic_to_shared(__ptr));
}

inline _LIBCUDACXX_DEVICE _CUDA_VSTD::uint32_t __as_ptr_dsmem(const void* __ptr)
{
  // No difference in implementation to __as_ptr_smem.
  // Consider adding debug asserts here.
  return __as_ptr_smem(__ptr);
}

inline _LIBCUDACXX_DEVICE _CUDA_VSTD::uint32_t __as_ptr_remote_dsmem(const void* __ptr)
{
  // No difference in implementation to __as_ptr_smem.
  // Consider adding debug asserts here.
  return __as_ptr_smem(__ptr);
}

inline _LIBCUDACXX_DEVICE _CUDA_VSTD::uint64_t __as_ptr_gmem(const void* __ptr)
{
  // Consider adding debug asserts here.
  return static_cast<_CUDA_VSTD::uint64_t>(__cvta_generic_to_global(__ptr));
}


/*************************************************************
 *
 * Conversion from state space "pointer" -> generic pointer
 *
 **************************************************************/
template <typename _Tp>
inline _LIBCUDACXX_DEVICE _Tp* __from_ptr_smem(_CUDA_VSTD::size_t __ptr)
{
  // Consider adding debug asserts here.
  return reinterpret_cast<_Tp*>(__cvta_shared_to_generic(__ptr));
}

template <typename _Tp>
inline _LIBCUDACXX_DEVICE _Tp* __from_ptr_dsmem(_CUDA_VSTD::size_t __ptr)
{
  // Consider adding debug asserts here.
  return __from_ptr_smem<_Tp>(__ptr);
}

template <typename _Tp>
inline _LIBCUDACXX_DEVICE _Tp* __from_ptr_remote_dsmem(_CUDA_VSTD::size_t __ptr)
{
  // Consider adding debug asserts here.
  return __from_ptr_smem<_Tp>(__ptr);
}

template <typename _Tp>
inline _LIBCUDACXX_DEVICE _Tp* __from_ptr_gmem(_CUDA_VSTD::size_t __ptr)
{
  // Consider adding debug asserts here.
  return reinterpret_cast<_Tp*>(__cvta_global_to_generic(__ptr));
}


/*************************************************************
 *
 * Conversion from template type -> concrete binary type
 *
 **************************************************************/
template <typename _Tp>
inline _LIBCUDACXX_DEVICE _CUDA_VSTD::uint32_t __as_b32(_Tp __val)
{
#if _CCCL_STD_VER >= 2017 && !defined(_LIBCUDACXX_HAS_NO_CXX14_CONSTEXPR)
  static_assert(sizeof(_Tp) == 4, "");
#endif // _CCCL_STD_VER >= 2017  && !_LIBCUDACXX_HAS_NO_CXX14_CONSTEXPR
  // Consider using std::bitcast
  return *reinterpret_cast<_CUDA_VSTD::uint32_t*>(&__val);
}

template <typename _Tp>
inline _LIBCUDACXX_DEVICE _CUDA_VSTD::uint64_t __as_b64(_Tp __val)
{
#if _CCCL_STD_VER >= 2017 && !defined(_LIBCUDACXX_HAS_NO_CXX14_CONSTEXPR)
  static_assert(sizeof(_Tp) == 8, "");
#endif // _CCCL_STD_VER >= 2017 && !_LIBCUDACXX_HAS_NO_CXX14_CONSTEXPR
  // Consider using std::bitcast
  return *reinterpret_cast<_CUDA_VSTD::uint64_t*>(&__val);
}

_LIBCUDACXX_END_NAMESPACE_CUDA_PTX

#endif // _CUDA_PTX_HELPER_FUNCTIONS_H_
