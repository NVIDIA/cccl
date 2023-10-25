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

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA_PTX

// Private helper functions
inline _LIBCUDACXX_DEVICE _CUDA_VSTD::uint32_t __as_smem_ptr(const void* __ptr)
{
  return static_cast<_CUDA_VSTD::uint32_t>(__cvta_generic_to_shared(__ptr));
}
inline _LIBCUDACXX_DEVICE _CUDA_VSTD::uint32_t __as_remote_dsmem_ptr(const void* __ptr)
{
  return static_cast<_CUDA_VSTD::uint32_t>(__cvta_generic_to_shared(__ptr));
}
inline _LIBCUDACXX_DEVICE _CUDA_VSTD::uint64_t __as_gmem_ptr(const void* __ptr)
{
  return static_cast<_CUDA_VSTD::uint64_t>(__cvta_generic_to_global(__ptr));
}

template <typename _Tp>
inline _LIBCUDACXX_DEVICE int __as_b32(_Tp __val)
{
  static_assert(sizeof(_Tp) == 4, "");
  return *reinterpret_cast<int*>(&__val);
}

template <typename _Tp>
inline _LIBCUDACXX_DEVICE _CUDA_VSTD::uint64_t __as_b64(_Tp __val)
{
  static_assert(sizeof(_Tp) == 8, "");
  return *reinterpret_cast<_CUDA_VSTD::uint64_t*>(&__val);
}

_LIBCUDACXX_END_NAMESPACE_CUDA_PTX

#endif // _CUDA_PTX_HELPER_FUNCTIONS_H_
