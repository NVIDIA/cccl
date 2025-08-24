//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA__STD__CUDA_ENSURE_CURRENT_DEVICE_H
#define _CUDA__STD__CUDA_ENSURE_CURRENT_DEVICE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK()

#  include <cuda/std/__cuda/api_wrapper.h>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

#  if !_CCCL_COMPILER(NVRTC)

//! @brief `__ensure_current_device` is a simple helper that the current device is set to the right one.
//! Only changes the current device if the target device is not the current one
struct __ensure_current_device
{
  int __target_device_   = 0;
  int __original_device_ = 0;

  //! @brief Queries the current device and if that is different than \p __target_device sets the current device to
  //! \p __target_device
  __ensure_current_device(const int __target_device)
      : __target_device_(__target_device)
  {
    _CCCL_TRY_CUDA_API(::cudaGetDevice, "Failed to query current device", &__original_device_);
    if (__original_device_ != __target_device_)
    {
      _CCCL_TRY_CUDA_API(::cudaSetDevice, "Failed to set device", __target_device_);
    }
  }

  //! @brief If the \p __original_device was not equal to \p __target_device sets the current device back to
  //! \p __original_device
  ~__ensure_current_device()
  {
    if (__original_device_ != __target_device_)
    {
      _CCCL_TRY_CUDA_API(::cudaSetDevice, "Failed to set device", __original_device_);
    }
  }
};

#  endif // !_CCCL_COMPILER(NVRTC)

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK()

#endif //_CUDA__STD__CUDA_ENSURE_CURRENT_DEVICE_H
