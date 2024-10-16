//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if defined(__CUDACC__) && !defined(_NVHPC_CUDA)
#  define CUDASTF_HOST_DEVICE __host__ __device__
#  define CUDASTF_DEVICE      __device__
#  define CUDASTF_HOST        __host__
#  define CUDASTF_NO_DEVICE_STACK
#elif defined(__CUDACC__) && defined(_NVHPC_CUDA)
#  define CUDASTF_HOST_DEVICE
#  define CUDASTF_DEVICE
#  define CUDASTF_HOST
#  define CUDASTF_NO_DEVICE_STACK _Pragma("diag_suppress no_device_stack")
#else
#  define CUDASTF_HOST_DEVICE
#  define CUDASTF_DEVICE
#  define CUDASTF_HOST
#  define CUDASTF_NO_DEVICE_STACK
#endif

#if __CUDA_ARCH__ >= 700
#  define CUDASTF_GRID_CONSTANT __grid_constant__
#else
#  define CUDASTF_GRID_CONSTANT
#endif
