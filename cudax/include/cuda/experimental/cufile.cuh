//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#ifndef __CUDAX_CUFILE_CUH
#define __CUDAX_CUFILE_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_OS(WINDOWS)
#  error "<cuda/cufile> is not supported on Windows"
#endif // _CCCL_OS(WINDOWS)

#if !_CCCL_HAS_INCLUDE(<cufile.h>)
#  error "<cuda/cufile> requires libcufile-dev package to be installed"
#endif // !_CCCL_HAS_INCLUDE(<cufile.h>)

#if _CCCL_CTK_BELOW(12, 9)
#  error "<cuda/cufile> requires at least CUDA 12.9"
#endif // _CCCL_CTK_BELOW(12, 9)

#include <cuda/experimental/__cufile/cufile.cuh>
#include <cuda/experimental/__cufile/cufile_ref.cuh>
#include <cuda/experimental/__cufile/driver.cuh>
#include <cuda/experimental/__cufile/driver_attributes.cuh>
#include <cuda/experimental/__cufile/exception.cuh>
#include <cuda/experimental/__cufile/open_mode.cuh>

#endif // __CUDAX_CUFILE_CUH
