//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD__CCCL_ALGORITHM_WRAPPER_H
#define _CUDA_STD__CCCL_ALGORITHM_WRAPPER_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// When nvc++ uses CCCL components as part of its implementation of
// Standard C++ algorithms, a cycle of included files may result when CCCL code
// tries to use a standard algorithm. The THRUST_INCLUDING_ALGORITHMS_HEADER macro
// is defined only when CCCL is including an algorithms-related header, giving
// the compiler a chance to detect and break the cycle of includes.

#if !_CCCL_COMPILER(NVRTC)
#  define THRUST_INCLUDING_ALGORITHMS_HEADER
#  include <algorithm>
#  undef THRUST_INCLUDING_ALGORITHMS_HEADER
#endif // !_CCCL_COMPILER(NVRTC)

#endif // _CUDA_STD__CCCL_ALGORITHM_WRAPPER_H
