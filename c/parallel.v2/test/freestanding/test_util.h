//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef C_PARALLEL_FREESTANDING_TEST_UTIL_H
#define C_PARALLEL_FREESTANDING_TEST_UTIL_H

#include <cassert>
#include <iostream>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                                                       \
  do                                                                                                           \
  {                                                                                                            \
    cudaError_t err = call;                                                                                    \
    if (err != cudaSuccess)                                                                                    \
    {                                                                                                          \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(err) << "\n"; \
      return 1;                                                                                                \
    }                                                                                                          \
  } while (0)

#endif // C_PARALLEL_FREESTANDING_TEST_UTIL_H
