//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _TEST_LIBCUDACXX_CUDA_FUNCTIONAL_MIN_MAX_COMMON_H
#define _TEST_LIBCUDACXX_CUDA_FUNCTIONAL_MIN_MAX_COMMON_H

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#define CUDA_SAFE_CALL(...)                                                           \
  do                                                                                  \
  {                                                                                   \
    cudaError_t err = __VA_ARGS__;                                                    \
    if (err != cudaSuccess)                                                           \
    {                                                                                 \
      printf("CUDA ERROR: %s: %s\n", cudaGetErrorName(err), cudaGetErrorString(err)); \
      std::exit(EXIT_FAILURE);                                                        \
    }                                                                                 \
  } while (false)

namespace
{

template <typename OpT, typename T>
__host__ __device__ void test_op(const T& lhs, const T& rhs, const T& expected)
{
  const auto op = OpT{};

  assert(op(lhs, rhs) == expected);

  if (rhs == lhs)
  {
    assert(op(lhs, rhs) == op(rhs, lhs));
  }
  else
  {
    assert(op(lhs, rhs) != op(rhs, lhs));
  }
}

} // namespace

#endif // _TEST_LIBCUDACXX_CUDA_FUNCTIONAL_MIN_MAX_COMMON_H
