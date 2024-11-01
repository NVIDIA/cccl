//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/functional>
#include <cuda/std/cassert>

#include "min_max_common.h"
#include "test_macros.h"

namespace
{

template <typename OpT, typename T>
__global__ test_kernel(T lhs, T rhs, T expected)
{
  test_op<OpT>(lhs, rhs, expected);
}

template <typename T>
void test_device(const T& lhs, const T& rhs, const T& expected)
{
  test_kernel<cuda::minimum<T>><<<1, 1>>>(lhs, rhs, expected);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  test_kernel<cuda::minimum<>><<<1, 1>>>(lhs, rhs, expected);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  test_kernel<cuda::minimum<void>><<<1, 1>>>(lhs, rhs, expected);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
}

#if !defined(TEST_COMPILER_NVRTC)
template <typename T>
void test_host(const T& lhs, const T& rhs, const T& expected)
{
  test_op<cuda::minimum<T>>(lhs, rhs, expected);
  test_op<cuda::minimum<>>(lhs, rhs, expected);
  test_op<cuda::minimum<void>>(lhs, rhs, expected);
}
#endif

template <typename T>
__host__ __device__ void test(T lhs, T rhs, T expected)
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, test_device(lhs, rhs, expected), test_host(lhs, rhs, expected));
}

} // namespace

int main(int, char**)
{
  test(0, 1, 0);
  test(1, 0, 0);
  test(0, 0, 0);
  test(-1, 1, -1);
  test(1.0F, 2.0F, 1.0F);
  return 0;
}
