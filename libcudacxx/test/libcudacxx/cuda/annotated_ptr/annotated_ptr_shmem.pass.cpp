//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: pre-sm-70
// UNSUPPORTED: !nvcc
// UNSUPPORTED: nvrtc
// UNSUPPORTED: c++98, c++03

#include "utils.h"

template <typename T, typename U>
__device__ __host__ __noinline__ void shared_mem_test_dev()
{
  T* smem  = alloc<T, 128>(true);
  smem[10] = 42;

  cuda::annotated_ptr<U, cuda::access_property::shared> p{smem + 10};

  assert(*p == 42);
}

__device__ __host__ __noinline__ void all_tests()
{
  shared_mem_test_dev<int, int>();
  shared_mem_test_dev<int, const int>();
  shared_mem_test_dev<int, volatile int>();
  shared_mem_test_dev<int, const volatile int>();
}

__global__ void shared_mem_test()
{
  all_tests();
};

// TODO: is this needed?
__device__ __host__ __noinline__ void test_all()
{
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE, (all_tests();), (shared_mem_test<<<1, 1, 0, 0>>>(); assert_rt(cudaStreamSynchronize(0));))
}

int main(int argc, char** argv)
{
  test_all();
  return 0;
}
