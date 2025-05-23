//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/memory>

using cuda::device::address_space;
using cuda::device::is_address_from;

template <address_space exp_space>
__device__ void test_is_address_from(const void* ptr)
{
  assert(is_address_from(address_space::global, ptr) == (exp_space == address_space::global));
  assert(is_address_from(address_space::shared, ptr) == (exp_space == address_space::shared));
  assert(is_address_from(address_space::constant, ptr) == (exp_space == address_space::constant));
  assert(is_address_from(address_space::local, ptr) == (exp_space == address_space::local));
  assert(is_address_from(address_space::grid_constant, ptr) == (exp_space == address_space::grid_constant));
  assert(is_address_from(address_space::cluster_shared, ptr) == (exp_space == address_space::cluster_shared));
}

struct MutableStruct
{
  mutable int v;
};

__device__ int global_var;
__constant__ int constant_var;

__global__ void test_kernel(const _CCCL_GRID_CONSTANT MutableStruct grid_constant_var)
{
  __shared__ int shared_var;
  int local_var;

  test_is_address_from<address_space::global>(&global_var);
  test_is_address_from<address_space::shared>(&shared_var);
  test_is_address_from<address_space::constant>(&constant_var);
  test_is_address_from<address_space::local>(&local_var);

  // __grid_constant__ address may come from other address spaces, so we test the variable address comes from the
  // address_space::grid_constant
  assert(is_address_from(address_space::grid_constant, &grid_constant_var) == _CCCL_HAS_GRID_CONSTANT());

  // todo: test address_space::cluster_shared
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (test_kernel<<<1, 1>>>(MutableStruct{}); assert(cudaDeviceSynchronize() == cudaSuccess);))
  return 0;
}
