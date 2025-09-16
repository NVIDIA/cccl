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

struct MyStruct
{
  int v;
};

__device__ int global_var;
__constant__ int constant_var;

__global__ void test_kernel(const _CCCL_GRID_CONSTANT MyStruct grid_constant_var)
{
  using cuda::device::address_space;
  using cuda::device::is_address_from;
  using cuda::device::is_object_from;
  __shared__ int shared_var;
  int local_var;

  // 1. Test non-volatile pointers/objects
  {
    assert(is_address_from(&global_var, address_space::global));
    assert(is_address_from(&shared_var, address_space::shared));
    assert(is_address_from(&constant_var, address_space::constant));
    assert(is_address_from(&local_var, address_space::local));
    assert(is_address_from(&grid_constant_var, address_space::grid_constant) == _CCCL_HAS_GRID_CONSTANT());
    // todo: test address_space::cluster_shared

    assert(is_object_from(global_var, address_space::global));
    assert(is_object_from(shared_var, address_space::shared));
    assert(is_object_from(constant_var, address_space::constant));
    assert(is_object_from(local_var, address_space::local));
    assert(is_object_from(grid_constant_var, address_space::grid_constant) == _CCCL_HAS_GRID_CONSTANT());
    // todo: test address_space::cluster_shared
  }

  // 2. Test volatile pointers/objects
  {
    volatile auto& v_global_var        = global_var;
    volatile auto& v_shared_var        = shared_var;
    volatile auto& v_constant_var      = constant_var;
    volatile auto& v_local_var         = local_var;
    volatile auto& v_grid_constant_var = grid_constant_var;

    assert(is_address_from(&v_global_var, address_space::global));
    assert(is_address_from(&v_shared_var, address_space::shared));
    assert(is_address_from(&v_constant_var, address_space::constant));
    assert(is_address_from(&v_local_var, address_space::local));
    assert(is_address_from(&v_grid_constant_var, address_space::grid_constant) == _CCCL_HAS_GRID_CONSTANT());
    // todo: test address_space::cluster_shared

    assert(is_object_from(v_global_var, address_space::global));
    assert(is_object_from(v_shared_var, address_space::shared));
    assert(is_object_from(v_constant_var, address_space::constant));
    assert(is_object_from(v_local_var, address_space::local));
    assert(is_object_from(v_grid_constant_var, address_space::grid_constant) == _CCCL_HAS_GRID_CONSTANT());
    // todo: test address_space::cluster_shared
  }
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (test_kernel<<<1, 1>>>(MyStruct{}); assert(cudaDeviceSynchronize() == cudaSuccess);))
  return 0;
}
