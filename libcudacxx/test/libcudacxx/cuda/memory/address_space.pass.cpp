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
  int local_var{};

  // For pre sm_90 archs, this other_block_shared_var will just reference shared_var, which satisfies the
  // address_space::cluster_shared, too.
  int* other_block_shared_var_ptr = &shared_var;
  NV_IF_TARGET(NV_PROVIDES_SM_90, ({
                 const auto rank_in_cluster            = __clusterRelativeBlockRank();
                 const auto other_rank_rank_in_cluster = (rank_in_cluster + 1) % __clusterSizeInBlocks();
                 other_block_shared_var_ptr =
                   static_cast<int*>(__cluster_map_shared_rank(&shared_var, other_rank_rank_in_cluster));
               }))
  int& other_block_shared_var = *other_block_shared_var_ptr;

  // 1. Test non-volatile pointers/objects
  {
    assert(is_address_from(&global_var, address_space::global));
    assert(is_address_from(&shared_var, address_space::shared));
    assert(is_address_from(&constant_var, address_space::constant));
    assert(is_address_from(&local_var, address_space::local));
    assert(is_address_from(&grid_constant_var, address_space::grid_constant) == _CCCL_HAS_GRID_CONSTANT());
    assert(is_address_from(&shared_var, address_space::cluster_shared));
    assert(is_address_from(&other_block_shared_var, address_space::cluster_shared));

    assert(is_object_from(global_var, address_space::global));
    assert(is_object_from(shared_var, address_space::shared));
    assert(is_object_from(constant_var, address_space::constant));
    assert(is_object_from(local_var, address_space::local));
    assert(is_object_from(grid_constant_var, address_space::grid_constant) == _CCCL_HAS_GRID_CONSTANT());
    assert(is_object_from(shared_var, address_space::cluster_shared));
    assert(is_object_from(other_block_shared_var, address_space::cluster_shared));
  }

  // 2. Test volatile pointers/objects
  {
    volatile auto& v_global_var             = global_var;
    volatile auto& v_shared_var             = shared_var;
    volatile auto& v_constant_var           = constant_var;
    volatile auto& v_local_var              = local_var;
    volatile auto& v_grid_constant_var      = grid_constant_var;
    volatile auto& v_other_block_shared_var = other_block_shared_var;

    assert(is_address_from(&v_global_var, address_space::global));
    assert(is_address_from(&v_shared_var, address_space::shared));
    assert(is_address_from(&v_constant_var, address_space::constant));
    assert(is_address_from(&v_local_var, address_space::local));
    assert(is_address_from(&v_grid_constant_var, address_space::grid_constant) == _CCCL_HAS_GRID_CONSTANT());
    assert(is_address_from(&v_shared_var, address_space::cluster_shared));
    assert(is_address_from(&v_other_block_shared_var, address_space::cluster_shared));

    assert(is_object_from(v_global_var, address_space::global));
    assert(is_object_from(v_shared_var, address_space::shared));
    assert(is_object_from(v_constant_var, address_space::constant));
    assert(is_object_from(v_local_var, address_space::local));
    assert(is_object_from(v_grid_constant_var, address_space::grid_constant) == _CCCL_HAS_GRID_CONSTANT());
    assert(is_object_from(v_shared_var, address_space::cluster_shared));
    assert(is_object_from(v_other_block_shared_var, address_space::cluster_shared));
  }
}

#if !_CCCL_COMPILER(NVRTC)
void test()
{
  int cc_major{};
  assert(cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, 0) == cudaSuccess);

  cudaLaunchAttribute launch_attrs[1]{};
  launch_attrs[0].id               = cudaLaunchAttributeClusterDimension;
  launch_attrs[0].val.clusterDim.x = 2;
  launch_attrs[0].val.clusterDim.y = 1;
  launch_attrs[0].val.clusterDim.z = 1;

  cudaLaunchConfig_t launch_config{};
  launch_config.gridDim  = 2;
  launch_config.blockDim = 1;
  launch_config.attrs    = launch_attrs;
  launch_config.numAttrs = (cc_major >= 9) ? 1 : 0;

  MyStruct my_struct{};

  void* args[]{&my_struct};
  assert(cudaLaunchKernelExC(&launch_config, (const void*) test_kernel, args) == cudaSuccess);
  assert(cudaDeviceSynchronize() == cudaSuccess);
}
#endif // !_CCCL_COMPILER(NVRTC)

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (test();))
  return 0;
}
