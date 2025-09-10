//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/devices>

#include <testing.cuh>

__device__ int foo(const int& x)
{
  return x;
}

template <cuda::arch::id Arch>
__global__ void arch_specific_kernel_mock_do_not_launch()
{
  NV_IF_TARGET(
    NV_PROVIDES_SM_60,
    (
      // I will try to pack something like this into an API
      if constexpr (cuda::arch::traits<Arch>().compute_capability != cuda::arch::current_traits().compute_capability) {
        return;
      }

      [[maybe_unused]] __shared__ int array[cuda::arch::traits<Arch>().max_shared_memory_per_block / sizeof(int)];

      // constexpr is useless and I can't use intrinsics here :(
      if constexpr (cuda::arch::current_traits().cluster_supported) {
        [[maybe_unused]] int dummy;
        asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(dummy));
      } if constexpr (cuda::arch::current_traits().redux_intrinisic) {
        [[maybe_unused]] int dummy1 = 0, dummy2 = 0;
        asm volatile("redux.sync.add.s32 %0, %1, 0xffffffff;" : "=r"(dummy1) : "r"(dummy2));
      } if constexpr (cuda::arch::current_traits().cp_async_supported) { asm volatile("cp.async.commit_group;"); }

      // Confirm trait value is defined device code and usable as a reference
      foo(cuda::arch::traits<Arch>().compute_capability);
      foo(cuda::arch::current_traits().compute_capability);))
}

template __global__ void arch_specific_kernel_mock_do_not_launch<cuda::arch::id::sm_70>();
template __global__ void arch_specific_kernel_mock_do_not_launch<cuda::arch::id::sm_75>();
template __global__ void arch_specific_kernel_mock_do_not_launch<cuda::arch::id::sm_80>();
template __global__ void arch_specific_kernel_mock_do_not_launch<cuda::arch::id::sm_86>();
template __global__ void arch_specific_kernel_mock_do_not_launch<cuda::arch::id::sm_89>();
template __global__ void arch_specific_kernel_mock_do_not_launch<cuda::arch::id::sm_90>();
template __global__ void arch_specific_kernel_mock_do_not_launch<cuda::arch::id::sm_100>();
template __global__ void arch_specific_kernel_mock_do_not_launch<cuda::arch::id::sm_103>();
template __global__ void arch_specific_kernel_mock_do_not_launch<cuda::arch::id::sm_110>();
template __global__ void arch_specific_kernel_mock_do_not_launch<cuda::arch::id::sm_120>();
template __global__ void arch_specific_kernel_mock_do_not_launch<cuda::arch::id::sm_90a>();
template __global__ void arch_specific_kernel_mock_do_not_launch<cuda::arch::id::sm_100a>();
template __global__ void arch_specific_kernel_mock_do_not_launch<cuda::arch::id::sm_103a>();
template __global__ void arch_specific_kernel_mock_do_not_launch<cuda::arch::id::sm_110a>();
template __global__ void arch_specific_kernel_mock_do_not_launch<cuda::arch::id::sm_120a>();

template <unsigned int ComputeCapability>
void constexpr compare_static_and_dynamic()
{
  constexpr cuda::arch::traits_t static_traits =
    cuda::arch::traits<cuda::arch::id_for_compute_capability(ComputeCapability)>();
  constexpr cuda::arch::traits_t dynamic_traits = cuda::arch::traits_for_compute_capability(ComputeCapability);

  static_assert(static_traits.arch_id == dynamic_traits.arch_id);
  static_assert(static_traits.max_threads_per_block == dynamic_traits.max_threads_per_block);
  static_assert(static_traits.max_block_dim_x == dynamic_traits.max_block_dim_x);
  static_assert(static_traits.max_block_dim_y == dynamic_traits.max_block_dim_y);
  static_assert(static_traits.max_block_dim_z == dynamic_traits.max_block_dim_z);
  static_assert(static_traits.max_grid_dim_x == dynamic_traits.max_grid_dim_x);
  static_assert(static_traits.max_grid_dim_y == dynamic_traits.max_grid_dim_y);
  static_assert(static_traits.max_grid_dim_z == dynamic_traits.max_grid_dim_z);

  static_assert(static_traits.warp_size == dynamic_traits.warp_size);
  static_assert(static_traits.total_constant_memory == dynamic_traits.total_constant_memory);
  static_assert(static_traits.max_resident_grids == dynamic_traits.max_resident_grids);
  static_assert(static_traits.max_shared_memory_per_block == dynamic_traits.max_shared_memory_per_block);
  static_assert(static_traits.gpu_overlap == dynamic_traits.gpu_overlap);
  static_assert(static_traits.can_map_host_memory == dynamic_traits.can_map_host_memory);
  static_assert(static_traits.concurrent_kernels == dynamic_traits.concurrent_kernels);
  static_assert(static_traits.stream_priorities_supported == dynamic_traits.stream_priorities_supported);
  static_assert(static_traits.global_l1_cache_supported == dynamic_traits.global_l1_cache_supported);
  static_assert(static_traits.local_l1_cache_supported == dynamic_traits.local_l1_cache_supported);
  static_assert(static_traits.max_registers_per_block == dynamic_traits.max_registers_per_block);
  static_assert(static_traits.max_registers_per_multiprocessor == dynamic_traits.max_registers_per_multiprocessor);

  static_assert(static_traits.compute_capability == dynamic_traits.compute_capability);
  static_assert(static_traits.compute_capability_major == dynamic_traits.compute_capability_major);
  static_assert(static_traits.compute_capability_minor == dynamic_traits.compute_capability_minor);
  static_assert(static_traits.compute_capability == dynamic_traits.compute_capability);
  static_assert(
    static_traits.max_shared_memory_per_multiprocessor == dynamic_traits.max_shared_memory_per_multiprocessor);
  static_assert(static_traits.max_blocks_per_multiprocessor == dynamic_traits.max_blocks_per_multiprocessor);
  static_assert(static_traits.max_warps_per_multiprocessor == dynamic_traits.max_warps_per_multiprocessor);
  static_assert(static_traits.max_threads_per_multiprocessor == dynamic_traits.max_threads_per_multiprocessor);
  static_assert(static_traits.reserved_shared_memory_per_block == dynamic_traits.reserved_shared_memory_per_block);
  static_assert(static_traits.max_shared_memory_per_block_optin == dynamic_traits.max_shared_memory_per_block_optin);
  static_assert(static_traits.cluster_supported == dynamic_traits.cluster_supported);
  static_assert(static_traits.redux_intrinisic == dynamic_traits.redux_intrinisic);
  static_assert(static_traits.elect_intrinsic == dynamic_traits.elect_intrinsic);
  static_assert(static_traits.cp_async_supported == dynamic_traits.cp_async_supported);
  static_assert(static_traits.tma_supported == dynamic_traits.tma_supported);
}

C2H_CCCLRT_TEST("Traits", "[device]")
{
  compare_static_and_dynamic<70>();
  compare_static_and_dynamic<75>();
  compare_static_and_dynamic<80>();
  compare_static_and_dynamic<86>();
  compare_static_and_dynamic<89>();
  compare_static_and_dynamic<90>();
  compare_static_and_dynamic<100>();
  compare_static_and_dynamic<103>();
  compare_static_and_dynamic<110>();
  compare_static_and_dynamic<120>();

  // Compare arch traits with attributes
  for (const cuda::physical_device& dev : cuda::devices)
  {
    auto traits = dev.arch_traits();

    CCCLRT_REQUIRE(traits.max_threads_per_block == dev.attribute(cuda::device_attributes::max_threads_per_block));
    CCCLRT_REQUIRE(traits.max_block_dim_x == dev.attribute(cuda::device_attributes::max_block_dim_x));
    CCCLRT_REQUIRE(traits.max_block_dim_y == dev.attribute(cuda::device_attributes::max_block_dim_y));
    CCCLRT_REQUIRE(traits.max_block_dim_z == dev.attribute(cuda::device_attributes::max_block_dim_z));
    CCCLRT_REQUIRE(traits.max_grid_dim_x == dev.attribute(cuda::device_attributes::max_grid_dim_x));
    CCCLRT_REQUIRE(traits.max_grid_dim_y == dev.attribute(cuda::device_attributes::max_grid_dim_y));
    CCCLRT_REQUIRE(traits.max_grid_dim_z == dev.attribute(cuda::device_attributes::max_grid_dim_z));

    CCCLRT_REQUIRE(traits.warp_size == dev.attribute(cuda::device_attributes::warp_size));
    CCCLRT_REQUIRE(traits.total_constant_memory == dev.attribute(cuda::device_attributes::total_constant_memory));
    CCCLRT_REQUIRE(
      traits.max_shared_memory_per_block == dev.attribute(cuda::device_attributes::max_shared_memory_per_block));
    CCCLRT_REQUIRE(traits.gpu_overlap == dev.attribute(cuda::device_attributes::gpu_overlap));
    CCCLRT_REQUIRE(traits.can_map_host_memory == dev.attribute(cuda::device_attributes::can_map_host_memory));
    CCCLRT_REQUIRE(traits.concurrent_kernels == dev.attribute(cuda::device_attributes::concurrent_kernels));
    CCCLRT_REQUIRE(
      traits.stream_priorities_supported == dev.attribute(cuda::device_attributes::stream_priorities_supported));
    CCCLRT_REQUIRE(
      traits.global_l1_cache_supported == dev.attribute(cuda::device_attributes::global_l1_cache_supported));
    CCCLRT_REQUIRE(traits.local_l1_cache_supported == dev.attribute(cuda::device_attributes::local_l1_cache_supported));
    CCCLRT_REQUIRE(traits.max_registers_per_block == dev.attribute(cuda::device_attributes::max_registers_per_block));
    CCCLRT_REQUIRE(traits.max_registers_per_multiprocessor
                   == dev.attribute(cuda::device_attributes::max_registers_per_multiprocessor));
    CCCLRT_REQUIRE(traits.compute_capability_major == dev.attribute(cuda::device_attributes::compute_capability_major));
    CCCLRT_REQUIRE(traits.compute_capability_minor == dev.attribute(cuda::device_attributes::compute_capability_minor));
    CCCLRT_REQUIRE(traits.compute_capability == dev.attribute(cuda::device_attributes::compute_capability));
    CCCLRT_REQUIRE(traits.max_shared_memory_per_multiprocessor
                   == dev.attribute(cuda::device_attributes::max_shared_memory_per_multiprocessor));
    CCCLRT_REQUIRE(
      traits.max_blocks_per_multiprocessor == dev.attribute(cuda::device_attributes::max_blocks_per_multiprocessor));
    CCCLRT_REQUIRE(
      traits.max_threads_per_multiprocessor == dev.attribute(cuda::device_attributes::max_threads_per_multiprocessor));
    CCCLRT_REQUIRE(traits.reserved_shared_memory_per_block
                   == dev.attribute(cuda::device_attributes::reserved_shared_memory_per_block));
    CCCLRT_REQUIRE(traits.max_shared_memory_per_block_optin
                   == dev.attribute(cuda::device_attributes::max_shared_memory_per_block_optin));
  }
}
