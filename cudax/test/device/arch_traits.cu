//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/device.cuh>

#include <testing.cuh>

template <typename Arch>
__global__ void arch_specific_kernel_mock_do_not_launch()
{
  // I will try to pack something like this into an API
  if constexpr (Arch::compute_capability != cudax::current_arch().compute_capability)
  {
    return;
  }

  [[maybe_unused]] __shared__ int array[Arch::max_shared_memory_per_block / sizeof(int)];

  // constexpr is useless and I can't use intrinsics here :(
  if constexpr (cudax::current_arch().cluster_supported)
  {
    [[maybe_unused]] int dummy;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(dummy));
  }
  if constexpr (cudax::current_arch().redux_intrinisic)
  {
    [[maybe_unused]] int dummy1 = 0, dummy2 = 0;
    asm volatile("redux.sync.add.s32 %0, %1, 0xffffffff;" : "=r"(dummy1) : "r"(dummy2));
  }
  if constexpr (cudax::current_arch().cp_async_supported)
  {
    asm volatile("cp.async.commit_group;");
  }
}

template __global__ void arch_specific_kernel_mock_do_not_launch<cudax::arch<700>>();
template __global__ void arch_specific_kernel_mock_do_not_launch<cudax::arch<750>>();
template __global__ void arch_specific_kernel_mock_do_not_launch<cudax::arch<800>>();
template __global__ void arch_specific_kernel_mock_do_not_launch<cudax::arch<860>>();
template __global__ void arch_specific_kernel_mock_do_not_launch<cudax::arch<890>>();
template __global__ void arch_specific_kernel_mock_do_not_launch<cudax::arch<900>>();

template <unsigned int Arch>
void constexpr compare_static_and_dynamic()
{
  using StaticTraits                            = cudax::arch<Arch>;
  constexpr cudax::arch_traits_t dynamic_traits = cudax::arch_traits(Arch);

  static_assert(sizeof(StaticTraits) == 1);

  static_assert(StaticTraits::max_threads_per_block == dynamic_traits.max_threads_per_block);
  static_assert(StaticTraits::max_block_dim_x == dynamic_traits.max_block_dim_x);
  static_assert(StaticTraits::max_block_dim_y == dynamic_traits.max_block_dim_y);
  static_assert(StaticTraits::max_block_dim_z == dynamic_traits.max_block_dim_z);
  static_assert(StaticTraits::max_grid_dim_x == dynamic_traits.max_grid_dim_x);
  static_assert(StaticTraits::max_grid_dim_y == dynamic_traits.max_grid_dim_y);
  static_assert(StaticTraits::max_grid_dim_z == dynamic_traits.max_grid_dim_z);

  static_assert(StaticTraits::warp_size == dynamic_traits.warp_size);
  static_assert(StaticTraits::total_constant_memory == dynamic_traits.total_constant_memory);
  static_assert(StaticTraits::max_resident_grids == dynamic_traits.max_resident_grids);
  static_assert(StaticTraits::max_shared_memory_per_block == dynamic_traits.max_shared_memory_per_block);
  static_assert(StaticTraits::gpu_overlap == dynamic_traits.gpu_overlap);
  static_assert(StaticTraits::can_map_host_memory == dynamic_traits.can_map_host_memory);
  static_assert(StaticTraits::concurrent_kernels == dynamic_traits.concurrent_kernels);
  static_assert(StaticTraits::stream_priorities_supported == dynamic_traits.stream_priorities_supported);
  static_assert(StaticTraits::global_l1_cache_supported == dynamic_traits.global_l1_cache_supported);
  static_assert(StaticTraits::local_l1_cache_supported == dynamic_traits.local_l1_cache_supported);
  static_assert(StaticTraits::max_registers_per_block == dynamic_traits.max_registers_per_block);
  static_assert(StaticTraits::max_registers_per_multiprocessor == dynamic_traits.max_registers_per_multiprocessor);
  static_assert(StaticTraits::max_registers_per_thread == dynamic_traits.max_registers_per_thread);

  static_assert(StaticTraits::compute_capability_major == dynamic_traits.compute_capability_major);
  static_assert(StaticTraits::compute_capability_minor == dynamic_traits.compute_capability_minor);
  static_assert(StaticTraits::compute_capability == dynamic_traits.compute_capability);
  static_assert(
    StaticTraits::max_shared_memory_per_multiprocessor == dynamic_traits.max_shared_memory_per_multiprocessor);
  static_assert(StaticTraits::max_blocks_per_multiprocessor == dynamic_traits.max_blocks_per_multiprocessor);
  static_assert(StaticTraits::max_warps_per_multiprocessor == dynamic_traits.max_warps_per_multiprocessor);
  static_assert(StaticTraits::max_threads_per_multiprocessor == dynamic_traits.max_threads_per_multiprocessor);
  static_assert(StaticTraits::reserved_shared_memory_per_block == dynamic_traits.reserved_shared_memory_per_block);
  static_assert(StaticTraits::max_shared_memory_per_block_optin == dynamic_traits.max_shared_memory_per_block_optin);
  static_assert(StaticTraits::cluster_supported == dynamic_traits.cluster_supported);
  static_assert(StaticTraits::redux_intrinisic == dynamic_traits.redux_intrinisic);
  static_assert(StaticTraits::elect_intrinsic == dynamic_traits.elect_intrinsic);
  static_assert(StaticTraits::cp_async_supported == dynamic_traits.cp_async_supported);
  static_assert(StaticTraits::tma_supported == dynamic_traits.tma_supported);

  constexpr cudax::arch_traits_t casted = StaticTraits{};
  static_assert(casted.compute_capability == dynamic_traits.compute_capability);
}

TEST_CASE("Traits", "[device]")
{
  compare_static_and_dynamic<700>();
  compare_static_and_dynamic<750>();
  compare_static_and_dynamic<800>();
  compare_static_and_dynamic<860>();
  compare_static_and_dynamic<890>();
  compare_static_and_dynamic<900>();

  // Compare arch traits with attributes
  for (const cudax::device& dev : cudax::devices)
  {
    auto traits = dev.arch_traits();

    CUDAX_REQUIRE(traits.max_threads_per_block == dev.attr(cudax::device::attrs::max_threads_per_block));
    CUDAX_REQUIRE(traits.max_block_dim_x == dev.attr(cudax::device::attrs::max_block_dim_x));
    CUDAX_REQUIRE(traits.max_block_dim_y == dev.attr(cudax::device::attrs::max_block_dim_y));
    CUDAX_REQUIRE(traits.max_block_dim_z == dev.attr(cudax::device::attrs::max_block_dim_z));
    CUDAX_REQUIRE(traits.max_grid_dim_x == dev.attr(cudax::device::attrs::max_grid_dim_x));
    CUDAX_REQUIRE(traits.max_grid_dim_y == dev.attr(cudax::device::attrs::max_grid_dim_y));
    CUDAX_REQUIRE(traits.max_grid_dim_z == dev.attr(cudax::device::attrs::max_grid_dim_z));

    CUDAX_REQUIRE(traits.warp_size == dev.attr(cudax::device::attrs::warp_size));
    CUDAX_REQUIRE(traits.total_constant_memory == dev.attr(cudax::device::attrs::total_constant_memory));
    CUDAX_REQUIRE(traits.max_shared_memory_per_block == dev.attr(cudax::device::attrs::max_shared_memory_per_block));
    CUDAX_REQUIRE(traits.gpu_overlap == dev.attr(cudax::device::attrs::gpu_overlap));
    CUDAX_REQUIRE(traits.can_map_host_memory == dev.attr(cudax::device::attrs::can_map_host_memory));
    CUDAX_REQUIRE(traits.concurrent_kernels == dev.attr(cudax::device::attrs::concurrent_kernels));
    CUDAX_REQUIRE(traits.stream_priorities_supported == dev.attr(cudax::device::attrs::stream_priorities_supported));
    CUDAX_REQUIRE(traits.global_l1_cache_supported == dev.attr(cudax::device::attrs::global_l1_cache_supported));
    CUDAX_REQUIRE(traits.local_l1_cache_supported == dev.attr(cudax::device::attrs::local_l1_cache_supported));
    CUDAX_REQUIRE(traits.max_registers_per_block == dev.attr(cudax::device::attrs::max_registers_per_block));
    CUDAX_REQUIRE(
      traits.max_registers_per_multiprocessor == dev.attr(cudax::device::attrs::max_registers_per_multiprocessor));
    CUDAX_REQUIRE(traits.compute_capability_major == dev.attr(cudax::device::attrs::compute_capability_major));
    CUDAX_REQUIRE(traits.compute_capability_minor == dev.attr(cudax::device::attrs::compute_capability_minor));
    CUDAX_REQUIRE(traits.compute_capability == dev.attr(cudax::device::attrs::compute_capability));
    CUDAX_REQUIRE(traits.max_shared_memory_per_multiprocessor
                  == dev.attr(cudax::device::attrs::max_shared_memory_per_multiprocessor));
    CUDAX_REQUIRE(
      traits.max_blocks_per_multiprocessor == dev.attr(cudax::device::attrs::max_blocks_per_multiprocessor));
    CUDAX_REQUIRE(
      traits.max_threads_per_multiprocessor == dev.attr(cudax::device::attrs::max_threads_per_multiprocessor));
    CUDAX_REQUIRE(
      traits.reserved_shared_memory_per_block == dev.attr(cudax::device::attrs::reserved_shared_memory_per_block));
    CUDAX_REQUIRE(
      traits.max_shared_memory_per_block_optin == dev.attr(cudax::device::attrs::max_shared_memory_per_block_optin));
  }
}
