//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// This example is based on the simpleCooperativeGroups from CUDA/cuda-samples repository, rewritten to use the CCCL
// Cooperative Groups.

// This sample is a simple code that illustrates basic usage of CCCL cooperative groups within the thread block.
// The code launches a single thread block, creates a cooperative group of all threads in the block, and a set of tiled
// partition cooperative groups. For each, it uses a generic reduction function to calculate the sum of all the ranks in
// that group. In each case the result is printed, together with the expected answer (which is calculated using
// the analytical formula (n - 1) * n / 2, noting that the ranks start at zero).

#include <cuda/devices>
#include <cuda/hierarchy>
#include <cuda/launch>
#include <cuda/std/cstdint>
#include <cuda/stream>

#include <cuda/experimental/coop/group>

namespace cudax = cuda::experimental;

// CUDA device function
//
// calculates the sum of val across the group g. The workspace array, x,
// must be large enough to contain g.size() integers.
template <class Group, class T>
__device__ T sum_reduction(const Group& g, T* scratch, T val)
{
  // Rank of this thread in the group.
  const auto rank = cuda::gpu_thread.rank(g);

  // For each iteration of this loop, the number of threads active in the reduction, i, is halved, and each active
  // thread (with index [rank]) performs a single summation of it's own value with that of a "partner" (with index
  // [rank+i]).
  for (auto i = cuda::gpu_thread.count(g) / 2; i > 0; i /= 2)
  {
    // Store value for this thread in temporary array.
    scratch[rank] = val;

    // Synchronize all threads in group.
    g.sync_aligned();

    if (rank < i)
    {
      // Active threads perform summation of their value with their partner's value.
      val += scratch[rank + i];
    }

    // Synchronize all threads in group.
    g.sync_aligned();
  }

  // Root thread in group returns result, and others return T(-1).
  return (cuda::gpu_thread.is_root_rank(g)) ? val : static_cast<T>(-1);
}

// Kernel functor that demonstrates how the CCCL Cooperative Groups can be used.
struct GroupsKernel
{
  template <class Config>
  __device__ void operator()(Config config) const
  {
    // block_group includes all threads in the block.
    const cudax::coop::this_block block_group{config};
    const auto block_group_size = cuda::gpu_thread.count(block_group);

    // Get the scratch in shared memory required for reduction.
    const auto scratch = cuda::dynamic_shared_memory(config);

    cuda::std::uint32_t input, output, expected_output;

    // Input to the reduction, for each thread, is its rank in the group.
    input = cuda::gpu_thread.rank(block_group);

    // Expected output from analytical formula (n - 1) * n / 2 (noting that indexing starts at 0 rather than 1)
    expected_output = (block_group_size - 1) * block_group_size / 2;

    // Perform reduction.
    output = sum_reduction(block_group, scratch.data(), input);

    // The root thread in group prints out result.
    if (cuda::gpu_thread.is_root_rank(block_group))
    {
      printf(" Sum of all ranks 0..%d in block_group is %d (expected %d)\n\n",
             block_group_size - 1,
             output,
             expected_output);

      printf(" Now creating %d groups, each of size 16 threads:\n\n", block_group_size / 16);
    }

    // Wait for the root thread.
    block_group.sync_aligned();

    // Create a half_warp group that splits every warp into 2 groups.
    const cudax::coop::generic_group half_warp{
      cuda::gpu_thread, cudax::coop::this_warp{config}, cudax::coop::group_by<16>{}, cudax::coop::lane_synchronizer{}};

    // This offset allows each group to have its own unique area in the scratch
    // array
    const auto scratch_offset = cuda::gpu_thread.rank(block_group) - cuda::gpu_thread.rank(half_warp);

    // input to reduction, for each thread, is its' rank in the group
    input = cuda::gpu_thread.rank(half_warp);

    // expected output from analytical formula (n-1)(n)/2
    // (noting that indexing starts at 0 rather than 1)
    expected_output = 15 * 16 / 2;

    // Perform reduction.
    output = sum_reduction(half_warp, scratch.data() + scratch_offset, input);

    // Each root thread prints out the result.
    if (cuda::gpu_thread.is_root_rank(half_warp))
    {
      printf("   Sum of all ranks 0..15 in this half_warp group is %d (expected %d)\n", output, expected_output);
    }
  }
};

int main()
try
{
  // Check that there is a device we can use.
  if (cuda::devices.size() < 1)
  {
    fprintf(stderr, "No CUDA device found.");
    return 1;
  }

  // Select the device.
  const auto device = cuda::devices[0];

  // Create a stream.
  cuda::stream stream{device};

  // Create 1D kernel configuration with 64 threads and the necessary shared memory allocated.
  const auto threadsPerBlock = 64;
  const auto config          = cuda::make_config(
    cuda::grid_dims<1>(),
    cuda::block_dims(dim3{threadsPerBlock}),
    cuda::dynamic_shared_memory<cuda::std::uint32_t[]>(threadsPerBlock));

  // Launch the kernel.
  printf("\nLaunching a single block with %d threads...\n\n", threadsPerBlock);
  cuda::launch(stream, config, GroupsKernel{});

  // Wait for the kernel to finish.
  stream.sync();
  printf("\n...Done.\n\n");
}
catch (const std::exception& e)
{
  fprintf(stderr, "caught an exception: \"%s\"\n", e.what());
  return 1;
}
catch (...)
{
  fprintf(stderr, "caught an unknown exception\n");
  return 1;
}
