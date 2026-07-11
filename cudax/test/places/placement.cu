//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 *
 * @brief Test placement evaluation and geometry-aware (shaped) composite
 *        allocation: evaluate_localized_placement(), the
 *        allocate_nd(data_dims, elemsize) data_place interface, and
 *        cute_partition-backed composite places.
 *
 * Runs on a single GPU (all places on device 0); with two or more GPUs it
 * additionally asserts physical residency of the allocated blocks and the
 * peer-access path.
 */

#include <cuda/experimental/__places/cute_partition.cuh>
#include <cuda/experimental/__places/partitions/blocked_partition.cuh>
#include <cuda/experimental/__places/places.cuh>

#include <cstdio>

using namespace cuda::experimental::places;

namespace
{
__global__ void init_kernel(int* ptr, size_t n, int value)
{
  size_t tid = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (tid < n)
  {
    ptr[tid] = value + static_cast<int>(tid % 1024);
  }
}

__global__ void check_kernel(const int* ptr, size_t n, int value, int* result)
{
  size_t tid = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (tid < n)
  {
    if (ptr[tid] != value + static_cast<int>(tid % 1024))
    {
      atomicExch(result, 1); // Set error flag
    }
  }
}

// Check if VMM is supported on the current device
bool vmm_supported(int dev_id = 0)
{
  CUdevice dev;
  cuda_try(cuDeviceGet(&dev, dev_id));
  int supportsVMM;
  cuda_try(cuDeviceGetAttribute(&supportsVMM, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, dev));
  return supportsVMM == 1;
}

exec_place make_device_grid(int ndevs, size_t nplaces)
{
  ::std::vector<exec_place> places;
  for (size_t i = 0; i < nplaces; i++)
  {
    places.push_back(exec_place::device(static_cast<int>(i % static_cast<size_t>(ndevs))));
  }
  return make_grid(mv(places));
}

// Write to the buffer through a kernel and verify the content
void write_and_check(int* ptr, size_t n, int value)
{
  const int nthreads = 256;
  const int nblocks  = static_cast<int>((n + nthreads - 1) / nthreads);

  init_kernel<<<nblocks, nthreads>>>(ptr, n, value);
  cuda_try(cudaGetLastError());

  int* d_result;
  cuda_try(cudaMalloc(&d_result, sizeof(int)));
  cuda_try(cudaMemset(d_result, 0, sizeof(int)));
  check_kernel<<<nblocks, nthreads>>>(ptr, n, value, d_result);
  cuda_try(cudaGetLastError());

  int h_result = -1;
  cuda_try(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));
  cuda_try(cudaFree(d_result));
  EXPECT(h_result == 0, "kernel readback mismatch");
}

void test_evaluate_blocked_even()
{
  printf("Testing evaluate_localized_placement (even blocked split)...\n");

  const size_t block_size = 2 * 1024 * 1024;
  const size_t n          = 2 * block_size; // bytes, elemsize 1: exactly 2 blocks
  const dim4 data_dims(n);

  auto grid = make_device_grid(1, 2);

  auto stats =
    evaluate_localized_placement(grid, &blocked_partition_custom<0>::get_executor, data_dims, 1, 10, block_size);

  EXPECT(stats.total_bytes == n);
  EXPECT(stats.vm_bytes == n);
  EXPECT(stats.block_size == block_size);
  EXPECT(stats.nblocks == 2);
  // One block per place, and the split is block-aligned: every probe agrees
  EXPECT(stats.nallocs == 2);
  EXPECT(stats.accuracy() == 1.0);

  size_t total = 0;
  for (const auto& entry : stats.bytes_per_place)
  {
    total += entry.second;
  }
  EXPECT(total == n);

  printf("  evaluate (even blocked) test PASSED\n");
}

void test_evaluate_straddling_block()
{
  printf("Testing evaluate_localized_placement (majority tie-breaking)...\n");

  // 4,000,000 one-byte elements blocked over 2 places: each place owns
  // 2,000,000 bytes, just short of the 2 MiB block. Block 0 straddles the two
  // owners (~95%/5%): the majority vote must keep it on place 0, and the
  // accuracy must reflect the straddling.
  const size_t block_size = 2 * 1024 * 1024;
  const size_t n          = 4'000'000;
  const dim4 data_dims(n);

  auto grid = make_device_grid(1, 2);

  auto stats =
    evaluate_localized_placement(grid, &blocked_partition_custom<0>::get_executor, data_dims, 1, 64, block_size);

  EXPECT(stats.nblocks == 2);
  EXPECT(stats.nallocs == 2); // majority breaks the tie: block 0 and block 1 differ
  EXPECT(stats.accuracy() < 1.0);
  EXPECT(stats.accuracy() > 0.9);

  // The decision procedure is seeded: evaluating twice gives the same stats
  auto stats2 =
    evaluate_localized_placement(grid, &blocked_partition_custom<0>::get_executor, data_dims, 1, 64, block_size);
  EXPECT(stats.matching_samples == stats2.matching_samples);
  EXPECT(stats.total_samples == stats2.total_samples);
  EXPECT(stats.bytes_per_place == stats2.bytes_per_place);

  printf("  evaluate (majority tie-breaking) test PASSED\n");
}

void test_evaluate_cute_matches_mapper()
{
  printf("Testing evaluate_localized_placement (cute partition vs mapper)...\n");

  const size_t block_size = 2 * 1024 * 1024;
  const size_t n          = 4 * block_size;
  const dim4 data_dims(n);

  auto grid = make_device_grid(1, 2);
  auto part = make_partition(data_dims, {dim_spec{dim_policy::blocked, 0, 0}}, grid.get_dims());

  auto stats_mapper =
    evaluate_localized_placement(grid, &blocked_partition_custom<0>::get_executor, data_dims, 1, 10, block_size);
  auto stats_cute = evaluate_localized_placement(grid, part, 1, 10, block_size);

  EXPECT(stats_mapper.nblocks == stats_cute.nblocks);
  EXPECT(stats_mapper.nallocs == stats_cute.nallocs);
  EXPECT(stats_mapper.bytes_per_place == stats_cute.bytes_per_place);
  EXPECT(stats_mapper.matching_samples == stats_cute.matching_samples);

  printf("  evaluate (cute vs mapper) test PASSED\n");
}

void test_shaped_alloc_callback_composite(int ndevs)
{
  printf("Testing shaped allocation on a partition_fn_t composite place...\n");

  const size_t n = 1024 * 1024; // ints
  const dim4 data_dims(n);

  auto grid     = make_device_grid(ndevs, 2);
  data_place dp = data_place::composite(blocked_partition_custom<0>{}, grid);

  // The byte-count allocate cannot know the tensor geometry: it must throw
  bool thrown = false;
  try
  {
    dp.allocate(static_cast<::std::ptrdiff_t>(n * sizeof(int)));
  }
  catch (const ::std::runtime_error&)
  {
    thrown = true;
  }
  EXPECT(thrown, "byte-count allocate on a composite place must throw");

  void* ptr = dp.allocate_nd(data_dims, sizeof(int));
  EXPECT(ptr != nullptr);

  write_and_check(static_cast<int*>(ptr), n, 17);

  dp.deallocate(ptr, n * sizeof(int));

  printf("  shaped allocation (callback composite) test PASSED\n");
}

void test_shaped_alloc_cute_composite(int ndevs)
{
  printf("Testing shaped allocation on a cute_partition composite place...\n");

  const size_t n = 1024 * 1024; // ints
  const dim4 data_dims(n);

  auto grid = make_device_grid(ndevs, 2);
  auto part = make_partition(data_dims, {dim_spec{dim_policy::blocked, 0, 0}}, grid.get_dims());

  data_place dp = make_composite_data_place(grid, part);

  // The partition is specific to one tensor: other extents must be rejected
  bool thrown = false;
  try
  {
    dp.allocate_nd(dim4(n / 2), sizeof(int));
  }
  catch (const ::std::invalid_argument&)
  {
    thrown = true;
  }
  EXPECT(thrown, "extent mismatch with the partition must throw");

  void* ptr = dp.allocate_nd(data_dims, sizeof(int));
  EXPECT(ptr != nullptr);

  write_and_check(static_cast<int*>(ptr), n, 41);

  dp.deallocate(ptr, n * sizeof(int));

  printf("  shaped allocation (cute composite) test PASSED\n");
}

#if !defined(DOXYGEN_SHOULD_SKIP_THIS)
void test_multi_gpu_residency(int ndevs)
{
  if (ndevs < 2)
  {
    printf("Skipping multi-GPU residency test (requires 2+ devices)\n");
    return;
  }
  if (!vmm_supported(1))
  {
    printf("Skipping multi-GPU residency test (device 1 lacks VMM support)\n");
    return;
  }

  printf("Testing multi-GPU residency of a blocked shaped allocation...\n");

  // Query the allocation granularity so each place owns a whole number of blocks
  CUmemAllocationProp prop = {};
  prop.type                = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type       = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id         = 0;
  size_t granularity       = cuda_try<cuMemGetAllocationGranularity>(&prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);

  const size_t n = 2 * granularity / sizeof(int); // one block per place
  const dim4 data_dims(n);

  ::std::vector<exec_place> places;
  places.push_back(exec_place::device(0));
  places.push_back(exec_place::device(1));
  auto grid = make_grid(mv(places));

  data_place dp = data_place::composite(blocked_partition_custom<0>{}, grid);
  void* ptr     = dp.allocate_nd(data_dims, sizeof(int));
  EXPECT(ptr != nullptr);

  // Each half of the range must be physically backed by its owner
  for (int half = 0; half < 2; half++)
  {
    int ordinal           = -1;
    CUdeviceptr probe_ptr = reinterpret_cast<CUdeviceptr>(ptr) + static_cast<size_t>(half) * granularity;
    cuda_try(cuPointerGetAttribute(&ordinal, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, probe_ptr));
    EXPECT(ordinal == half, "block is not resident on the place that owns it");
  }

  // Peer path: touch the whole range (including device-0-owned blocks) from
  // device 1, which exercises the cuMemSetAccess mappings
  int peer_01 = cuda_try<cudaDeviceCanAccessPeer>(0, 1);
  int peer_10 = cuda_try<cudaDeviceCanAccessPeer>(1, 0);
  if (peer_01 && peer_10)
  {
    cuda_try(cudaSetDevice(1));
    write_and_check(static_cast<int*>(ptr), n, 73);
    cuda_try(cudaSetDevice(0));
  }
  else
  {
    printf("  (peer access unavailable between devices 0 and 1: cross-device touch skipped)\n");
  }

  dp.deallocate(ptr, n * sizeof(int));

  printf("  multi-GPU residency test PASSED\n");
}
#endif // !DOXYGEN_SHOULD_SKIP_THIS
} // namespace

int main()
{
  int ndevs = 0;
  if (cudaGetDeviceCount(&ndevs) != cudaSuccess || ndevs == 0)
  {
    printf("Skipping placement tests: no CUDA device\n");
    return 0;
  }

  cuda_try(cudaSetDevice(0));
  cuda_try(cudaFree(nullptr));

  if (!vmm_supported())
  {
    printf("Skipping placement tests: VMM is not supported on this machine\n");
    return 0;
  }

  printf("=== Testing placement evaluation and shaped allocation ===\n\n");

  test_evaluate_blocked_even();
  test_evaluate_straddling_block();
  test_evaluate_cute_matches_mapper();
  test_shaped_alloc_callback_composite(ndevs);
  test_shaped_alloc_cute_composite(ndevs);
  test_multi_gpu_residency(ndevs);

  printf("\n=== All placement tests PASSED ===\n");
  return 0;
}
