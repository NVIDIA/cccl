// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/warp/warp_reduce_batched_broadcast.cuh>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cuda/std/array>

#include <c2h/catch2_test_helper.h>

__global__ __launch_bounds__(8) void WarpReduceBatchedBroadcastOverviewKernel(int* out)
{
  // example-begin warp-reduce-batched-broadcast-overview
  constexpr int num_batches          = 3;
  constexpr int logical_warp_threads = 4;
  using WarpReduceBatchedBroadcast   = cub::WarpReduceBatchedBroadcast<int, num_batches, logical_warp_threads>;

  typename WarpReduceBatchedBroadcast::TempStorage temp_storage;

  const int tid = static_cast<int>(threadIdx.x);

  const cuda::std::array<int, num_batches> inputs{tid - 1, tid, tid + 1};
  cuda::std::array<int, num_batches> results = WarpReduceBatchedBroadcast{temp_storage}.Sum(inputs);
  // results across threads:
  // [{2, 6, 10}, {2, 6, 10}, {2, 6, 10}, {2, 6, 10},
  //  {18, 22, 26}, {18, 22, 26}, {18, 22, 26}, {18, 22, 26}]
  // example-end warp-reduce-batched-broadcast-overview

  _CCCL_PRAGMA_UNROLL_FULL()
  for (int batch = 0; batch < num_batches; ++batch)
  {
    out[tid * num_batches + batch] = results[batch];
  }
}

C2H_TEST("WarpReduceBatchedBroadcast overview documentation kernel", "[warp][reduce][batched]")
{
  constexpr int num_threads = 8;
  constexpr int num_batches = 3;
  c2h::device_vector<int> d_out(num_threads * num_batches);

  WarpReduceBatchedBroadcastOverviewKernel<<<1, num_threads>>>(thrust::raw_pointer_cast(d_out.data()));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<int> expected(num_threads * num_batches);
  for (int tid = 0; tid < num_threads; ++tid)
  {
    const int logical_warp_id       = tid / 4;
    expected[tid * num_batches]     = 2 + logical_warp_id * 16;
    expected[tid * num_batches + 1] = 6 + logical_warp_id * 16;
    expected[tid * num_batches + 2] = 10 + logical_warp_id * 16;
  }
  REQUIRE(expected == d_out);
}

__global__ __launch_bounds__(8) void WarpReduceBatchedBroadcastCArraysKernel(int* out)
{
  constexpr int num_batches          = 3;
  constexpr int logical_warp_threads = 4;
  using WarpReduceBatchedBroadcast   = cub::WarpReduceBatchedBroadcast<int, num_batches, logical_warp_threads>;

  typename WarpReduceBatchedBroadcast::TempStorage temp_storage;

  const int tid = static_cast<int>(threadIdx.x);

  const int inputs[num_batches] = {tid - 1, tid, tid + 1};
  int results[num_batches];
  WarpReduceBatchedBroadcast{temp_storage}.Sum(inputs, results);

  _CCCL_PRAGMA_UNROLL_FULL()
  for (int batch = 0; batch < num_batches; ++batch)
  {
    out[tid * num_batches + batch] = results[batch];
  }
}

C2H_TEST("WarpReduceBatchedBroadcast accepts C arrays", "[warp][reduce][batched]")
{
  constexpr int num_threads = 8;
  constexpr int num_batches = 3;
  c2h::device_vector<int> d_out(num_threads * num_batches);

  WarpReduceBatchedBroadcastCArraysKernel<<<1, num_threads>>>(thrust::raw_pointer_cast(d_out.data()));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<int> expected(num_threads * num_batches);
  for (int tid = 0; tid < num_threads; ++tid)
  {
    const int logical_warp_id       = tid / 4;
    expected[tid * num_batches]     = 2 + logical_warp_id * 16;
    expected[tid * num_batches + 1] = 6 + logical_warp_id * 16;
    expected[tid * num_batches + 2] = 10 + logical_warp_id * 16;
  }
  REQUIRE(expected == d_out);
}

__global__ __launch_bounds__(4) void WarpReduceBatchedBroadcastZeroBatchesKernel(int* out)
{
  constexpr int num_batches          = 0;
  constexpr int logical_warp_threads = 4;
  using WarpReduceBatchedBroadcast   = cub::WarpReduceBatchedBroadcast<int, num_batches, logical_warp_threads>;

  typename WarpReduceBatchedBroadcast::TempStorage temp_storage;

  cuda::std::array<int, num_batches> inputs{};
  cuda::std::array<int, num_batches> results = WarpReduceBatchedBroadcast{temp_storage}.Sum(inputs);
  WarpReduceBatchedBroadcast{temp_storage}.Sum(inputs, results);

  if (threadIdx.x == 0)
  {
    out[0] = static_cast<int>(results.size());
  }
}

C2H_TEST("WarpReduceBatchedBroadcast accepts zero batches", "[warp][reduce][batched]")
{
  c2h::device_vector<int> d_out(1, -1);

  WarpReduceBatchedBroadcastZeroBatchesKernel<<<1, 4>>>(thrust::raw_pointer_cast(d_out.data()));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  REQUIRE(d_out[0] == 0);
}
