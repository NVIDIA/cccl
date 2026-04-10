// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/warp/warp_reduce_batched.cuh>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cuda/__functional/maximum.h>
#include <cuda/std/array>
#include <cuda/std/span>

#include <c2h/catch2_test_helper.h>

__global__ __launch_bounds__(64) void WarpReduceBatchedOverviewKernel(int* out)
{
  // example-begin warp-reduce-batched-overview
  // implicitly 32 threads per logical warp, 3 batches
  using WarpReduceBatched = cub::WarpReduceBatched<int, 3>;

  // Assume 64 threads per block, so 64 / 32 = 2 logical warps
  // Each logical warp has its own TempStorage
  __shared__ typename WarpReduceBatched::TempStorage temp_storage[2];

  const int logical_warp_id = static_cast<int>(threadIdx.x) / 32;

  int thread_data[3];
  thread_data[0] = static_cast<int>(threadIdx.x);
  thread_data[1] = static_cast<int>(threadIdx.x) + 1;
  thread_data[2] = static_cast<int>(threadIdx.x) - 1;

  int results[1];
  WarpReduceBatched{temp_storage[logical_warp_id]}.Reduce(thread_data, results, cuda::maximum{});
  // results across threads: [31, 32, 30, ?, ?, ..., ?, 63, 64, 62, ?, ?, ..., ?]
  // example-end warp-reduce-batched-overview

  const int logical_lane_id = static_cast<int>(threadIdx.x) % 32;
  if (logical_lane_id < 3)
  {
    out[logical_warp_id * 3 + logical_lane_id] = results[0];
  }
}

C2H_TEST("WarpReduceBatched overview documentation kernel", "[warp][reduce][batched]")
{
  c2h::device_vector<int> d_out(6);

  WarpReduceBatchedOverviewKernel<<<1, 64>>>(thrust::raw_pointer_cast(d_out.data()));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<int> expected{31, 32, 30, 63, 64, 62};
  REQUIRE(expected == d_out);
}

__global__ void WarpReduceBatchedReduceApiKernel(int* out)
{
  // example-begin warp-reduce-batched-reduce
  // 16 threads per logical warp, 3 batches
  // Can't allow for physical warp synchronization since only the first logical warp participates due to the
  // conditional. The other threads (assuming there are more than 16 threads per block) can't exit early due to the
  // barrier.
  using WarpReduceBatched = cub::WarpReduceBatched<int, 3, 16>;

  // Only the first logical warp participates, so only a single TempStorage is needed
  __shared__ typename WarpReduceBatched::TempStorage temp_storage;

  cuda::std::array<int, 1> results{};
  if (threadIdx.x < 16)
  {
    cuda::std::array<int, 3> inputs{};
    inputs[0] = static_cast<int>(threadIdx.x);
    inputs[1] = static_cast<int>(threadIdx.x) + 1;
    inputs[2] = static_cast<int>(threadIdx.x) - 1;

    WarpReduceBatched{temp_storage}.Reduce(inputs, results, cuda::maximum{});
  }
  // results across threads: [15, 16, 14, ?, ?, ..., ?, 0, 0, ..., 0]
  __syncthreads();
  // Can reuse TempStorage after the barrier.
  // example-end warp-reduce-batched-reduce

  _CCCL_ASSERT(threadIdx.x < 16 || results[0] == 0, "");
  if (threadIdx.x < 3)
  {
    out[threadIdx.x] = results[0];
  }
}

C2H_TEST("WarpReduceBatched::Reduce documentation kernel", "[warp][reduce][batched]")
{
  c2h::device_vector<int> d_out(3);

  WarpReduceBatchedReduceApiKernel<<<1, 64>>>(thrust::raw_pointer_cast(d_out.data()));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<int> expected{15, 16, 14};
  REQUIRE(expected == d_out);
}

__global__ __launch_bounds__(8) void WarpReduceBatchedSumApiKernel(int* out)
{
  // example-begin warp-reduce-batched-sum
  // Implicitly 32 threads per logical warp, 3 batches
  // We can enable physical warp synchronization since all non-exited lanes do participate in the reduction
  using WarpReduceBatched = cub::WarpReduceBatched<int, 5, 2, true>;

  // Assume 64 threads per block, so 8 / 2 = 4 logical warps
  __shared__ typename WarpReduceBatched::TempStorage temp_storage[4];

  const int logical_warp_id = static_cast<int>(threadIdx.x) / 2;

  cuda::std::array<int, 5> thread_data{};
  thread_data[0] = static_cast<int>(threadIdx.x);
  thread_data[1] = static_cast<int>(threadIdx.x) + 1;
  thread_data[2] = static_cast<int>(threadIdx.x) - 1;
  thread_data[3] = static_cast<int>(threadIdx.x) + 2;
  thread_data[4] = static_cast<int>(threadIdx.x) - 2;
  // Use a static size spans to pick the first 3 elements as inputs and the last 2 elements as outputs (aliasing inputs)
  cuda::std::span<int, 5> inputs{cuda::std::begin(thread_data), 5};
  cuda::std::span<int, 3> results{cuda::std::begin(thread_data) + 2, 3};
  WarpReduceBatched{temp_storage[logical_warp_id]}.Sum(inputs, results);
  // results across threads (striped!): [[1, -1, -3], [3, 5, ?], [5, 3, 1], [7, 9, ?], [9, 7, 5], [11, 13, ?], [13, 11,
  // 9], [15, 17, ?]] example-end warp-reduce-batched-sum

  const int logical_lane_id                      = static_cast<int>(threadIdx.x) % 2;
  out[logical_warp_id * 5 + logical_lane_id]     = results[0];
  out[logical_warp_id * 5 + 2 + logical_lane_id] = results[1];
  if (logical_lane_id == 0)
  {
    out[logical_warp_id * 5 + 4] = results[2];
  }
}

C2H_TEST("WarpReduceBatched::Sum documentation kernel", "[warp][reduce][batched]")
{
  c2h::device_vector<int> d_out(20);

  WarpReduceBatchedSumApiKernel<<<1, 8>>>(thrust::raw_pointer_cast(d_out.data()));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<int> expected{1, 3, -1, 5, -3, 5, 7, 3, 9, 1, 9, 11, 7, 13, 5, 13, 15, 11, 17, 9};
  REQUIRE(expected == d_out);
}
