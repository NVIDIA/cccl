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
  using WarpReduceBatched = cub::WarpReduceBatched<int, 3>;

  // Assume 64 threads per block, so 64 / 32 = 2 logical warps
  // Each logical warp has its own TempStorage
  __shared__ typename WarpReduceBatched::TempStorage temp_storage[2];

  const int logical_warp_id = static_cast<int>(threadIdx.x) / 32;

  int thread_data[3];
  thread_data[0] = static_cast<int>(threadIdx.x) - 1;
  thread_data[1] = static_cast<int>(threadIdx.x);
  thread_data[2] = static_cast<int>(threadIdx.x) + 1;

  int result = WarpReduceBatched{temp_storage[logical_warp_id]}.Reduce(thread_data, cuda::maximum{});
  // results across threads: [30, 31, 32, ?, ?, ..., ?, 62, 63, 64, ?, ?, ..., ?]
  // example-end warp-reduce-batched-overview

  const int logical_lane_id = static_cast<int>(threadIdx.x) % 32;
  if (logical_lane_id < 3)
  {
    out[logical_warp_id * 3 + logical_lane_id] = result;
  }
}

C2H_TEST("WarpReduceBatched overview documentation kernel", "[warp][reduce][batched]")
{
  c2h::device_vector<int> d_out(6);

  WarpReduceBatchedOverviewKernel<<<1, 64>>>(thrust::raw_pointer_cast(d_out.data()));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<int> expected{30, 31, 32, 62, 63, 64};
  REQUIRE(expected == d_out);
}

__global__ void WarpReduceBatchedReduceApiKernel(int* out)
{
  // example-begin warp-reduce-batched-reduce
  // Can't allow for physical warp synchronization since only the first logical warp participates due to the
  // conditional. The other threads (assuming there are more than 16 threads per block) can't exit early due to the
  // barrier.
  using WarpReduceBatched = cub::WarpReduceBatched<int, 3, 16>;

  // Only the first logical warp participates, so only a single TempStorage is needed
  __shared__ typename WarpReduceBatched::TempStorage temp_storage;

  int result{};
  if (threadIdx.x < 16)
  {
    cuda::std::array<int, 3> inputs{};
    inputs[0] = static_cast<int>(threadIdx.x) - 1;
    inputs[1] = static_cast<int>(threadIdx.x);
    inputs[2] = static_cast<int>(threadIdx.x) + 1;

    result = WarpReduceBatched{temp_storage}.Reduce(inputs, cuda::maximum{});
  }
  // results across threads: [14, 15, 16, ?, ?, ..., ?, 0, 0, ..., 0]
  __syncthreads();
  // Can reuse TempStorage after the barrier.
  // example-end warp-reduce-batched-reduce
  _CCCL_ASSERT(threadIdx.x < 16 || result == 0, "");
  if (threadIdx.x < 3)
  {
    out[threadIdx.x] = result;
  }
}

C2H_TEST("WarpReduceBatched::Reduce documentation kernel", "[warp][reduce][batched]")
{
  c2h::device_vector<int> d_out(3);

  WarpReduceBatchedReduceApiKernel<<<1, 64>>>(thrust::raw_pointer_cast(d_out.data()));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<int> expected{14, 15, 16};
  REQUIRE(expected == d_out);
}

__global__ __launch_bounds__(8) void WarpReduceBatchedReduceToStripedApiKernel(int* out)
{
  // example-begin warp-reduce-batched-reduce-to-striped
  // Can't allow for physical warp synchronization since only every other logical warp participates.
  // The other threads (assuming there are more than 2 threads per block) can't exit early due to the
  // barrier.
  using WarpReduceBatched = cub::WarpReduceBatched<int, 3, 2>;

  // Assume 8 threads per block, so 8 / 2 = 4 logical warps
  // Only every other logical warp participates, so only 2 TempStorage are needed
  __shared__ typename WarpReduceBatched::TempStorage temp_storage[2];

  const int logical_warp_id   = static_cast<int>(threadIdx.x) / 2;
  const bool is_participating = logical_warp_id % 2 == 0;
  const int participant_idx   = logical_warp_id / 2;

  cuda::std::array<int, 2> results{};
  if (is_participating)
  {
    cuda::std::array<int, 3> inputs{};
    inputs[0] = static_cast<int>(threadIdx.x) - 1;
    inputs[1] = static_cast<int>(threadIdx.x);
    inputs[2] = static_cast<int>(threadIdx.x) + 1;

    WarpReduceBatched{temp_storage[participant_idx]}.ReduceToStriped(inputs, results, cuda::maximum{});
  }
  // results across threads:
  // [[0, 2], [1, ?], [0, 0], [0, 0], [4, 6], [5, ?], [0, 0], [0, 0]]
  __syncthreads();
  // Can reuse TempStorage after the barrier.
  // example-end warp-reduce-batched-reduce-to-striped

  _CCCL_ASSERT(is_participating || (results[0] == 0 && results[1] == 0), "");
  if (is_participating)
  {
    int const logical_lane_id                  = static_cast<int>(threadIdx.x) % 2;
    out[participant_idx * 3 + logical_lane_id] = results[0];
    if (logical_lane_id == 0)
    {
      out[participant_idx * 3 + 2] = results[1];
    }
  }
}

C2H_TEST("WarpReduceBatched::ReduceToStriped documentation kernel", "[warp][reduce][batched]")
{
  c2h::device_vector<int> d_out(6);

  WarpReduceBatchedReduceToStripedApiKernel<<<1, 8>>>(thrust::raw_pointer_cast(d_out.data()));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<int> expected{0, 1, 2, 4, 5, 6};
  REQUIRE(expected == d_out);
}

__global__ __launch_bounds__(8) void WarpReduceBatchedReduceToBlockedApiKernel(int* out)
{
  // example-begin warp-reduce-batched-reduce-to-blocked
  // Can't allow for physical warp synchronization since only every other logical warp participates.
  // The other threads (assuming there are more than 2 threads per block) can't exit early due to the
  // barrier.
  using WarpReduceBatched = cub::WarpReduceBatched<int, 3, 2>;

  // Assume 8 threads per block, so 8 / 2 = 4 logical warps
  // Only every other logical warp participates, so only 2 TempStorage are needed
  __shared__ typename WarpReduceBatched::TempStorage temp_storage[2];

  const int logical_warp_id   = static_cast<int>(threadIdx.x) / 2;
  const bool is_participating = logical_warp_id % 2 == 0;
  const int participant_idx   = logical_warp_id / 2;

  cuda::std::array<int, 2> results{};
  if (is_participating)
  {
    cuda::std::array<int, 3> inputs{};
    inputs[0] = static_cast<int>(threadIdx.x) - 1;
    inputs[1] = static_cast<int>(threadIdx.x);
    inputs[2] = static_cast<int>(threadIdx.x) + 1;

    WarpReduceBatched{temp_storage[participant_idx]}.ReduceToBlocked(inputs, results, cuda::maximum{});
  }
  // results across threads:
  // [[0, 1], [2, ?], [0, 0], [0, 0], [4, 5], [6, ?], [0, 0], [0, 0]]
  __syncthreads();
  // Can reuse TempStorage after the barrier.
  // example-end warp-reduce-batched-reduce-to-blocked

  _CCCL_ASSERT(is_participating || (results[0] == 0 && results[1] == 0), "");
  if (is_participating)
  {
    int const logical_lane_id                      = static_cast<int>(threadIdx.x) % 2;
    out[participant_idx * 3 + logical_lane_id * 2] = results[0];
    if (logical_lane_id == 0)
    {
      out[participant_idx * 3 + 1] = results[1];
    }
  }
}

C2H_TEST("WarpReduceBatched::ReduceToBlocked documentation kernel", "[warp][reduce][batched]")
{
  c2h::device_vector<int> d_out(6);

  WarpReduceBatchedReduceToBlockedApiKernel<<<1, 8>>>(thrust::raw_pointer_cast(d_out.data()));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<int> expected{0, 1, 2, 4, 5, 6};
  REQUIRE(expected == d_out);
}

__global__ __launch_bounds__(8) void WarpReduceBatchedSumApiKernel(int* out)
{
  // example-begin warp-reduce-batched-sum
  // We can enable physical warp synchronization since all non-exited lanes do participate in the primitive.
  using WarpReduceBatched = cub::WarpReduceBatched<int, 3, 4, true>;

  // Assume 8 threads per block, so 8 / 4 = 2 logical warps
  __shared__ typename WarpReduceBatched::TempStorage temp_storage[2];

  const int logical_warp_id = static_cast<int>(threadIdx.x) / 4;

  cuda::std::array<int, 3> inputs{};
  inputs[0]  = static_cast<int>(threadIdx.x) - 1;
  inputs[1]  = static_cast<int>(threadIdx.x);
  inputs[2]  = static_cast<int>(threadIdx.x) + 1;
  int result = WarpReduceBatched{temp_storage[logical_warp_id]}.Sum(inputs);
  // results across threads:
  // [2, 6, 10, ?, 18, 22, 26, ?]
  // example-end warp-reduce-batched-sum
  const int logical_lane_id = static_cast<int>(threadIdx.x) % 4;
  if (logical_lane_id < 3)
  {
    out[logical_warp_id * 3 + logical_lane_id] = result;
  }
}

C2H_TEST("WarpReduceBatched::Sum documentation kernel", "[warp][reduce][batched]")
{
  c2h::device_vector<int> d_out(6);

  WarpReduceBatchedSumApiKernel<<<1, 8>>>(thrust::raw_pointer_cast(d_out.data()));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<int> expected{2, 6, 10, 18, 22, 26};
  REQUIRE(expected == d_out);
}

__global__ __launch_bounds__(8) void WarpReduceBatchedSumToStripedApiKernel(int* out)
{
  // example-begin warp-reduce-batched-sum-to-striped
  // We can enable physical warp synchronization since all non-exited lanes do participate in the primitive.
  using WarpReduceBatched = cub::WarpReduceBatched<int, 5, 2, true>;

  // Assume 8 threads per block, so 8 / 2 = 4 logical warps
  __shared__ typename WarpReduceBatched::TempStorage temp_storage[4];

  const int logical_warp_id = static_cast<int>(threadIdx.x) / 2;

  cuda::std::array<int, 5> thread_data{};
  thread_data[0] = static_cast<int>(threadIdx.x) - 2;
  thread_data[1] = static_cast<int>(threadIdx.x) - 1;
  thread_data[2] = static_cast<int>(threadIdx.x);
  thread_data[3] = static_cast<int>(threadIdx.x) + 1;
  thread_data[4] = static_cast<int>(threadIdx.x) + 2;
  // Use a static size spans to pick the first 3 elements as inputs and the last 2 elements as outputs (aliasing inputs)
  cuda::std::span<int, 5> inputs{cuda::std::begin(thread_data), 5};
  cuda::std::span<int, 3> results{cuda::std::begin(thread_data) + 2, 3};
  WarpReduceBatched{temp_storage[logical_warp_id]}.SumToStriped(inputs, results);
  // results across threads:
  // [[-3, 1, 5], [-1, 3, ?], [1, 5, 9], [3, 7, ?], ..., [9, 13, 17], [11, 15, ?]]
  // example-end warp-reduce-batched-sum-to-striped

  const int logical_lane_id                      = static_cast<int>(threadIdx.x) % 2;
  out[logical_warp_id * 5 + logical_lane_id]     = results[0];
  out[logical_warp_id * 5 + 2 + logical_lane_id] = results[1];
  if (logical_lane_id == 0)
  {
    out[logical_warp_id * 5 + 4] = results[2];
  }
}

C2H_TEST("WarpReduceBatched::SumToStriped documentation kernel", "[warp][reduce][batched]")
{
  c2h::device_vector<int> d_out(20);

  WarpReduceBatchedSumToStripedApiKernel<<<1, 8>>>(thrust::raw_pointer_cast(d_out.data()));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<int> expected{-3, -1, 1, 3, 5, 1, 3, 5, 7, 9, 5, 7, 9, 11, 13, 9, 11, 13, 15, 17};
  REQUIRE(expected == d_out);
}

__global__ __launch_bounds__(8) void WarpReduceBatchedSumToBlockedApiKernel(int* out)
{
  // example-begin warp-reduce-batched-sum-to-blocked
  // We can enable physical warp synchronization since all non-exited lanes do participate in the primitive.
  using WarpReduceBatched = cub::WarpReduceBatched<int, 5, 2, true>;

  // Assume 8 threads per block, so 8 / 2 = 4 logical warps
  __shared__ typename WarpReduceBatched::TempStorage temp_storage[4];

  const int logical_warp_id = static_cast<int>(threadIdx.x) / 2;

  cuda::std::array<int, 5> thread_data{};
  thread_data[0] = static_cast<int>(threadIdx.x) - 2;
  thread_data[1] = static_cast<int>(threadIdx.x) - 1;
  thread_data[2] = static_cast<int>(threadIdx.x);
  thread_data[3] = static_cast<int>(threadIdx.x) + 1;
  thread_data[4] = static_cast<int>(threadIdx.x) + 2;
  // Use a static size spans to pick the first 3 elements as inputs and the last 2 elements as outputs (aliasing inputs)
  cuda::std::span<int, 5> inputs{cuda::std::begin(thread_data), 5};
  cuda::std::span<int, 3> results{cuda::std::begin(thread_data) + 2, 3};
  WarpReduceBatched{temp_storage[logical_warp_id]}.SumToBlocked(inputs, results);
  // results across threads:
  // [[-3, -1, 1], [3, 5, ?], [1, 3, 5], [7, 9, ?], ..., [9, 11, 13], [15, 17, ?]]
  // example-end warp-reduce-batched-sum-to-blocked

  const int logical_lane_id                          = static_cast<int>(threadIdx.x) % 2;
  out[logical_warp_id * 5 + logical_lane_id * 3]     = results[0];
  out[logical_warp_id * 5 + logical_lane_id * 3 + 1] = results[1];
  if (logical_lane_id == 0)
  {
    out[logical_warp_id * 5 + 2] = results[2];
  }
}

C2H_TEST("WarpReduceBatched::SumToBlocked documentation kernel", "[warp][reduce][batched]")
{
  c2h::device_vector<int> d_out(20);

  WarpReduceBatchedSumToBlockedApiKernel<<<1, 8>>>(thrust::raw_pointer_cast(d_out.data()));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<int> expected{-3, -1, 1, 3, 5, 1, 3, 5, 7, 9, 5, 7, 9, 11, 13, 9, 11, 13, 15, 17};
  REQUIRE(expected == d_out);
}
