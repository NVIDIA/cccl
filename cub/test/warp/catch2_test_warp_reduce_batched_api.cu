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
  constexpr int num_batches = 3;
  using WarpReduceBatched   = cub::WarpReduceBatched<int, num_batches>;

  // Assume 64 threads per block, so 64 / 32 = 2 logical warps
  // Each logical warp has its own TempStorage
  __shared__ typename WarpReduceBatched::TempStorage temp_storage[2];

  const int warp_id = static_cast<int>(threadIdx.x) / 32;

  const int tid = static_cast<int>(threadIdx.x);
  int thread_data[num_batches];
  thread_data[0] = tid - 1;
  thread_data[1] = tid;
  thread_data[2] = tid + 1;

  int result = WarpReduceBatched{temp_storage[warp_id]}.Reduce(thread_data, cuda::maximum{});
  // results across threads: [30, 31, 32, ?, ?, ..., ?, 62, 63, 64, ?, ?, ..., ?]
  // example-end warp-reduce-batched-overview

  const int lane_id = static_cast<int>(threadIdx.x) % 32;
  if (lane_id < num_batches)
  {
    out[warp_id * num_batches + lane_id] = result;
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
  constexpr int num_batches          = 3;
  constexpr int logical_warp_threads = 16;
  using WarpReduceBatched            = cub::WarpReduceBatched<int, num_batches, logical_warp_threads>;

  // Only the first logical warp participates, so only a single TempStorage is needed
  __shared__ typename WarpReduceBatched::TempStorage temp_storage;

  const int tid = static_cast<int>(threadIdx.x);
  int result{};
  if (threadIdx.x < logical_warp_threads)
  {
    const cuda::std::array<int, num_batches> inputs{tid - 1, tid, tid + 1};
    result = WarpReduceBatched{temp_storage}.Reduce(inputs, cuda::maximum{});
  }
  // results across threads: [14, 15, 16, ?, ?, ..., ?, 0, 0, ..., 0]
  __syncthreads();
  // Can reuse TempStorage after the barrier.
  // example-end warp-reduce-batched-reduce
  _CCCL_ASSERT(tid < logical_warp_threads || result == 0, "");
  if (threadIdx.x < num_batches)
  {
    out[tid] = result;
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
  constexpr int num_batches          = 3;
  constexpr int logical_warp_threads = 2;
  using WarpReduceBatched            = cub::WarpReduceBatched<int, num_batches, logical_warp_threads>;

  // Assume 8 threads per block, so 8 / 2 = 4 logical warps
  // Only every other logical warp participates, so only 2 TempStorage are needed
  __shared__ typename WarpReduceBatched::TempStorage temp_storage[2];

  const int tid               = static_cast<int>(threadIdx.x);
  const int logical_warp_id   = tid / logical_warp_threads;
  const bool is_participating = logical_warp_id % 2 == 0;
  const int participant_idx   = logical_warp_id / 2;

  constexpr int max_out_per_thread = cuda::ceil_div(num_batches, logical_warp_threads);
  cuda::std::array<int, max_out_per_thread> results{};
  if (is_participating)
  {
    const cuda::std::array<int, num_batches> inputs{tid - 1, tid, tid + 1};
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
    int const logical_lane_id = tid % logical_warp_threads;
    for (int i = 0; i < max_out_per_thread; ++i)
    {
      // Striped
      const int batch_idx = i * logical_warp_threads + logical_lane_id;
      if (batch_idx < num_batches)
      {
        out[participant_idx * num_batches + batch_idx] = results[i];
      }
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
  constexpr int num_batches          = 3;
  constexpr int logical_warp_threads = 2;
  using WarpReduceBatched            = cub::WarpReduceBatched<int, num_batches, logical_warp_threads>;

  // Assume 8 threads per block, so 8 / 2 = 4 logical warps
  // Only every other logical warp participates, so only 2 TempStorage are needed
  __shared__ typename WarpReduceBatched::TempStorage temp_storage[2];

  const int tid               = static_cast<int>(threadIdx.x);
  const int logical_warp_id   = tid / logical_warp_threads;
  const bool is_participating = logical_warp_id % 2 == 0;
  const int participant_idx   = logical_warp_id / 2;

  constexpr int max_out_per_thread = cuda::ceil_div(num_batches, logical_warp_threads);
  cuda::std::array<int, max_out_per_thread> results{};
  if (is_participating)
  {
    const cuda::std::array<int, num_batches> inputs{tid - 1, tid, tid + 1};

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
    int const logical_lane_id = tid % logical_warp_threads;
    for (int i = 0; i < max_out_per_thread; ++i)
    {
      // Blocked
      const int batch_idx = logical_lane_id * max_out_per_thread + i;
      if (batch_idx < num_batches)
      {
        out[participant_idx * num_batches + batch_idx] = results[i];
      }
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
  constexpr int num_batches          = 3;
  constexpr int logical_warp_threads = 4;
  // We can enable physical warp synchronization since all non-exited lanes do participate in the primitive.
  constexpr bool sync_physical_warp = true;
  using WarpReduceBatched = cub::WarpReduceBatched<int, num_batches, logical_warp_threads, sync_physical_warp>;

  // Assume 8 threads per block, so 8 / 4 = 2 logical warps
  __shared__ typename WarpReduceBatched::TempStorage temp_storage[2];

  const int tid             = static_cast<int>(threadIdx.x);
  const int logical_warp_id = tid / logical_warp_threads;

  cuda::std::array<int, num_batches> inputs{tid - 1, tid, tid + 1};
  int result = WarpReduceBatched{temp_storage[logical_warp_id]}.Sum(inputs);
  // results across threads:
  // [2, 6, 10, ?, 18, 22, 26, ?]
  // example-end warp-reduce-batched-sum
  const int logical_lane_id = tid % logical_warp_threads;
  if (logical_lane_id < num_batches)
  {
    out[logical_warp_id * num_batches + logical_lane_id] = result;
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
  constexpr int num_batches          = 5;
  constexpr int logical_warp_threads = 2;
  // We can enable physical warp synchronization since all non-exited lanes do participate in the primitive.
  constexpr bool sync_physical_warp = true;
  using WarpReduceBatched = cub::WarpReduceBatched<int, num_batches, logical_warp_threads, sync_physical_warp>;

  // Assume 8 threads per block, so 8 / 2 = 4 logical warps
  __shared__ typename WarpReduceBatched::TempStorage temp_storage[4];

  const int tid             = static_cast<int>(threadIdx.x);
  const int logical_warp_id = tid / logical_warp_threads;

  cuda::std::array<int, num_batches> inputs{tid - 2, tid - 1, tid, tid + 1, tid + 2};
  constexpr int max_out_per_thread = cuda::ceil_div(num_batches, logical_warp_threads);
  // Use a static size span to alias the last 3 elements of inputs for results.
  cuda::std::span<int, max_out_per_thread> results{cuda::std::end(inputs) - max_out_per_thread, max_out_per_thread};
  WarpReduceBatched{temp_storage[logical_warp_id]}.SumToStriped(inputs, results);
  // results across threads:
  // [[-3, 1, 5], [-1, 3, ?], [1, 5, 9], [3, 7, ?], ..., [9, 13, 17], [11, 15, ?]]
  // example-end warp-reduce-batched-sum-to-striped

  const int logical_lane_id = tid % logical_warp_threads;
  for (int i = 0; i < max_out_per_thread; ++i)
  {
    // Striped
    const int batch_idx = i * logical_warp_threads + logical_lane_id;
    if (batch_idx < num_batches)
    {
      out[logical_warp_id * num_batches + batch_idx] = results[i];
    }
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
  constexpr int num_batches          = 5;
  constexpr int logical_warp_threads = 2;
  constexpr bool sync_physical_warp  = true;
  using WarpReduceBatched = cub::WarpReduceBatched<int, num_batches, logical_warp_threads, sync_physical_warp>;

  // Assume 8 threads per block, so 8 / 2 = 4 logical warps
  __shared__ typename WarpReduceBatched::TempStorage temp_storage[4];

  const int tid             = static_cast<int>(threadIdx.x);
  const int logical_warp_id = tid / logical_warp_threads;

  cuda::std::array<int, num_batches> inputs{tid - 2, tid - 1, tid, tid + 1, tid + 2};
  constexpr int max_out_per_thread = cuda::ceil_div(num_batches, logical_warp_threads);
  // Use a static size span to alias the last 3 elements of inputs for results.
  cuda::std::span<int, max_out_per_thread> results{cuda::std::end(inputs) - max_out_per_thread, max_out_per_thread};
  WarpReduceBatched{temp_storage[logical_warp_id]}.SumToBlocked(inputs, results);
  // results across threads:
  // [[-3, -1, 1], [3, 5, ?], [1, 3, 5], [7, 9, ?], ..., [9, 11, 13], [15, 17, ?]]
  // example-end warp-reduce-batched-sum-to-blocked

  const int logical_lane_id = tid % logical_warp_threads;
  for (int i = 0; i < max_out_per_thread; ++i)
  {
    // Blocked
    const int batch_idx = logical_lane_id * max_out_per_thread + i;
    if (batch_idx < num_batches)
    {
      out[logical_warp_id * num_batches + batch_idx] = results[i];
    }
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
