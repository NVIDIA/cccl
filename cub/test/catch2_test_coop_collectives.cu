// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/block/block_reduce_broadcast.cuh>
#include <cub/block/block_row_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/warp/warp_reduce.cuh>
#include <cub/warp/warp_reduce_batched.cuh>
#include <cub/warp/warp_reduce_batched_broadcast.cuh>
#include <cub/warp/warp_reduce_broadcast.cuh>
#include <cub/warp/warp_scan.cuh>

#include <thrust/detail/raw_pointer_cast.h>

#include <cuda/std/array>

#include <c2h/catch2_test_helper.h>

struct max_int_op
{
  _CCCL_DEVICE_API _CCCL_FORCEINLINE int operator()(int lhs, int rhs) const
  {
    return lhs > rhs ? lhs : rhs;
  }
};

static_assert(sizeof(typename cub::BlockRowReduceWarpBroadcast<float, 1, 4>::TempStorage) == 16);

__global__ void WarpReduceBroadcastKernel(int* out)
{
  // example-begin warp-reduce-broadcast
  using warp_reduce_t = cub::WarpReduceBroadcast<int>;
  __shared__ typename warp_reduce_t::TempStorage temp_storage[2];

  const int warp_id = static_cast<int>(threadIdx.x) / 32;
  const int result  = warp_reduce_t(temp_storage[warp_id]).Sum(static_cast<int>(threadIdx.x));
  // example-end warp-reduce-broadcast

  out[threadIdx.x] = result;
}

C2H_TEST("warp reduce broadcast returns aggregate to every lane", "[coop][warp][reduce]")
{
  c2h::device_vector<int> d_out(64);

  WarpReduceBroadcastKernel<<<1, 64>>>(thrust::raw_pointer_cast(d_out.data()));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<int> expected(64);
  for (int i = 0; i < 64; ++i)
  {
    expected[i] = i < 32 ? 496 : 1520;
  }
  REQUIRE(expected == d_out);
}

__global__ void WarpReduceBroadcastLogicalKernel(int* out)
{
  // example-begin warp-reduce-broadcast-logical
  using warp_reduce_t = cub::WarpReduceBroadcast<int, 4>;
  __shared__ typename warp_reduce_t::TempStorage temp_storage[2];

  const int physical_warp_id = static_cast<int>(threadIdx.x) / 32;
  const int result           = warp_reduce_t(temp_storage[physical_warp_id]).Sum(static_cast<int>(threadIdx.x));
  // example-end warp-reduce-broadcast-logical

  out[threadIdx.x] = result;
}

C2H_TEST("warp reduce broadcast supports tiny logical warps", "[coop][warp][reduce]")
{
  c2h::device_vector<int> d_out(64);

  WarpReduceBroadcastLogicalKernel<<<1, 64>>>(thrust::raw_pointer_cast(d_out.data()));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<int> expected(64);
  for (int i = 0; i < 64; ++i)
  {
    const int group_start = (i / 4) * 4;
    expected[i]           = group_start * 4 + 6;
  }
  REQUIRE(expected == d_out);
}

__global__ void WarpReduceBatchedOwnerKernel(int* out)
{
  // example-begin warp-reduce-batched-owner
  constexpr int batches = 3;
  using warp_reduce_t   = cub::WarpReduceBatched<int, batches>;
  __shared__ typename warp_reduce_t::TempStorage temp_storage[2];

  const int tid     = static_cast<int>(threadIdx.x);
  const int warp_id = tid / 32;
  const int lane_id = tid % 32;

  int inputs[batches] = {tid, tid + 1, tid + 2};
  int result          = warp_reduce_t(temp_storage[warp_id]).Sum(inputs);
  // example-end warp-reduce-batched-owner

  if (lane_id < batches)
  {
    out[warp_id * batches + lane_id] = result;
  }
}

C2H_TEST("warp reduce batched owner-lane layout returns one batch per lane", "[coop][warp][reduce]")
{
  c2h::device_vector<int> d_out(6);

  WarpReduceBatchedOwnerKernel<<<1, 64>>>(thrust::raw_pointer_cast(d_out.data()));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<int> expected{496, 528, 560, 1520, 1552, 1584};
  REQUIRE(expected == d_out);
}

__global__ void WarpReduceBatchedBroadcastKernel(int* out)
{
  // example-begin warp-reduce-batched-broadcast
  constexpr int batches              = 5;
  constexpr int logical_warp_threads = 4;
  using warp_reduce_t                = cub::WarpReduceBatchedBroadcast<int, batches, logical_warp_threads>;
  __shared__ typename warp_reduce_t::TempStorage temp_storage[2];

  const int physical_warp_id = static_cast<int>(threadIdx.x) / 32;
  const int logical_lane     = static_cast<int>(threadIdx.x) % logical_warp_threads;

  ::cuda::std::array<int, batches> inputs{};
  for (int batch = 0; batch < batches; ++batch)
  {
    inputs[batch] = batch * 10 + logical_lane;
  }

  const auto outputs = warp_reduce_t(temp_storage[physical_warp_id]).Sum(inputs);
  // example-end warp-reduce-batched-broadcast

  for (int batch = 0; batch < batches; ++batch)
  {
    out[threadIdx.x * batches + batch] = outputs[batch];
  }
}

C2H_TEST("warp reduce batched broadcast returns every batch to every lane", "[coop][warp][reduce]")
{
  constexpr int threads = 64;
  constexpr int batches = 5;
  c2h::device_vector<int> d_out(threads * batches);

  WarpReduceBatchedBroadcastKernel<<<1, threads>>>(thrust::raw_pointer_cast(d_out.data()));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<int> expected(threads * batches);
  for (int i = 0; i < threads; ++i)
  {
    for (int batch = 0; batch < batches; ++batch)
    {
      expected[i * batches + batch] = batch * 40 + 6;
    }
  }
  REQUIRE(expected == d_out);
}

__global__ void BlockReduceBroadcastKernel(int* out)
{
  // example-begin block-reduce-broadcast
  using block_reduce_t = cub::BlockReduceBroadcast<int, 128>;
  __shared__ typename block_reduce_t::TempStorage temp_storage;

  const int result = block_reduce_t(temp_storage).Sum(static_cast<int>(threadIdx.x));
  // example-end block-reduce-broadcast

  out[threadIdx.x] = result;
}

C2H_TEST("block reduce broadcast returns aggregate to every thread", "[coop][block][reduce]")
{
  c2h::device_vector<int> d_out(128);

  BlockReduceBroadcastKernel<<<1, 128>>>(thrust::raw_pointer_cast(d_out.data()));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<int> expected(128, 8128);
  REQUIRE(expected == d_out);
}

__global__ void BlockRowReduceKernel(int* out)
{
  // example-begin block-row-reduce
  using row_reduce_t = cub::BlockRowReduce<int, 2, 2>;
  __shared__ typename row_reduce_t::TempStorage temp_storage;

  const int result = row_reduce_t(temp_storage).Sum(static_cast<int>(threadIdx.x));
  // example-end block-row-reduce

  out[threadIdx.x] = result;
}

C2H_TEST("block row reduce returns one aggregate per row", "[coop][block][reduce]")
{
  c2h::device_vector<int> d_out(128);

  BlockRowReduceKernel<<<1, 128>>>(thrust::raw_pointer_cast(d_out.data()));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<int> expected(128);
  for (int i = 0; i < 128; ++i)
  {
    expected[i] = i < 64 ? 2016 : 6112;
  }
  REQUIRE(expected == d_out);
}

__global__ void BlockRowReduceWarpBroadcastKernel(int* out)
{
  // example-begin block-row-reduce-warp-broadcast
  using row_reduce_t = cub::BlockRowReduceWarpBroadcast<int, 2, 2>;
  __shared__ typename row_reduce_t::TempStorage temp_storage;

  const int result = row_reduce_t(temp_storage).Sum(static_cast<int>(threadIdx.x));
  // example-end block-row-reduce-warp-broadcast

  out[threadIdx.x] = result;
}

C2H_TEST("block row reduce warp broadcast returns one aggregate per row", "[coop][block][reduce]")
{
  c2h::device_vector<int> d_out(128);

  BlockRowReduceWarpBroadcastKernel<<<1, 128>>>(thrust::raw_pointer_cast(d_out.data()));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<int> expected(128);
  for (int i = 0; i < 128; ++i)
  {
    expected[i] = i < 64 ? 2016 : 6112;
  }
  REQUIRE(expected == d_out);
}

__global__ void BlockRowReduceMaxKernel(int* out)
{
  using row_reduce_t = cub::BlockRowReduce<int, 2, 2>;
  __shared__ typename row_reduce_t::TempStorage temp_storage;

  const int value  = static_cast<int>(threadIdx.x) - 200;
  const int result = row_reduce_t(temp_storage).Reduce(value, max_int_op{});

  out[threadIdx.x] = result;
}

__global__ void BlockRowReduceWarpBroadcastMaxKernel(int* out)
{
  using row_reduce_t = cub::BlockRowReduceWarpBroadcast<int, 2, 2>;
  __shared__ typename row_reduce_t::TempStorage temp_storage;

  const int value  = static_cast<int>(threadIdx.x) - 200;
  const int result = row_reduce_t(temp_storage).CommutativeReduce(value, max_int_op{}, -10000);

  out[threadIdx.x] = result;
}

C2H_TEST("block row reduce supports custom row reductions", "[coop][block][reduce]")
{
  c2h::device_vector<int> d_row_out(128);
  c2h::device_vector<int> d_warp_broadcast_out(128);

  BlockRowReduceMaxKernel<<<1, 128>>>(thrust::raw_pointer_cast(d_row_out.data()));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  BlockRowReduceWarpBroadcastMaxKernel<<<1, 128>>>(thrust::raw_pointer_cast(d_warp_broadcast_out.data()));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<int> expected(128);
  for (int i = 0; i < 128; ++i)
  {
    expected[i] = i < 64 ? -137 : -73;
  }
  REQUIRE(expected == d_row_out);
  REQUIRE(expected == d_warp_broadcast_out);
}

__global__ void WarpSegmentedReduceKernel(int* out)
{
  // example-begin warp-segmented-row-reduce
  using warp_reduce_t = cub::WarpReduce<int>;
  __shared__ typename warp_reduce_t::TempStorage temp_storage;

  const int lane_id   = static_cast<int>(threadIdx.x);
  const int head_flag = (lane_id % 8) == 0;
  const int result    = warp_reduce_t(temp_storage).HeadSegmentedSum(lane_id, head_flag);
  // example-end warp-segmented-row-reduce

  if (head_flag)
  {
    out[lane_id / 8] = result;
  }
}

C2H_TEST("warp segmented reduce maps fixed row segments to segment heads", "[coop][warp][reduce]")
{
  c2h::device_vector<int> d_out(4);

  WarpSegmentedReduceKernel<<<1, 32>>>(thrust::raw_pointer_cast(d_out.data()));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<int> expected{28, 92, 156, 220};
  REQUIRE(expected == d_out);
}

__global__ void WarpAndBlockScanKernel(int* warp_out, int* block_out)
{
  // example-begin warp-block-scan
  using warp_scan_t  = cub::WarpScan<int>;
  using block_scan_t = cub::BlockScan<int, 64>;

  __shared__ typename warp_scan_t::TempStorage warp_storage[2];
  __shared__ typename block_scan_t::TempStorage block_storage;

  const int tid     = static_cast<int>(threadIdx.x);
  const int warp_id = tid / 32;

  warp_out[tid] = warp_scan_t(warp_storage[warp_id]).Broadcast(tid, 0);

  int prefix{};
  block_scan_t(block_storage).ExclusiveSum(1, prefix);
  block_out[tid] = prefix;
  // example-end warp-block-scan
}

C2H_TEST("warp broadcast and block scan cover scan-style cooperative primitives", "[coop][scan]")
{
  c2h::device_vector<int> d_warp_out(64);
  c2h::device_vector<int> d_block_out(64);

  WarpAndBlockScanKernel<<<1, 64>>>(
    thrust::raw_pointer_cast(d_warp_out.data()), thrust::raw_pointer_cast(d_block_out.data()));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<int> expected_warp(64);
  c2h::host_vector<int> expected_block(64);
  for (int i = 0; i < 64; ++i)
  {
    expected_warp[i]  = i < 32 ? 0 : 32;
    expected_block[i] = i;
  }

  REQUIRE(expected_warp == d_warp_out);
  REQUIRE(expected_block == d_block_out);
}
