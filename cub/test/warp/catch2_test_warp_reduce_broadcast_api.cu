// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/warp/warp_reduce_broadcast.cuh>

#include <thrust/detail/raw_pointer_cast.h>

#include <c2h/catch2_test_helper.h>

__global__ __launch_bounds__(32) void WarpReduceBroadcastOverviewKernel(int* out)
{
  // example-begin warp-reduce-broadcast-overview
  using WarpReduceBroadcast = cub::WarpReduceBroadcast<int>;

  __shared__ typename WarpReduceBroadcast::TempStorage temp_storage;

  int thread_data = static_cast<int>(threadIdx.x);
  int aggregate   = WarpReduceBroadcast{temp_storage}.Sum(thread_data);

  out[threadIdx.x] = aggregate;
  // example-end warp-reduce-broadcast-overview
}

C2H_TEST("WarpReduceBroadcast overview documentation kernel", "[warp][reduce][broadcast]")
{
  c2h::device_vector<int> d_out(32);

  WarpReduceBroadcastOverviewKernel<<<1, 32>>>(thrust::raw_pointer_cast(d_out.data()));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<int> expected(32, 496);
  REQUIRE(expected == d_out);
}
