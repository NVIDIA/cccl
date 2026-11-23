// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/thread/thread_operators.cuh>
#include <cub/util_type.cuh>
#include <cub/warp/warp_scan.cuh>

#include <cuda/std/functional>

#include "cub_test_macros.h"

// Tests the specialization of WarpScan::InclusiveScan for cub::ReduceByKeyOp: a scan of the values that starts again
// where the key changes. The first segment of a warp (the one that begins at lane 0) has to be accumulated as well.

constexpr int warp_threads = 32;
constexpr int total_warps  = 2;

template <class T>
__global__ void reduce_by_key_scan_kernel(const int* keys, const T* values, T* out)
{
  using pair_t      = cub::KeyValuePair<int, T>;
  using warp_scan_t = cub::WarpScan<pair_t, warp_threads>;
  __shared__ typename warp_scan_t::TempStorage storage[total_warps];

  const int tid = static_cast<int>(threadIdx.x);
  pair_t item(keys[tid], values[tid]);
  pair_t result;
  warp_scan_t(storage[tid / warp_threads]).InclusiveScan(item, result, cub::ReduceByKeyOp<::cuda::std::plus<>>{});
  out[tid] = result.value;
}

using types = c2h::type_list<std::int32_t, std::int64_t>;

CUB_TEST("WarpScan inclusive scan by key accumulates every segment, the first one too", "[scan][warp]", CUB_SMALL, types)
{
  using type = c2h::get<0, TestType>;

  // The length of the first segment, and of the others (a segment that is as long as the warp is all of it)
  const int first_length = GENERATE(1, 2, 5, 16, 31, 32);
  const int length       = GENERATE(1, 3, 7, 32);
  CAPTURE(first_length, length);

  constexpr int num_items = warp_threads * total_warps;
  c2h::host_vector<int> h_keys(num_items, thrust::no_init);
  c2h::host_vector<type> h_values(num_items, thrust::no_init);
  c2h::host_vector<type> h_expected(num_items, thrust::no_init);
  for (int tid = 0; tid < num_items; ++tid)
  {
    const int lane = tid % warp_threads;
    // (The keys of the second warp are different, but that does not matter: the warps are scanned separately.)
    h_keys[tid]     = (lane < first_length ? 0 : 1 + (lane - first_length) / length) + 100 * (tid / warp_threads);
    h_values[tid]   = static_cast<type>(tid % 7 + 1);
    h_expected[tid] = (lane == 0 || h_keys[tid] != h_keys[tid - 1])
                      ? h_values[tid]
                      : static_cast<type>(h_expected[tid - 1] + h_values[tid]);
  }

  c2h::device_vector<int> d_keys    = h_keys;
  c2h::device_vector<type> d_values = h_values;
  c2h::device_vector<type> d_out(num_items, thrust::no_init);
  reduce_by_key_scan_kernel<type><<<1, num_items>>>(
    thrust::raw_pointer_cast(d_keys.data()),
    thrust::raw_pointer_cast(d_values.data()),
    thrust::raw_pointer_cast(d_out.data()));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  REQUIRE(h_expected == d_out);
}
