// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_topk.cuh>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/output_ordering.h>
#include <cuda/__execution/require.h>
#include <cuda/devices>
#include <cuda/std/__execution/env.h>
#include <cuda/std/functional>
#include <cuda/stream>

#include <algorithm>
#include <iostream>

#include <c2h/catch2_test_helper.h>

// Simple user-defined key type for the decomposer-based examples.
struct topk_custom_t
{
  int rank;
  int payload;
};

struct topk_custom_decomposer_t
{
  __host__ __device__ ::cuda::std::tuple<int&> operator()(topk_custom_t& key) const
  {
    return {key.rank};
  }
  __host__ __device__ ::cuda::std::tuple<const int&> operator()(const topk_custom_t& key) const
  {
    return {key.rank};
  }
};

C2H_TEST("cub::DeviceTopK::MaxKeys env-alloc accepts stream_ref", "[topk][env]")
{
  // example-begin topk-max-keys-env
  auto d_in  = thrust::device_vector<int>{8, 6, 7, 5, 3, 0, 9, 1, 4, 2};
  auto d_out = thrust::device_vector<int>(3);
  int k      = 3;

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = cuda::std::execution::env{
    cuda::execution::require(cuda::execution::determinism::not_guaranteed, cuda::execution::output_ordering::unsorted),
    stream_ref};

  auto error = cub::DeviceTopK::MaxKeys(d_in.begin(), d_out.begin(), static_cast<int>(d_in.size()), k, env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceTopK::MaxKeys failed with status: " << error << '\n';
  }
  thrust::device_vector<int> expected{9, 8, 7}; // possibly in different order
  // example-end topk-max-keys-env

  stream.sync();
  REQUIRE(error == cudaSuccess);

  // Result order is unspecified for TopK; sort before compare.
  thrust::sort(d_out.begin(), d_out.end(), cuda::std::greater<int>{});
  REQUIRE(d_out == expected);
}

C2H_TEST("cub::DeviceTopK::MinKeys env-alloc accepts stream_ref", "[topk][env]")
{
  // example-begin topk-min-keys-env
  auto d_in  = thrust::device_vector<int>{8, 6, 7, 5, 3, 0, 9, 1, 4, 2};
  auto d_out = thrust::device_vector<int>(3);
  int k      = 3;

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = cuda::std::execution::env{
    cuda::execution::require(cuda::execution::determinism::not_guaranteed, cuda::execution::output_ordering::unsorted),
    stream_ref};

  auto error = cub::DeviceTopK::MinKeys(d_in.begin(), d_out.begin(), static_cast<int>(d_in.size()), k, env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceTopK::MinKeys failed with status: " << error << '\n';
  }
  thrust::device_vector<int> expected{0, 1, 2}; // possibly in different order
  // example-end topk-min-keys-env

  stream.sync();
  REQUIRE(error == cudaSuccess);

  thrust::sort(d_out.begin(), d_out.end());
  REQUIRE(d_out == expected);
}

C2H_TEST("cub::DeviceTopK::MaxPairs env-alloc accepts stream_ref", "[topk][env]")
{
  // example-begin topk-max-pairs-env
  auto d_keys_in    = thrust::device_vector<int>{8, 6, 7, 5, 3, 0, 9, 1, 4, 2};
  auto d_values_in  = thrust::device_vector<int>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  auto d_keys_out   = thrust::device_vector<int>(3);
  auto d_values_out = thrust::device_vector<int>(3);
  int k             = 3;

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = cuda::std::execution::env{
    cuda::execution::require(cuda::execution::determinism::not_guaranteed, cuda::execution::output_ordering::unsorted),
    stream_ref};

  auto error = cub::DeviceTopK::MaxPairs(
    d_keys_in.begin(),
    d_keys_out.begin(),
    d_values_in.begin(),
    d_values_out.begin(),
    static_cast<int>(d_keys_in.size()),
    k,
    env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceTopK::MaxPairs failed with status: " << error << '\n';
  }
  thrust::device_vector<int> expected_keys{9, 8, 7}; // possibly in different order
  // example-end topk-max-pairs-env

  stream.sync();
  REQUIRE(error == cudaSuccess);

  thrust::sort(d_keys_out.begin(), d_keys_out.end(), cuda::std::greater<int>{});
  REQUIRE(d_keys_out == expected_keys);
}

C2H_TEST("cub::DeviceTopK::MinPairs env-alloc accepts stream_ref", "[topk][env]")
{
  // example-begin topk-min-pairs-env
  auto d_keys_in    = thrust::device_vector<int>{8, 6, 7, 5, 3, 0, 9, 1, 4, 2};
  auto d_values_in  = thrust::device_vector<int>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  auto d_keys_out   = thrust::device_vector<int>(3);
  auto d_values_out = thrust::device_vector<int>(3);
  int k             = 3;

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = cuda::std::execution::env{
    cuda::execution::require(cuda::execution::determinism::not_guaranteed, cuda::execution::output_ordering::unsorted),
    stream_ref};

  auto error = cub::DeviceTopK::MinPairs(
    d_keys_in.begin(),
    d_keys_out.begin(),
    d_values_in.begin(),
    d_values_out.begin(),
    static_cast<int>(d_keys_in.size()),
    k,
    env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceTopK::MinPairs failed with status: " << error << '\n';
  }
  thrust::device_vector<int> expected_keys{0, 1, 2}; // possibly in different order
  // example-end topk-min-pairs-env

  stream.sync();
  REQUIRE(error == cudaSuccess);

  thrust::sort(d_keys_out.begin(), d_keys_out.end());
  REQUIRE(d_keys_out == expected_keys);
}

C2H_TEST("cub::DeviceTopK::MaxKeys env-alloc with decomposer accepts stream_ref", "[topk][env]")
{
  // example-begin topk-max-keys-decomposer-env
  thrust::host_vector<topk_custom_t> h_in{
    {8, 0}, {6, 1}, {7, 2}, {5, 3}, {3, 4}, {0, 5}, {9, 6}, {1, 7}, {4, 8}, {2, 9}};
  thrust::device_vector<topk_custom_t> d_in = h_in;
  thrust::device_vector<topk_custom_t> d_out(3);
  int k = 3;

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = cuda::std::execution::env{
    cuda::execution::require(cuda::execution::determinism::not_guaranteed, cuda::execution::output_ordering::unsorted),
    stream_ref};

  auto error = cub::DeviceTopK::MaxKeys(
    d_in.begin(), d_out.begin(), static_cast<int>(d_in.size()), k, topk_custom_decomposer_t{}, env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceTopK::MaxKeys failed with status: " << error << '\n';
  }
  thrust::host_vector<int> expected_ranks{9, 8, 7}; // possibly in different order
  // example-end topk-max-keys-decomposer-env

  stream.sync();
  REQUIRE(error == cudaSuccess);

  thrust::host_vector<topk_custom_t> h_out = d_out;
  std::sort(h_out.begin(), h_out.end(), [](const topk_custom_t& a, const topk_custom_t& b) {
    return a.rank > b.rank;
  });
  thrust::host_vector<int> actual_ranks{h_out[0].rank, h_out[1].rank, h_out[2].rank};
  REQUIRE(actual_ranks == expected_ranks);
}

C2H_TEST("cub::DeviceTopK::MinKeys env-alloc with decomposer accepts stream_ref", "[topk][env]")
{
  // example-begin topk-min-keys-decomposer-env
  thrust::host_vector<topk_custom_t> h_in{
    {8, 0}, {6, 1}, {7, 2}, {5, 3}, {3, 4}, {0, 5}, {9, 6}, {1, 7}, {4, 8}, {2, 9}};
  thrust::device_vector<topk_custom_t> d_in = h_in;
  thrust::device_vector<topk_custom_t> d_out(3);
  int k = 3;

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = cuda::std::execution::env{
    cuda::execution::require(cuda::execution::determinism::not_guaranteed, cuda::execution::output_ordering::unsorted),
    stream_ref};

  auto error = cub::DeviceTopK::MinKeys(
    d_in.begin(), d_out.begin(), static_cast<int>(d_in.size()), k, topk_custom_decomposer_t{}, env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceTopK::MinKeys failed with status: " << error << '\n';
  }
  thrust::host_vector<int> expected_ranks{0, 1, 2}; // possibly in different order
  // example-end topk-min-keys-decomposer-env

  stream.sync();
  REQUIRE(error == cudaSuccess);

  thrust::host_vector<topk_custom_t> h_out = d_out;
  std::sort(h_out.begin(), h_out.end(), [](const topk_custom_t& a, const topk_custom_t& b) {
    return a.rank < b.rank;
  });
  thrust::host_vector<int> actual_ranks{h_out[0].rank, h_out[1].rank, h_out[2].rank};
  REQUIRE(actual_ranks == expected_ranks);
}

C2H_TEST("cub::DeviceTopK::MaxPairs env-alloc with decomposer accepts stream_ref", "[topk][env]")
{
  // example-begin topk-max-pairs-decomposer-env
  thrust::host_vector<topk_custom_t> h_keys_in{
    {8, 0}, {6, 1}, {7, 2}, {5, 3}, {3, 4}, {0, 5}, {9, 6}, {1, 7}, {4, 8}, {2, 9}};
  thrust::device_vector<topk_custom_t> d_keys_in = h_keys_in;
  thrust::device_vector<int> d_values_in{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  thrust::device_vector<topk_custom_t> d_keys_out(3);
  thrust::device_vector<int> d_values_out(3);
  int k = 3;

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = cuda::std::execution::env{
    cuda::execution::require(cuda::execution::determinism::not_guaranteed, cuda::execution::output_ordering::unsorted),
    stream_ref};

  auto error = cub::DeviceTopK::MaxPairs(
    d_keys_in.begin(),
    d_keys_out.begin(),
    d_values_in.begin(),
    d_values_out.begin(),
    static_cast<int>(d_keys_in.size()),
    k,
    topk_custom_decomposer_t{},
    env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceTopK::MaxPairs failed with status: " << error << '\n';
  }
  thrust::host_vector<int> expected_ranks{9, 8, 7}; // possibly in different order
  // example-end topk-max-pairs-decomposer-env

  stream.sync();
  REQUIRE(error == cudaSuccess);

  thrust::host_vector<topk_custom_t> h_keys_out = d_keys_out;
  std::sort(h_keys_out.begin(), h_keys_out.end(), [](const topk_custom_t& a, const topk_custom_t& b) {
    return a.rank > b.rank;
  });
  thrust::host_vector<int> actual_ranks{h_keys_out[0].rank, h_keys_out[1].rank, h_keys_out[2].rank};
  REQUIRE(actual_ranks == expected_ranks);
}

C2H_TEST("cub::DeviceTopK::MinPairs env-alloc with decomposer accepts stream_ref", "[topk][env]")
{
  // example-begin topk-min-pairs-decomposer-env
  thrust::host_vector<topk_custom_t> h_keys_in{
    {8, 0}, {6, 1}, {7, 2}, {5, 3}, {3, 4}, {0, 5}, {9, 6}, {1, 7}, {4, 8}, {2, 9}};
  thrust::device_vector<topk_custom_t> d_keys_in = h_keys_in;
  thrust::device_vector<int> d_values_in{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  thrust::device_vector<topk_custom_t> d_keys_out(3);
  thrust::device_vector<int> d_values_out(3);
  int k = 3;

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = cuda::std::execution::env{
    cuda::execution::require(cuda::execution::determinism::not_guaranteed, cuda::execution::output_ordering::unsorted),
    stream_ref};

  auto error = cub::DeviceTopK::MinPairs(
    d_keys_in.begin(),
    d_keys_out.begin(),
    d_values_in.begin(),
    d_values_out.begin(),
    static_cast<int>(d_keys_in.size()),
    k,
    topk_custom_decomposer_t{},
    env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceTopK::MinPairs failed with status: " << error << '\n';
  }
  thrust::host_vector<int> expected_ranks{0, 1, 2}; // possibly in different order
  // example-end topk-min-pairs-decomposer-env

  stream.sync();
  REQUIRE(error == cudaSuccess);

  thrust::host_vector<topk_custom_t> h_keys_out = d_keys_out;
  std::sort(h_keys_out.begin(), h_keys_out.end(), [](const topk_custom_t& a, const topk_custom_t& b) {
    return a.rank < b.rank;
  });
  thrust::host_vector<int> actual_ranks{h_keys_out[0].rank, h_keys_out[1].rank, h_keys_out[2].rank};
  REQUIRE(actual_ranks == expected_ranks);
}
