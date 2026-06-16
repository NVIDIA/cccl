// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_batched_topk.cuh>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/output_ordering.h>
#include <cuda/__execution/require.h>
#include <cuda/argument>
#include <cuda/devices>
#include <cuda/iterator>
#include <cuda/std/__execution/env.h>
#include <cuda/std/functional>
#include <cuda/stream>

#include <iostream>

#include <c2h/catch2_test_helper.h>

C2H_TEST("cub::DeviceBatchedTopK::MaxKeys env-alloc example", "[batched_topk][device][env]")
{
  // example-begin batched-topk-max-keys-env
  constexpr int num_segments = 2;
  constexpr int segment_size = 8;
  constexpr int k            = 3;

  auto keys_in  = thrust::device_vector<int>{5, -3, 1, 7, 8, 2, 4, 6, /**/ 0, 9, 3, 2, 1, 8, 7, 4};
  auto keys_out = thrust::device_vector<int>(num_segments * k, thrust::no_init);

  auto d_keys_in =
    cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(keys_in.data())), segment_size);
  auto d_keys_out =
    cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(keys_out.data())), k);

  cuda::stream stream{cuda::devices[0]};
  auto env = cuda::std::execution::env{
    cuda::execution::require(cuda::execution::determinism::not_guaranteed, cuda::execution::output_ordering::unsorted),
    cuda::stream_ref{stream}};

  // The env-based overload allocates and frees the temporary storage internally.
  auto error = cub::DeviceBatchedTopK::MaxKeys(
    d_keys_in,
    d_keys_out,
    cuda::args::constant<segment_size>{},
    cuda::args::constant<k>{},
    cuda::args::immediate{cuda::std::int64_t{num_segments}},
    env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceBatchedTopK::MaxKeys failed with status: " << error << '\n';
  }
  // example-end batched-topk-max-keys-env

  stream.sync();
  REQUIRE(error == cudaSuccess);
  thrust::sort(keys_out.begin(), keys_out.begin() + k, cuda::std::greater<int>{});
  thrust::sort(keys_out.begin() + k, keys_out.begin() + 2 * k, cuda::std::greater<int>{});
  REQUIRE(keys_out == thrust::device_vector<int>{8, 7, 6, 9, 8, 7});
}

C2H_TEST("cub::DeviceBatchedTopK::MinKeys env-alloc example", "[batched_topk][device][env]")
{
  // example-begin batched-topk-min-keys-env
  constexpr int num_segments = 2;
  constexpr int segment_size = 8;
  constexpr int k            = 3;

  auto keys_in  = thrust::device_vector<int>{5, -3, 1, 7, 8, 2, 4, 6, /**/ 0, 9, 3, 2, 1, 8, 7, 4};
  auto keys_out = thrust::device_vector<int>(num_segments * k, thrust::no_init);

  auto d_keys_in =
    cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(keys_in.data())), segment_size);
  auto d_keys_out =
    cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(keys_out.data())), k);

  cuda::stream stream{cuda::devices[0]};
  auto env = cuda::std::execution::env{
    cuda::execution::require(cuda::execution::determinism::not_guaranteed, cuda::execution::output_ordering::unsorted),
    cuda::stream_ref{stream}};

  auto error = cub::DeviceBatchedTopK::MinKeys(
    d_keys_in,
    d_keys_out,
    cuda::args::constant<segment_size>{},
    cuda::args::constant<k>{},
    cuda::args::immediate{cuda::std::int64_t{num_segments}},
    env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceBatchedTopK::MinKeys failed with status: " << error << '\n';
  }
  // example-end batched-topk-min-keys-env

  stream.sync();
  REQUIRE(error == cudaSuccess);
  thrust::sort(keys_out.begin(), keys_out.begin() + k);
  thrust::sort(keys_out.begin() + k, keys_out.begin() + 2 * k);
  REQUIRE(keys_out == thrust::device_vector<int>{-3, 1, 2, 0, 1, 2});
}

C2H_TEST("cub::DeviceBatchedTopK::MaxPairs env-alloc example", "[batched_topk][device][env]")
{
  // example-begin batched-topk-max-pairs-env
  constexpr int num_segments = 2;
  constexpr int segment_size = 8;
  constexpr int k            = 3;

  auto keys_in    = thrust::device_vector<int>{5, -3, 1, 7, 8, 2, 4, 6, /**/ 0, 9, 3, 2, 1, 8, 7, 4};
  auto keys_out   = thrust::device_vector<int>(num_segments * k, thrust::no_init);
  auto values_out = thrust::device_vector<int>(num_segments * k, thrust::no_init);

  auto d_keys_in =
    cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(keys_in.data())), segment_size);
  auto d_keys_out =
    cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(keys_out.data())), k);
  auto d_values_in = cuda::make_constant_iterator(cuda::make_counting_iterator(0));
  auto d_values_out =
    cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(values_out.data())), k);

  cuda::stream stream{cuda::devices[0]};
  auto env = cuda::std::execution::env{
    cuda::execution::require(cuda::execution::determinism::not_guaranteed, cuda::execution::output_ordering::unsorted),
    cuda::stream_ref{stream}};

  auto error = cub::DeviceBatchedTopK::MaxPairs(
    d_keys_in,
    d_keys_out,
    d_values_in,
    d_values_out,
    cuda::args::constant<segment_size>{},
    cuda::args::constant<k>{},
    cuda::args::immediate{cuda::std::int64_t{num_segments}},
    env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceBatchedTopK::MaxPairs failed with status: " << error << '\n';
  }
  // example-end batched-topk-max-pairs-env

  stream.sync();
  REQUIRE(error == cudaSuccess);

  // Each input value is the global item index. Bounds-checked: every returned value indexes back to the input
  // element (within its own segment) whose key was selected.
  thrust::host_vector<int> h_keys_in(keys_in);
  thrust::host_vector<int> h_keys_out(keys_out);
  thrust::host_vector<int> h_values_out(values_out);
  for (int s = 0; s < num_segments; ++s)
  {
    for (int j = 0; j < k; ++j)
    {
      const int idx = s * k + j;
      const int v   = h_values_out[idx];
      REQUIRE(v >= 0);
      REQUIRE(v < segment_size);
      REQUIRE(h_keys_in[s * segment_size + v] == h_keys_out[idx]);
    }
  }

  // The selected keys must be the per-segment top-k (output is unordered; sort each segment descending).
  thrust::sort(keys_out.begin(), keys_out.begin() + k, cuda::std::greater<int>{});
  thrust::sort(keys_out.begin() + k, keys_out.begin() + 2 * k, cuda::std::greater<int>{});
  REQUIRE(keys_out == thrust::device_vector<int>{8, 7, 6, 9, 8, 7});
}

C2H_TEST("cub::DeviceBatchedTopK::MinPairs env-alloc example", "[batched_topk][device][env]")
{
  // example-begin batched-topk-min-pairs-env
  constexpr int num_segments = 2;
  constexpr int segment_size = 8;
  constexpr int k            = 3;

  auto keys_in    = thrust::device_vector<int>{5, -3, 1, 7, 8, 2, 4, 6, /**/ 0, 9, 3, 2, 1, 8, 7, 4};
  auto keys_out   = thrust::device_vector<int>(num_segments * k, thrust::no_init);
  auto values_out = thrust::device_vector<int>(num_segments * k, thrust::no_init);

  auto d_keys_in =
    cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(keys_in.data())), segment_size);
  auto d_keys_out =
    cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(keys_out.data())), k);
  auto d_values_in = cuda::make_constant_iterator(cuda::make_counting_iterator(0));
  auto d_values_out =
    cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(values_out.data())), k);

  cuda::stream stream{cuda::devices[0]};
  auto env = cuda::std::execution::env{
    cuda::execution::require(cuda::execution::determinism::not_guaranteed, cuda::execution::output_ordering::unsorted),
    cuda::stream_ref{stream}};

  auto error = cub::DeviceBatchedTopK::MinPairs(
    d_keys_in,
    d_keys_out,
    d_values_in,
    d_values_out,
    cuda::args::constant<segment_size>{},
    cuda::args::constant<k>{},
    cuda::args::immediate{cuda::std::int64_t{num_segments}},
    env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceBatchedTopK::MinPairs failed with status: " << error << '\n';
  }
  // example-end batched-topk-min-pairs-env

  stream.sync();
  REQUIRE(error == cudaSuccess);

  // Each input value is the global item index. Bounds-checked: every returned value indexes back to the input
  // element (within its own segment) whose key was selected.
  thrust::host_vector<int> h_keys_in(keys_in);
  thrust::host_vector<int> h_keys_out(keys_out);
  thrust::host_vector<int> h_values_out(values_out);
  for (int s = 0; s < num_segments; ++s)
  {
    for (int j = 0; j < k; ++j)
    {
      const int idx = s * k + j;
      const int v   = h_values_out[idx];
      REQUIRE(v >= 0);
      REQUIRE(v < segment_size);
      REQUIRE(h_keys_in[s * segment_size + v] == h_keys_out[idx]);
    }
  }

  // The selected keys must be the per-segment top-k (output is unordered; sort each segment ascending).
  thrust::sort(keys_out.begin(), keys_out.begin() + k);
  thrust::sort(keys_out.begin() + k, keys_out.begin() + 2 * k);
  REQUIRE(keys_out == thrust::device_vector<int>{-3, 1, 2, 0, 1, 2});
}
