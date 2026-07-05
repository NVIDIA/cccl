// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Defer the unsupported-architecture diagnosis to the dispatch's runtime check (not a compile-time static_assert)
// so this test compiles across all target architectures, including pre-SM90, for the full configuration space. See
// _CUB_DISABLE_TOPK_UNSUPPORTED_ARCH_ASSERT in cub/device/dispatch/dispatch_batched_topk.cuh. Precedes CUB includes.
#define _CUB_DISABLE_TOPK_UNSUPPORTED_ARCH_ASSERT

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
    cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                             cuda::execution::tie_break::unspecified,
                             cuda::execution::output_ordering::unsorted),
    cuda::stream_ref{stream}};

  // The env-based overload allocates and frees the temporary storage internally.
  auto error = cub::DeviceBatchedTopK::MaxKeys(
    d_keys_in,
    d_keys_out,
    cuda::args::constant<segment_size>{},
    cuda::args::constant<k>{},
    cuda::args::immediate{cuda::std::int64_t{num_segments}},
    env);
  // Each segment's k largest keys are written to keys_out in unspecified order. The result set is fixed,
  // shown here sorted per segment:
  auto expected_result_set = thrust::device_vector<int>{8, 7, 6, /* segment 0 */ 9, 8, 7 /* segment 1 */};
  // example-end batched-topk-max-keys-env

  stream.sync();
  REQUIRE(error == cudaSuccess);
  // keys_out is unordered, so sort each segment (descending) before comparing against the expected set.
  thrust::sort(keys_out.begin(), keys_out.begin() + k, cuda::std::greater<int>{});
  thrust::sort(keys_out.begin() + k, keys_out.begin() + 2 * k, cuda::std::greater<int>{});
  REQUIRE(keys_out == expected_result_set);
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
    cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                             cuda::execution::tie_break::unspecified,
                             cuda::execution::output_ordering::unsorted),
    cuda::stream_ref{stream}};

  auto error = cub::DeviceBatchedTopK::MinKeys(
    d_keys_in,
    d_keys_out,
    cuda::args::constant<segment_size>{},
    cuda::args::constant<k>{},
    cuda::args::immediate{cuda::std::int64_t{num_segments}},
    env);
  // Each segment's k smallest keys are written to keys_out in unspecified order. The result set is fixed,
  // shown here sorted per segment:
  auto expected_result_set = thrust::device_vector<int>{-3, 1, 2, /* segment 0 */ 0, 1, 2 /* segment 1 */};
  // example-end batched-topk-min-keys-env

  stream.sync();
  REQUIRE(error == cudaSuccess);
  // keys_out is unordered, so sort each segment (ascending) before comparing against the expected set.
  thrust::sort(keys_out.begin(), keys_out.begin() + k);
  thrust::sort(keys_out.begin() + k, keys_out.begin() + 2 * k);
  REQUIRE(keys_out == expected_result_set);
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
    cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                             cuda::execution::tie_break::unspecified,
                             cuda::execution::output_ordering::unsorted),
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
  // keys_out holds each segment's k largest keys. The key set is fixed (shown here sorted per segment). For
  // keys that tie, which equal element's value is returned is unspecified.
  auto expected_result_set = thrust::device_vector<int>{8, 7, 6, /* segment 0 */ 9, 8, 7 /* segment 1 */};
  // example-end batched-topk-max-pairs-env

  stream.sync();
  REQUIRE(error == cudaSuccess);

  // Each returned value is the source index of its key within the segment. Check that every value indexes
  // back to the input element whose key was selected.
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

  // keys_out is unordered, so sort each segment (descending) before comparing against the expected set.
  thrust::sort(keys_out.begin(), keys_out.begin() + k, cuda::std::greater<int>{});
  thrust::sort(keys_out.begin() + k, keys_out.begin() + 2 * k, cuda::std::greater<int>{});
  REQUIRE(keys_out == expected_result_set);
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
    cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                             cuda::execution::tie_break::unspecified,
                             cuda::execution::output_ordering::unsorted),
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
  // keys_out holds each segment's k smallest keys. The key set is fixed (shown here sorted per segment). For
  // keys that tie, which equal element's value is returned is unspecified.
  auto expected_result_set = thrust::device_vector<int>{-3, 1, 2, /* segment 0 */ 0, 1, 2 /* segment 1 */};
  // example-end batched-topk-min-pairs-env

  stream.sync();
  REQUIRE(error == cudaSuccess);

  // Each returned value is the source index of its key within the segment. Check that every value indexes
  // back to the input element whose key was selected.
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

  // keys_out is unordered, so sort each segment (ascending) before comparing against the expected set.
  thrust::sort(keys_out.begin(), keys_out.begin() + k);
  thrust::sort(keys_out.begin() + k, keys_out.begin() + 2 * k);
  REQUIRE(keys_out == expected_result_set);
}
