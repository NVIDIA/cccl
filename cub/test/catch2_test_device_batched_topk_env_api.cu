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

#include "cub_test_macros.h"

void check_max_keys_env_output_padding();
void check_min_pairs_env_output_padding();

CUB_TEST("cub::DeviceBatchedTopK::MaxKeys env-alloc example", "[batched_topk][device][env]", CUB_SMALL)
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

  SECTION("the existing env overload composes the fixed-width output-padding property")
  {
    check_max_keys_env_output_padding();
  }
}

CUB_TEST("cub::DeviceBatchedTopK::MinKeys env-alloc example", "[batched_topk][device][env]", CUB_SMALL)
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

CUB_TEST("cub::DeviceBatchedTopK::MaxPairs env-alloc example", "[batched_topk][device][env]", CUB_SMALL)
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

CUB_TEST("cub::DeviceBatchedTopK::MinPairs env-alloc example", "[batched_topk][device][env]", CUB_SMALL)
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

  SECTION("the existing env overload composes key and value output-padding properties")
  {
    check_min_pairs_env_output_padding();
  }
}

void check_max_keys_env_output_padding()
{
  using seg_size_t           = cuda::std::int16_t;
  constexpr int num_segments = 2;
  constexpr int input_stride = 4;
  constexpr int k            = 4;
  constexpr int key_pad      = -1;
  constexpr int sentinel     = -12345;

  auto keys_in       = thrust::device_vector<int>{5, -3, 1, 7, /**/ 0, 9, 3, 2};
  auto segment_sizes = thrust::device_vector<seg_size_t>{1, 3};
  auto keys_out      = thrust::device_vector<int>(num_segments * k, sentinel);

  auto d_keys_in =
    cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(keys_in.data())), input_stride);
  auto d_keys_out =
    cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(keys_out.data())), k);

  auto sizes_arg = cuda::args::deferred_sequence{
    thrust::raw_pointer_cast(segment_sizes.data()), cuda::args::bounds<seg_size_t{0}, seg_size_t{4}>()};
  constexpr auto k_arg = cuda::args::constant<k>{};
  auto num_segs        = cuda::args::immediate{cuda::std::int64_t{num_segments}};

  cuda::stream stream{cuda::devices[0]};
  auto requirements = cuda::execution::require(
    cuda::execution::determinism::not_guaranteed,
    cuda::execution::tie_break::unspecified,
    cuda::execution::output_ordering::unsorted);
  auto env =
    cuda::std::execution::env{requirements, cub::DeviceBatchedTopK::OutputPadding(key_pad), cuda::stream_ref{stream}};

  REQUIRE(cudaSuccess == cub::DeviceBatchedTopK::MaxKeys(d_keys_in, d_keys_out, sizes_arg, k_arg, num_segs, env));
  stream.sync();

  thrust::sort(keys_out.begin(), keys_out.begin() + 1);
  thrust::sort(keys_out.begin() + k, keys_out.begin() + k + 3);
  REQUIRE(keys_out == thrust::device_vector<int>{5, -1, -1, -1, /**/ 0, 3, 9, -1});
}

void check_min_pairs_env_output_padding()
{
  using seg_size_t           = cuda::std::int16_t;
  constexpr int num_segments = 2;
  constexpr int input_stride = 4;
  constexpr int k            = 4;
  constexpr int key_pad      = -1;
  constexpr int value_pad    = -2;

  auto keys_in       = thrust::device_vector<int>{5, -3, 1, 7, /**/ 0, 9, 3, 2};
  auto segment_sizes = thrust::device_vector<seg_size_t>{1, 3};
  auto keys_out      = thrust::device_vector<int>(num_segments * k, /*canary=*/-12345);
  auto values_out    = thrust::device_vector<int>(num_segments * k, /*canary=*/-23456);

  auto d_keys_in =
    cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(keys_in.data())), input_stride);
  auto d_keys_out =
    cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(keys_out.data())), k);
  auto d_values_in = cuda::make_constant_iterator(cuda::make_counting_iterator(0));
  auto d_values_out =
    cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(values_out.data())), k);

  auto sizes_arg = cuda::args::deferred_sequence{
    thrust::raw_pointer_cast(segment_sizes.data()), cuda::args::bounds<seg_size_t{0}, seg_size_t{4}>()};
  constexpr auto k_arg = cuda::args::constant<k>{};
  auto num_segs        = cuda::args::immediate{cuda::std::int64_t{num_segments}};

  cuda::stream stream{cuda::devices[0]};
  auto requirements = cuda::execution::require(
    cuda::execution::determinism::not_guaranteed,
    cuda::execution::tie_break::unspecified,
    cuda::execution::output_ordering::unsorted);
  auto env = cuda::std::execution::env{
    requirements, cub::DeviceBatchedTopK::OutputPadding(key_pad, value_pad), cuda::stream_ref{stream}};

  REQUIRE(cudaSuccess
          == cub::DeviceBatchedTopK::MinPairs(
            d_keys_in, d_keys_out, d_values_in, d_values_out, sizes_arg, k_arg, num_segs, env));
  stream.sync();

  thrust::host_vector<int> h_keys_in    = keys_in;
  thrust::host_vector<int> h_keys_out   = keys_out;
  thrust::host_vector<int> h_values_out = values_out;
  for (int segment = 0; segment < num_segments; ++segment)
  {
    const int valid       = segment == 0 ? 1 : 3;
    const int output_base = segment * k;
    const int input_base  = segment * input_stride;
    for (int item = 0; item < valid; ++item)
    {
      const int value = h_values_out[output_base + item];
      REQUIRE(value >= 0);
      REQUIRE(value < valid);
      REQUIRE(h_keys_out[output_base + item] == h_keys_in[input_base + value]);
    }
    for (int item = valid; item < k; ++item)
    {
      REQUIRE(h_keys_out[output_base + item] == key_pad);
      REQUIRE(h_values_out[output_base + item] == value_pad);
    }
  }

  thrust::sort(keys_out.begin(), keys_out.begin() + 1);
  thrust::sort(keys_out.begin() + k, keys_out.begin() + k + 3);
  REQUIRE(keys_out == thrust::device_vector<int>{5, -1, -1, -1, /**/ 0, 3, 9, -1});
}
