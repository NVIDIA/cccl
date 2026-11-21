// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_reduce.cuh>
#include <cub/util_device.cuh>

#include <cuda/std/cstdint>
#include <cuda/std/functional>

#include <cstddef>
#include <numeric>

#include "catch2_test_launch_helper.h"
#include "cub_test_macros.h"
#include <c2h/catch2_test_helper.h>

// %PARAM% TEST_LAUNCH lid 0:1:2
DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::Sum, small_sum);
DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::Reduce, small_reduce);

using offset_types = c2h::type_list<cuda::std::int32_t, cuda::std::int64_t>;

CUB_TEST("Device reduce sums across single-tile boundaries", "[reduce][device]", CUB_SMALL, offset_types)
{
  using offset_t       = c2h::get<0, TestType>;
  using value_t        = cuda::std::uint32_t;
  const offset_t count = GENERATE(0, 1, 31, 33, 129, 1025, 4095, 4096, 4097, 6143, 6144, 8191, 8192, 8193, 16385);
  const int offset     = GENERATE(0, 1, 2, 3);
  c2h::device_vector<value_t> input(count + offset + 1);
  c2h::gen(C2H_SEED(2), input);
  const c2h::host_vector<value_t> host_input = input;
  c2h::device_vector<value_t> output(1);
  auto* d_input          = thrust::raw_pointer_cast(input.data()) + offset;
  auto* d_output         = thrust::raw_pointer_cast(output.data());
  const value_t expected = std::accumulate(host_input.begin() + offset, host_input.begin() + offset + count, value_t{});

  small_sum(d_input, d_output, count);
  REQUIRE(output[0] == expected);

  constexpr value_t init = 0xfedcba98;
  small_reduce(d_input, d_output, count, cuda::std::plus<>{}, init);
  REQUIRE(output[0] == value_t(expected + init));
}

#if TEST_LAUNCH == 0
struct unchanged_sum_policy
    : cub::detail::reduce::policy_selector_from_types<cuda::std::uint32_t, int, cuda::std::plus<>>
{};

CUB_TEST("Device reduce small-sum tuning is bounded to Thor", "[reduce][device]", CUB_SMALL, offset_types)
{
  using offset_t = c2h::get<0, TestType>;
  using base_t   = cub::detail::reduce::policy_selector_from_types<cuda::std::uint32_t, offset_t, cuda::std::plus<>>;
  using small_t  = cub::detail::reduce::sm110_small_sum_policy_selector<offset_t>;
  for (const auto cc :
       {cuda::compute_capability{8, 0},
        cuda::compute_capability{9, 0},
        cuda::compute_capability{10, 0},
        cuda::compute_capability{10, 3},
        cuda::compute_capability{10, 7},
        cuda::compute_capability{12, 0}})
  {
    REQUIRE(small_t{}(cc) == base_t{}(cc));
  }
  constexpr cuda::compute_capability thor{11, 0};
  STATIC_REQUIRE(small_t{}(thor).single_tile.threads_per_block == 256);
  STATIC_REQUIRE(small_t{}(thor).single_tile.items_per_thread == 32);
  STATIC_REQUIRE(small_t{}(thor).multi_tile == base_t{}(thor).multi_tile);
}

CUB_TEST("Device reduce preserves custom selectors in the added interval", "[reduce][device]", CUB_SMALL)
{
  using value_t       = cuda::std::uint32_t;
  constexpr int count = 6144;
  c2h::device_vector<value_t> input(count, value_t{1});
  c2h::device_vector<value_t> output(1);
  auto* d_input             = thrust::raw_pointer_cast(input.data());
  auto* d_output            = thrust::raw_pointer_cast(output.data());
  std::size_t default_bytes = 0;
  std::size_t custom_bytes  = 0;
  REQUIRE_CUDART(cub::DeviceReduce::Sum(nullptr, default_bytes, d_input, d_output, count));
  REQUIRE_CUDART(cub::detail::reduce::dispatch(
    nullptr,
    custom_bytes,
    d_input,
    d_output,
    count,
    cuda::std::plus<>{},
    value_t{},
    nullptr,
    cuda::std::identity{},
    unchanged_sum_policy{}));
  cuda::compute_capability cc{};
  REQUIRE_CUDART(cub::detail::ptx_compute_cap(cc));
  if (cc == cuda::compute_capability{11, 0})
  {
    REQUIRE(default_bytes == 1);
    REQUIRE(custom_bytes > default_bytes);
  }
  c2h::device_vector<char> storage(custom_bytes);
  REQUIRE_CUDART(cub::detail::reduce::dispatch(
    storage.data().get(),
    custom_bytes,
    d_input,
    d_output,
    count,
    cuda::std::plus<>{},
    value_t{},
    nullptr,
    cuda::std::identity{},
    unchanged_sum_policy{}));
  REQUIRE(output[0] == value_t(count));
}
#endif // TEST_LAUNCH == 0
