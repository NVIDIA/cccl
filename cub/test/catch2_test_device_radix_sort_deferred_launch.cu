// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_radix_sort.cuh>

#include <cuda/argument>
#include <cuda/iterator>
#include <cuda/std/span>

#include <cstdint>

#include "catch2_radix_sort_helper.cuh"
#include "catch2_test_launch_helper.h"
#include "cub_test_macros.h"
#include <c2h/generators.h>

// %PARAM% TEST_LAUNCH lid 0:1:2

DECLARE_LAUNCH_WRAPPER(cub::DeviceRadixSort::SortKeys, sort_keys);

using deferred_radix_count_types = c2h::type_list<int32_t, uint32_t, int64_t, uint64_t>;

CUB_TEST("deferred radix count sources support spans and fancy iterators",
         "[device][radix_sort][deferred]",
         CUB_SMALL,
         deferred_radix_count_types)
{
  using count_t   = c2h::get<0, TestType>;
  constexpr int n = 19'321;
  c2h::device_vector<uint32_t> input(n, thrust::no_init), output(n, thrust::no_init);
  c2h::gen(C2H_SEED(2), input);
  const c2h::host_vector<uint32_t> original = input;
  const auto expected                       = radix_sort_reference(input, false);
  c2h::device_vector<count_t> device_count(1, static_cast<count_t>(n));
  auto test_source = [&](auto source) {
    const auto count = cuda::args::deferred{source, cuda::args::bounds<0, n>()};
    sort_keys(input.data().get(), output.data().get(), count, 0, 32);
    REQUIRE(expected == output);
    REQUIRE(original == input);
  };
  test_source(device_count.data().get());
  test_source(cuda::std::span<const count_t, 1>{device_count.data().get(), 1});
  test_source(cuda::transform_iterator{device_count.data().get(), cuda::std::identity{}});
}
