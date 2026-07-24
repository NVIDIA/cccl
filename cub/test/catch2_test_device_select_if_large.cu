// SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

// Large-memory DeviceSelect::If test, split out from catch2_test_device_select_if.cu so it can run
// serially (tagged [large-mem]) without serializing the small tests in that file.

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_select.cuh>
#include <cub/device/dispatch/dispatch_select_if.cuh>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/equal.h>

#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/limits>

#include <cstdint>
#include <new> // bad_alloc

#include "catch2_test_device_select_common.cuh"
#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.h>

DECLARE_LAUNCH_WRAPPER(cub::DeviceSelect::If, select_if);

// %PARAM% TEST_LAUNCH lid 0:1:2

C2H_TEST("DeviceSelect::If works for very large number of output items",
         "[large-mem][device][select_if][skip-cs-initcheck][skip-cs-racecheck][skip-cs-synccheck]")
try
{
  using type     = std::uint8_t;
  using offset_t = std::int64_t;

  // The partition size (the maximum number of items processed by a single kernel invocation) is an important boundary
  constexpr auto max_partition_size = static_cast<offset_t>(cuda::std::numeric_limits<std::int32_t>::max());

  offset_t num_items = GENERATE_COPY(
    values({
      offset_t{2} * max_partition_size + offset_t{20000000}, // 3 partitions
      offset_t{2} * max_partition_size, // 2 partitions
      max_partition_size + offset_t{1}, // 2 partitions
      max_partition_size, // 1 or 2 partitions
      max_partition_size - offset_t{745}, // 1 or 2 partitions
      max_partition_size - offset_t{10745} // 1 partition
    }),
    take(2, random(max_partition_size - offset_t{100000}, max_partition_size + offset_t{100000})));

  CAPTURE(num_items);

  // Prepare input iterator: it[i] = (i%mod)+(i/div)
  static constexpr offset_t mod = 200;
  static constexpr offset_t div = 1000000000;
  auto in = cuda::transform_iterator(cuda::counting_iterator(offset_t{0}), modx_and_add_divy<offset_t, type>{mod, div});

  // Prepare output
  c2h::device_vector<type> out(num_items);

  // Needs to be device accessible
  c2h::device_vector<offset_t> num_selected_out(1, 0);
  offset_t* d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  // Run test
  select_if(in, out.begin(), d_first_num_selected_out, num_items, cuda::always_true{});

  // Ensure that we created the correct output
  REQUIRE(num_selected_out[0] == num_items);
  bool all_results_correct = thrust::equal(out.cbegin(), out.cend(), in);
  REQUIRE(all_results_correct == true);
}
catch (std::bad_alloc&)
{
  // Exceeding memory is not a failure.
  SUCCEED("exceeding memory is not a failure");
}
