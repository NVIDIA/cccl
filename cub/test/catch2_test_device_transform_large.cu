// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

// Large-memory DeviceTransform test, split out from catch2_test_device_transform.cu so it can run
// serially (tagged [large-mem]) without serializing the small tests in that file.

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_transform.cuh>

#include <cuda/std/tuple>

#include <algorithm>
#include <new> // bad_alloc

#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.h>

// %PARAM% TEST_LAUNCH lid 0:1:2

DECLARE_LAUNCH_WRAPPER(cub::DeviceTransform::Transform, transform_many);

struct times_seven
{
  template <typename T>
  __host__ __device__ auto operator()(T v) const -> T
  {
    return static_cast<T>(v * 7);
  }
};

C2H_TEST("DeviceTransform::Transform works with large input",
         "[large-mem][device][transform][skip-cs-initcheck][skip-cs-racecheck][skip-cs-synccheck]")
try
{
  using type     = std::uint32_t;
  using offset_t = cuda::std::int64_t;

  const auto delta         = GENERATE(-123456, 123456);
  const offset_t num_items = static_cast<offset_t>((offset_t{1} << 30) + delta);
  CAPTURE(c2h::type_name<offset_t>(), num_items);

  c2h::device_vector<type> input(static_cast<size_t>(num_items), thrust::no_init);
  c2h::gen(C2H_SEED(1), input);

  c2h::device_vector<type> result(static_cast<size_t>(num_items), thrust::no_init);
  transform_many(cuda::std::make_tuple(input.begin()), result.begin(), num_items, times_seven{});

  // compute reference and verify
  c2h::host_vector<type> input_h = input;
  c2h::host_vector<type> reference_h(static_cast<size_t>(num_items), thrust::no_init);
  std::transform(input_h.begin(), input_h.end(), reference_h.begin(), times_seven{});
  REQUIRE((reference_h == result));
}
catch (const std::bad_alloc&)
{
  // allocation failure is not a test failure, so we can run tests on smaller GPUs
  SUCCEED("allocation failure is not a test failure");
}
