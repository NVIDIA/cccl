// SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_reduce.cuh>

#include <cuda/std/limits>

#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.h>

DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::ArgMin, device_arg_min);
DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::ArgMax, device_arg_max);

_CCCL_SUPPRESS_DEPRECATED_PUSH
DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::ArgMin, device_arg_min_old);
DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::ArgMax, device_arg_max_old);
_CCCL_SUPPRESS_DEPRECATED_POP

// %PARAM% TEST_LAUNCH lid 0:1

C2H_TEST("Device reduce arg{min,max} works with inf items", "[reduce][device]")
{
  using in_t     = float;
  using offset_t = int;
  using out_t    = cub::KeyValuePair<offset_t, in_t>;

  constexpr int n     = 10;
  constexpr float inf = cuda::std::numeric_limits<float>::infinity();

  c2h::device_vector<out_t> out(1);
  out_t* d_out          = thrust::raw_pointer_cast(out.data());
  offset_t* d_out_index = &d_out->key;
  in_t* d_out_extremum  = &d_out->value;

  /**
   * ArgMin should return max value for empty input. This interferes with
   * input data containing infinity values. This test checks that ArgMin
   * works correctly with infinity values.
   */
  SECTION("InfInArgMin")
  {
    c2h::device_vector<in_t> in(n, inf);
    const in_t* d_in = thrust::raw_pointer_cast(in.data());

    device_arg_min(d_in, d_out_extremum, d_out_index, n);

    const out_t result = out[0];
    REQUIRE(result.key == 0);
    REQUIRE(result.value == inf);
  }

  /**
   * ArgMax should return lowest value for empty input. This interferes with
   * input data containing infinity values. This test checks that ArgMax
   * works correctly with infinity values.
   */
  SECTION("InfInArgMax")
  {
    c2h::device_vector<in_t> in(n, -inf);
    const in_t* d_in = thrust::raw_pointer_cast(in.data());

    device_arg_max(d_in, d_out_extremum, d_out_index, n);

    const out_t result = out[0];
    REQUIRE(result.key == 0);
    REQUIRE(result.value == -inf);
  }

  /**
   * ArgMin should return max value for empty input. This interferes with
   * input data containing infinity values. This test checks that ArgMin
   * works correctly with infinity values.
   */
  SECTION("InfInArgMin deprecated interface")
  {
    c2h::device_vector<in_t> in(n, inf);
    const in_t* d_in = thrust::raw_pointer_cast(in.data());

    device_arg_min_old(d_in, d_out, n);

    const out_t result = out[0];
    REQUIRE(result.key == 0);
    REQUIRE(result.value == inf);
  }

  /**
   * ArgMax should return lowest value for empty input. This interferes with
   * input data containing infinity values. This test checks that ArgMax
   * works correctly with infinity values.
   */
  SECTION("InfInArgMax deprecated interface")
  {
    c2h::device_vector<in_t> in(n, -inf);
    const in_t* d_in = thrust::raw_pointer_cast(in.data());

    device_arg_max_old(d_in, d_out, n);

    const out_t result = out[0];
    REQUIRE(result.key == 0);
    REQUIRE(result.value == -inf);
  }
}
