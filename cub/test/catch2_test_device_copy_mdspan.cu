// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_copy.cuh>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>

#include <cuda/std/mdspan>

#include <c2h/catch2_test_helper.h>
#include <catch2_test_launch_helper.h>

// %PARAM% TEST_LAUNCH lid 0:1:2

DECLARE_LAUNCH_WRAPPER(cub::DeviceCopy::Copy, device_copy_mdspan);

C2H_TEST("DeviceCopy::Copy: Buffer size, empty mdspan", "[copy][mdspan]")
{
  // Test empty mdspan
  auto mdspan_in_empty  = cuda::std::mdspan<int, cuda::std::dims<1>>(nullptr, 0);
  auto mdspan_out_empty = cuda::std::mdspan<int, cuda::std::dims<1>>(nullptr, 0);
  device_copy_mdspan(mdspan_in_empty, mdspan_out_empty);
}

C2H_TEST("DeviceCopy::Copy works with 1D mdspan for int", "[copy][mdspan]")
{
  constexpr size_t num_items = 10000;
  // Create input and output device vectors
  c2h::device_vector<int> d_input(num_items);
  c2h::device_vector<int> d_output(num_items);
  // Initialize input data with sequence
  thrust::sequence(d_input.begin(), d_input.end(), 0);
  thrust::fill(d_output.begin(), d_output.end(), 42);

  auto mdspan_in1  = cuda::std::mdspan{thrust::raw_pointer_cast(d_input.data()), num_items};
  auto mdspan_out1 = cuda::std::mdspan{thrust::raw_pointer_cast(d_output.data()), num_items};
  device_copy_mdspan(mdspan_in1, mdspan_out1);
  REQUIRE(d_input == d_output);
  thrust::fill(d_output.begin(), d_output.end(), 42);

  auto mdspan_in2  = cuda::std::mdspan(thrust::raw_pointer_cast(d_input.data()), cuda::std::dims<2>{100, 100});
  auto mdspan_out2 = cuda::std::mdspan(thrust::raw_pointer_cast(d_output.data()), cuda::std::dims<2>{100, 100});
  device_copy_mdspan(mdspan_in2, mdspan_out2);
  REQUIRE(d_input == d_output);
  thrust::fill(d_output.begin(), d_output.end(), 42);

  auto mdspan_in3  = cuda::std::mdspan(thrust::raw_pointer_cast(d_input.data()), cuda::std::dims<4>{10, 10, 10, 10});
  auto mdspan_out3 = cuda::std::mdspan(thrust::raw_pointer_cast(d_output.data()), cuda::std::dims<4>{10, 10, 10, 10});
  device_copy_mdspan(mdspan_in3, mdspan_out3);
  REQUIRE(d_input == d_output);
}
