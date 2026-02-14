// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_copy.cuh>

#include <thrust/count.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

#include <cuda/std/array>
#include <cuda/std/mdspan>

#include <c2h/catch2_test_helper.h>
#include <catch2_test_launch_helper.h>

// %PARAM% TEST_LAUNCH lid 0:1:2

DECLARE_LAUNCH_WRAPPER(cub::DeviceCopy::Copy, device_copy_mdspan);

using dims_1d_t = cuda::std::dims<1, int>;
using dims_2d_t = cuda::std::dims<2, int>;
using dims_4d_t = cuda::std::dims<4, int>;

C2H_TEST("DeviceCopy::Copy: empty mdspan", "[copy][mdspan]")
{
  auto mdspan_in_empty  = cuda::std::mdspan<int, dims_1d_t>(nullptr, 0);
  auto mdspan_out_empty = cuda::std::mdspan<int, dims_1d_t>(nullptr, 0);
  device_copy_mdspan(mdspan_in_empty, mdspan_out_empty);
}

C2H_TEST("DeviceCopy::Copy: 1D, 2D, 4D mdspan with matching layouts", "[copy][mdspan]")
{
  constexpr size_t num_items = 10000;
  c2h::device_vector<int> d_input(num_items, thrust::no_init);
  c2h::device_vector<int> d_output(num_items, thrust::no_init);
  thrust::sequence(d_input.begin(), d_input.end(), 0);
  thrust::fill(d_output.begin(), d_output.end(), 42);

  auto mdspan_in1  = cuda::std::mdspan{thrust::raw_pointer_cast(d_input.data()), dims_1d_t{num_items}};
  auto mdspan_out1 = cuda::std::mdspan{thrust::raw_pointer_cast(d_output.data()), dims_1d_t{num_items}};
  device_copy_mdspan(mdspan_in1, mdspan_out1);
  REQUIRE(d_input == d_output);
  thrust::fill(d_output.begin(), d_output.end(), 42);

  using mdspan_2d_left_t = cuda::std::mdspan<int, dims_2d_t, cuda::std::layout_left>;
  auto d_mdspan_in2      = mdspan_2d_left_t(thrust::raw_pointer_cast(d_input.data()), dims_2d_t{100, 100});
  auto d_mdspan_out2     = mdspan_2d_left_t(thrust::raw_pointer_cast(d_output.data()), dims_2d_t{100, 100});
  device_copy_mdspan(d_mdspan_in2, d_mdspan_out2);
  REQUIRE(d_input == d_output);
  thrust::fill(d_output.begin(), d_output.end(), 42);

  auto mdspan_in3  = cuda::std::mdspan(thrust::raw_pointer_cast(d_input.data()), dims_4d_t{10, 10, 10, 10});
  auto mdspan_out3 = cuda::std::mdspan(thrust::raw_pointer_cast(d_output.data()), dims_4d_t{10, 10, 10, 10});
  device_copy_mdspan(mdspan_in3, mdspan_out3);
  REQUIRE(d_input == d_output);
}

C2H_TEST("DeviceCopy::Copy: 2D, 4D mdspan with compatible layouts", "[copy][mdspan]")
{
  constexpr size_t num_items = 10000;
  c2h::device_vector<int> d_input(num_items, thrust::no_init);
  c2h::device_vector<int> d_output(num_items, thrust::no_init);
  thrust::sequence(d_input.begin(), d_input.end(), 0);
  thrust::fill(d_output.begin(), d_output.end(), 42);

  using cuda::std::layout_left;
  using cuda::std::layout_stride;
  using mdspan_2d_left_t  = cuda::std::mdspan<int, dims_2d_t, layout_left>;
  using mdspan_strided_2d = cuda::std::mdspan<int, dims_2d_t, layout_stride>;
  layout_stride::mapping<dims_2d_t> map_out{dims_2d_t{100, 100}, cuda::std::array{1, 100}};

  auto d_mdspan_in2  = mdspan_2d_left_t(thrust::raw_pointer_cast(d_input.data()), dims_2d_t{100, 100});
  auto d_mdspan_out2 = mdspan_strided_2d(thrust::raw_pointer_cast(d_output.data()), map_out);
  device_copy_mdspan(d_mdspan_in2, d_mdspan_out2);
  REQUIRE(d_input == d_output);
}

struct is_42
{
  __host__ __device__ bool operator()(int x) const
  {
    return x == 42;
  }
};

C2H_TEST("DeviceCopy::Copy: 2D strided mdspan", "[copy][mdspan]")
{
  constexpr size_t num_items = (2 * 100 + 20) * 100;
  c2h::device_vector<int> d_input(num_items, thrust::no_init);
  c2h::device_vector<int> d_output(num_items, thrust::no_init);
  thrust::sequence(d_input.begin(), d_input.end(), 0);
  thrust::fill(d_output.begin(), d_output.end(), 42);
  using cuda::std::layout_stride;
  using mdspan_strided_2d = cuda::std::mdspan<int, dims_2d_t, layout_stride>;

  layout_stride::mapping<dims_2d_t> map_in{dims_2d_t{100, 100}, cuda::std::array{2, 220}};
  layout_stride::mapping<dims_2d_t> map_out{dims_2d_t{100, 100}, cuda::std::array{220, 2}};
  auto d_mdspan_in  = mdspan_strided_2d(thrust::raw_pointer_cast(d_input.data()), map_in);
  auto d_mdspan_out = mdspan_strided_2d(thrust::raw_pointer_cast(d_output.data()), map_out);
  device_copy_mdspan(d_mdspan_in, d_mdspan_out);

  c2h::host_vector<int> h_input  = d_input;
  c2h::host_vector<int> h_output = d_output;
  auto h_mdspan_in               = mdspan_strided_2d(thrust::raw_pointer_cast(h_input.data()), map_in);
  auto h_mdspan_out              = mdspan_strided_2d(thrust::raw_pointer_cast(h_output.data()), map_out);
  for (size_t i = 0; i < 100; i++)
  {
    for (size_t j = 0; j < 100; j++)
    {
      REQUIRE(h_mdspan_in(i, j) == h_mdspan_out(i, j));
    }
  }
  // Count elements that weren't overwritten (should remain as 42s)
  // Due to strided layout, not all elements in the contiguous buffer are accessed
  auto count = thrust::count_if(d_output.begin(), d_output.end(), thrust::detail::equal_to_value<int>{42});
  constexpr size_t expected_untouched = num_items - (100 * 100 - 1);
  REQUIRE(count == expected_untouched);
}

C2H_TEST("DeviceCopy::Copy: 2D strided mdspan + contiguous mdspan", "[copy][mdspan]")
{
  constexpr size_t num_items = (2 * 100 + 20) * 100;
  c2h::device_vector<int> d_input(num_items, thrust::no_init);
  c2h::device_vector<int> d_output(num_items, thrust::no_init);
  thrust::sequence(d_input.begin(), d_input.end(), 0);
  thrust::fill(d_output.begin(), d_output.end(), 42);
  using cuda::std::layout_stride;
  using mdspan_strided_2d    = cuda::std::mdspan<int, dims_2d_t, layout_stride>;
  using mdspan_contiguous_2d = cuda::std::mdspan<int, dims_2d_t>;
  layout_stride::mapping<dims_2d_t> map_in{dims_2d_t{100, 100}, cuda::std::array{2, 220}};
  auto d_mdspan_in  = mdspan_strided_2d(thrust::raw_pointer_cast(d_input.data()), map_in);
  auto d_mdspan_out = mdspan_contiguous_2d(thrust::raw_pointer_cast(d_output.data()), dims_2d_t{100, 100});
  device_copy_mdspan(d_mdspan_in, d_mdspan_out);

  c2h::host_vector<int> h_input  = d_input;
  c2h::host_vector<int> h_output = d_output;
  auto h_mdspan_in               = mdspan_strided_2d(thrust::raw_pointer_cast(h_input.data()), map_in);
  auto h_mdspan_out              = mdspan_contiguous_2d(thrust::raw_pointer_cast(h_output.data()), dims_2d_t{100, 100});
  for (size_t i = 0; i < 100; i++)
  {
    for (size_t j = 0; j < 100; j++)
    {
      REQUIRE(h_mdspan_in(i, j) == h_mdspan_out(i, j));
    }
  }
}
