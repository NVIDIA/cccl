// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Large-memory DeviceMemcpy::Batched test, split out from catch2_test_device_memcpy_batched.cu so
// it can be run serially (tagged [large-mem]) without serializing the small tests in that file.

#include <cub/device/device_memcpy.cuh>
#include <cub/util_macro.cuh>

#include <thrust/copy.h>
#include <thrust/detail/raw_pointer_cast.h>

#include <cuda/iterator>

#include <cstdint>
#include <iostream>
#include <limits>
#include <new> // bad_alloc

#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.h>

// %PARAM% TEST_LAUNCH lid 0:1:2

DECLARE_LAUNCH_WRAPPER(cub::DeviceMemcpy::Batched, memcpy_batched);

C2H_TEST("DeviceMemcpy::Batched works for a very large buffer",
         "[large-mem][memcpy][skip-cs-initcheck][skip-cs-racecheck][skip-cs-synccheck]")
try
{
  using data_t        = uint64_t;
  using byte_offset_t = uint64_t;

  byte_offset_t large_target_copy_size = static_cast<byte_offset_t>(std::numeric_limits<uint32_t>::max()) + (32 << 20);
  constexpr auto data_type_size        = static_cast<byte_offset_t>(sizeof(data_t));
  byte_offset_t num_items              = large_target_copy_size / data_type_size;
  byte_offset_t num_bytes              = num_items * data_type_size;
  c2h::device_vector<data_t> d_in(num_items);
  c2h::device_vector<data_t> d_out(num_items, 42);

  auto input_data_it = cuda::counting_iterator(data_t{42});
  thrust::copy(input_data_it, input_data_it + num_items, d_in.begin());

  const auto num_buffers = 1;
  auto d_buffer_srcs     = cuda::constant_iterator(static_cast<void*>(thrust::raw_pointer_cast(d_in.data())));
  auto d_buffer_dsts     = cuda::constant_iterator(static_cast<void*>(thrust::raw_pointer_cast(d_out.data())));
  auto d_buffer_sizes    = cuda::constant_iterator(num_bytes);
  memcpy_batched(d_buffer_srcs, d_buffer_dsts, d_buffer_sizes, num_buffers);

  const bool all_equal = thrust::equal(d_out.cbegin(), d_out.cend(), input_data_it);
  REQUIRE(all_equal == true);
}
catch (std::bad_alloc& e)
{
  std::cerr << "Caught bad_alloc: " << e.what() << '\n';
}
