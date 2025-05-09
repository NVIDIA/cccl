// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_memcpy.cuh>
#include <cub/util_macro.cuh>

#include <thrust/copy.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

#include "catch2_test_device_memcpy_batched_common.cuh"
#include "catch2_test_launch_helper.h"
#include "thrust/iterator/transform_iterator.h"
#include <c2h/catch2_test_helper.h>

// %PARAM% TEST_LAUNCH lid 0:1:2

DECLARE_LAUNCH_WRAPPER(cub::DeviceMemcpy::Batched, memcpy_batched);

C2H_TEST("DeviceMemcpy::Batched works", "[memcpy]")
try
{
  using src_ptr_t = const uint8_t*;
  using dst_ptr_t = uint8_t*;

  // Type used for indexing into the array of buffers
  using buffer_offset_t = uint32_t;

  // Type used for indexing into individual bytes of a buffer (large enough to cover the max buffer
  using buffer_size_t = uint32_t;

  // Type used for indexing into bytes over *all* the buffers' sizes
  using byte_offset_t = uint32_t;

  // Total number of bytes that are targeted to be copied on each run
  constexpr buffer_offset_t target_copy_size = 32U << 20;

  // Pairs of [min, max] buffer sizes
  auto [min_buffer_size, max_buffer_size] = GENERATE_COPY(
    table<buffer_size_t, buffer_size_t>(
      {{0, 1},
       {1, 2},
       {0, 32},
       {1, 1024},
       {1, 32 * 1024},
       {128 * 1024, 256 * 1024},
       {target_copy_size, target_copy_size}}),
    // Use c2h::adjust_seed_count to reduce runtime on sanitizers.
    take(c2h::adjust_seed_count(4),
         map(
           [](const std::vector<buffer_size_t>& chunk) {
             buffer_size_t lhs = chunk[0];
             buffer_size_t rhs = chunk[1];
             // Optionally ensure lhs < rhs, for example:
             return (lhs < rhs) ? std::make_tuple(lhs, rhs) : std::make_tuple(rhs, lhs);
           },
           chunk(2, random(buffer_size_t{1}, buffer_size_t{1000000})))));

  const double average_buffer_size = (min_buffer_size + max_buffer_size) / 2.0;
  const auto num_buffers           = static_cast<buffer_offset_t>(target_copy_size / average_buffer_size);

  c2h::device_vector<buffer_size_t> d_buffer_sizes(num_buffers);

  // Generate the buffer sizes: Make sure buffer sizes are a multiple of the most granular unit (one AtomicT) being
  // copied (round down)
  c2h::gen(C2H_SEED(2), d_buffer_sizes, min_buffer_size, max_buffer_size);
  byte_offset_t num_total_bytes = thrust::reduce(d_buffer_sizes.cbegin(), d_buffer_sizes.cend());

  // Shuffle input buffer source-offsets
  auto d_buffer_src_offsets = get_shuffled_buffer_offsets<buffer_offset_t, byte_offset_t>(d_buffer_sizes, C2H_SEED(1));

  // Shuffle output buffer destination-offsets
  auto d_buffer_dst_offsets = get_shuffled_buffer_offsets<buffer_offset_t, byte_offset_t>(d_buffer_sizes, C2H_SEED(1));

  // Generate random input data and initialize output data
  c2h::device_vector<cuda::std::uint8_t> d_in(num_total_bytes);
  c2h::device_vector<cuda::std::uint8_t> d_out(num_total_bytes, 42);
  c2h::gen(C2H_SEED(1), d_in);

  // Prepare host-side input data for verification
  c2h::host_vector<cuda::std::uint8_t> h_in(d_in);
  c2h::host_vector<cuda::std::uint8_t> h_out(num_total_bytes);
  c2h::host_vector<buffer_size_t> h_buffer_sizes(d_buffer_sizes);
  c2h::host_vector<byte_offset_t> h_src_offsets(d_buffer_src_offsets);
  c2h::host_vector<byte_offset_t> h_dst_offsets(d_buffer_dst_offsets);

  // Prepare d_buffer_srcs
  offset_to_ptr_op<src_ptr_t> src_transform_op{thrust::raw_pointer_cast(d_in.data())};
  auto d_buffer_srcs =
    thrust::make_transform_iterator(thrust::raw_pointer_cast(d_buffer_src_offsets.data()), src_transform_op);

  // Prepare d_buffer_dsts
  offset_to_ptr_op<dst_ptr_t> dst_transform_op{thrust::raw_pointer_cast(d_out.data())};
  auto d_buffer_dsts =
    thrust::make_transform_iterator(thrust::raw_pointer_cast(d_buffer_dst_offsets.data()), dst_transform_op);

  // Invoke device-side algorithm
  memcpy_batched(d_buffer_srcs, d_buffer_dsts, d_buffer_sizes.begin(), num_buffers);

  // Prepare CPU-side result for verification
  for (buffer_offset_t i = 0; i < num_buffers; i++)
  {
    std::memcpy(h_out.data() + h_dst_offsets[i], h_in.data() + h_src_offsets[i], h_buffer_sizes[i]);
  }

  REQUIRE(d_out == h_out);
}
catch (std::bad_alloc& e)
{
  std::cerr << "Caught bad_alloc: " << e.what() << std::endl;
}

C2H_TEST("DeviceMemcpy::Batched works for a very large buffer",
         "[memcpy][skip-cs-initcheck][skip-cs-racecheck][skip-cs-synccheck]")
try
{
  using data_t        = uint64_t;
  using byte_offset_t = uint64_t;
  using buffer_size_t = uint64_t;

  byte_offset_t large_target_copy_size = static_cast<byte_offset_t>(std::numeric_limits<uint32_t>::max()) + (32 << 20);
  constexpr auto data_type_size        = static_cast<byte_offset_t>(sizeof(data_t));
  byte_offset_t num_items              = large_target_copy_size / data_type_size;
  byte_offset_t num_bytes              = num_items * data_type_size;
  c2h::device_vector<data_t> d_in(num_items);
  c2h::device_vector<data_t> d_out(num_items, 42);

  auto input_data_it = thrust::make_counting_iterator(data_t{42});
  thrust::copy(input_data_it, input_data_it + num_items, d_in.begin());

  const auto num_buffers = 1;
  auto d_buffer_srcs     = thrust::make_constant_iterator(static_cast<void*>(thrust::raw_pointer_cast(d_in.data())));
  auto d_buffer_dsts     = thrust::make_constant_iterator(static_cast<void*>(thrust::raw_pointer_cast(d_out.data())));
  auto d_buffer_sizes    = thrust::make_constant_iterator(num_bytes);
  memcpy_batched(d_buffer_srcs, d_buffer_dsts, d_buffer_sizes, num_buffers);

  const bool all_equal = thrust::equal(d_out.cbegin(), d_out.cend(), input_data_it);
  REQUIRE(all_equal == true);
}
catch (std::bad_alloc& e)
{
  std::cerr << "Caught bad_alloc: " << e.what() << std::endl;
}

C2H_TEST("DeviceMemcpy::Batched works for a very large number of buffer",
         "[memcpy][skip-cs-racecheck][skip-cs-initcheck][skip-cs-synccheck]")
try
{
  using src_ptr_t       = const cuda::std::uint8_t*;
  using dst_ptr_t       = cuda::std::uint8_t*;
  using byte_offset_t   = cuda::std::uint64_t;
  using buffer_size_t   = cuda::std::int32_t;
  using buffer_offset_t = cuda::std::uint64_t;

  constexpr auto num_empty_buffers     = static_cast<buffer_offset_t>(std::numeric_limits<uint32_t>::max()) - (1 << 20);
  constexpr auto num_non_empty_buffers = buffer_offset_t{3 << 20};
  constexpr auto num_buffers           = num_empty_buffers + num_non_empty_buffers;

  buffer_size_t min_buffer_size = 1;
  buffer_size_t max_buffer_size = 100;

  // Generate the buffer sizes
  c2h::device_vector<buffer_size_t> d_buffer_sizes(num_non_empty_buffers);
  c2h::gen(C2H_SEED(2), d_buffer_sizes, min_buffer_size, max_buffer_size);
  byte_offset_t num_total_bytes = thrust::reduce(d_buffer_sizes.cbegin(), d_buffer_sizes.cend());

  // Shuffle buffer offsets
  auto d_buffer_src_offsets = get_shuffled_buffer_offsets<buffer_offset_t, byte_offset_t>(d_buffer_sizes, C2H_SEED(1));
  auto d_buffer_dst_offsets = get_shuffled_buffer_offsets<buffer_offset_t, byte_offset_t>(d_buffer_sizes, C2H_SEED(1));

  // Generate random input data and initialize output data
  c2h::device_vector<cuda::std::uint8_t> d_in(num_total_bytes);
  c2h::device_vector<cuda::std::uint8_t> d_out(num_total_bytes, 42);
  c2h::gen(C2H_SEED(1), d_in);

  // Prepare host-side input data for verification
  c2h::host_vector<cuda::std::uint8_t> h_in(d_in);
  c2h::host_vector<cuda::std::uint8_t> h_out(num_total_bytes);
  c2h::host_vector<buffer_size_t> h_buffer_sizes(d_buffer_sizes);
  c2h::host_vector<byte_offset_t> h_src_offsets(d_buffer_src_offsets);
  c2h::host_vector<byte_offset_t> h_dst_offsets(d_buffer_dst_offsets);

  // Prepare d_buffer_srcs
  offset_to_ptr_op<src_ptr_t> src_transform_op{thrust::raw_pointer_cast(d_in.data())};
  auto d_buffer_srcs =
    thrust::make_transform_iterator(thrust::raw_pointer_cast(d_buffer_src_offsets.data()), src_transform_op);

  // Return nullptr for the first num_empty_buffers and only the actual destination pointers for the rest
  prepend_n_constants_op<decltype(d_buffer_srcs), src_ptr_t> src_skip_first_n_op{
    d_buffer_srcs, nullptr, num_empty_buffers};
  auto d_buffer_srcs_skipped =
    thrust::make_transform_iterator(thrust::make_counting_iterator(buffer_offset_t{0}), src_skip_first_n_op);

  // Prepare d_buffer_dsts
  offset_to_ptr_op<dst_ptr_t> dst_transform_op{thrust::raw_pointer_cast(d_out.data())};
  thrust::transform_iterator<offset_to_ptr_op<dst_ptr_t>, byte_offset_t*> d_buffer_dsts(
    thrust::raw_pointer_cast(d_buffer_dst_offsets.data()), dst_transform_op);

  // Return nullptr for the first num_empty_buffers and only the actual destination pointers for the rest
  prepend_n_constants_op<decltype(d_buffer_dsts), dst_ptr_t> dst_skip_first_n_op{
    d_buffer_dsts, nullptr, num_empty_buffers};
  auto d_buffer_dsts_skipped =
    thrust::make_transform_iterator(thrust::make_counting_iterator(buffer_offset_t{0}), dst_skip_first_n_op);

  // Return 0 for the first num_empty_buffers and only the actual buffer sizes for the rest
  auto d_buffer_sizes_skipped = thrust::make_transform_iterator(
    thrust::make_counting_iterator(buffer_offset_t{0}),
    prepend_n_constants_op<decltype(d_buffer_sizes.cbegin()), buffer_size_t>{
      d_buffer_sizes.cbegin(), buffer_size_t{0}, num_empty_buffers});

  // Invoke device-side algorithm
  memcpy_batched(d_buffer_srcs_skipped, d_buffer_dsts_skipped, d_buffer_sizes_skipped, num_buffers);

  // Prepare CPU-side result for verification
  for (buffer_offset_t i = 0; i < num_non_empty_buffers; i++)
  {
    std::memcpy(h_out.data() + h_dst_offsets[i], h_in.data() + h_src_offsets[i], h_buffer_sizes[i]);
  }

  REQUIRE(d_out == h_out);
}
catch (std::bad_alloc& e)
{
  std::cerr << "Caught bad_alloc: " << e.what() << std::endl;
}
