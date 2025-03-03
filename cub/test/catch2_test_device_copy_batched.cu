// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_copy.cuh>
#include <cub/util_macro.cuh>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/fill.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/transform.h>

#include <cstdint>

#include "catch2_large_problem_helper.cuh"
#include "catch2_test_device_memcpy_batched_common.cuh"
#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.h>

// %PARAM% TEST_LAUNCH lid 0:1:2

DECLARE_LAUNCH_WRAPPER(cub::DeviceCopy::Batched, copy_batched);

/**
 * @brief Function object class template that takes an offset and returns an iterator at the given
 * offset relative to a fixed base iterator.
 *
 * @tparam IteratorT The random-access iterator type to be returned
 */
template <typename IteratorT>
struct offset_to_transform_it
{
  template <typename OffsetT>
  __host__ __device__ __forceinline__ auto operator()(OffsetT offset) const
  {
    return thrust::make_transform_output_iterator(base_it + offset, ::cuda::std::identity{});
  }
  IteratorT base_it;
};

template <typename T>
struct offset_to_constant_it
{
  template <typename OffsetT>
  __host__ __device__ __forceinline__ auto operator()(OffsetT offset) const
  {
    return thrust::make_constant_iterator(static_cast<T>(offset));
  }
};

struct object_with_non_trivial_ctor
{
  static constexpr std::int32_t magic_constant = 923390;

  std::int32_t field;
  std::int32_t magic;

  __host__ __device__ object_with_non_trivial_ctor()
  {
    magic = magic_constant;
    field = 0;
  }
  __host__ __device__ object_with_non_trivial_ctor(std::int32_t f)
  {
    magic = magic_constant;
    field = f;
  }

  object_with_non_trivial_ctor(const object_with_non_trivial_ctor& x) = default;

  __host__ __device__ object_with_non_trivial_ctor& operator=(const object_with_non_trivial_ctor& x)
  {
    if (magic == magic_constant)
    {
      field = x.field;
    }
    return *this;
  }
};

static std::ostream& operator<<(std::ostream& os, const object_with_non_trivial_ctor& val)
{
  os << '(' << val.field << ',' << val.magic << ')';
  return os;
}

__host__ __device__ __forceinline__ bool
operator==(const object_with_non_trivial_ctor& lhs, const object_with_non_trivial_ctor& rhs)
{
  return lhs.field == rhs.field && lhs.magic == rhs.magic;
}

C2H_TEST("DeviceCopy::Batched works", "[copy]")
try
{
  // Type used for indexing into the array of ranges
  using range_offset_t = uint32_t;

  // Type used for indexing into individual items of a range (large enough to cover the max range's size)
  using range_size_t = uint32_t;

  // Type used for indexing into items over *all* the ranges' sizes
  using item_offset_t = uint32_t;

  // Total number of items that are targeted to be copied on each run
  constexpr range_offset_t target_copy_size = 32U << 20;

  // Pairs of [min, max] range sizes
  auto range_size_range = GENERATE_COPY(
    table<std::int32_t, std::int32_t>(
      {{0, 1},
       {1, 2},
       {0, 32},
       {1, 1024},
       {1, 32 * 1024},
       {128 * 1024, 256 * 1024},
       {target_copy_size, target_copy_size}}),
    take(4,
         map(
           [](const std::vector<std::int32_t>& chunk) {
             std::int32_t lhs = chunk[0];
             std::int32_t rhs = chunk[1];
             // Optionally ensure lhs < rhs, for example:
             return (lhs < rhs) ? std::make_tuple(lhs, rhs) : std::make_tuple(rhs, lhs);
           },
           chunk(2, random(1, 1000000)))));

  const auto min_range_size       = static_cast<range_size_t>(std::get<0>(range_size_range));
  const auto max_range_size       = static_cast<range_size_t>(std::get<1>(range_size_range));
  const double average_range_size = (min_range_size + max_range_size) / 2.0;
  const auto num_ranges           = static_cast<range_offset_t>(target_copy_size / average_range_size);

  c2h::device_vector<range_size_t> d_range_sizes(num_ranges);

  // Generate the range sizes: Make sure range sizes are a multiple of the most granular unit (one AtomicT) being
  // copied (round down)
  c2h::gen(C2H_SEED(2), d_range_sizes, min_range_size, max_range_size);
  item_offset_t num_total_items = thrust::reduce(d_range_sizes.cbegin(), d_range_sizes.cend());

  // Shuffle output range destination-offsets
  auto d_range_dst_offsets = get_shuffled_buffer_offsets<range_offset_t, item_offset_t>(d_range_sizes, C2H_SEED(1));

  // Generate random input data and initialize output data
  c2h::device_vector<std::uint8_t> d_in(num_total_items);
  c2h::device_vector<std::uint8_t> d_out(num_total_items, 42);
  c2h::gen(C2H_SEED(1), d_in);

  // Prepare host-side input data for verification
  c2h::host_vector<std::uint8_t> h_in(d_in);
  c2h::host_vector<std::uint8_t> h_out(num_total_items);
  c2h::host_vector<range_size_t> h_range_sizes(d_range_sizes);
  c2h::host_vector<item_offset_t> h_dst_offsets(d_range_dst_offsets);

  // Prepare d_range_srcs
  offset_to_constant_it<std::uint8_t> offset_to_index_op{};
  auto d_range_srcs =
    thrust::make_transform_iterator(thrust::make_counting_iterator(range_offset_t{0}), offset_to_index_op);

  // Prepare d_range_dsts
  offset_to_transform_it<std::uint8_t*> dst_transform_op{
    static_cast<std::uint8_t*>(thrust::raw_pointer_cast(d_out.data()))};
  auto d_range_dsts = thrust::make_transform_iterator(d_range_dst_offsets.begin(), dst_transform_op);

  // Invoke device-side algorithm
  copy_batched(d_range_srcs, d_range_dsts, d_range_sizes.begin(), num_ranges);

  // Prepare CPU-side result for verification
  for (range_offset_t i = 0; i < num_ranges; i++)
  {
    auto out_begin = h_out.begin() + h_dst_offsets[i];
    auto out_end   = out_begin + h_range_sizes[i];
    std::fill(out_begin, out_end, static_cast<std::uint8_t>(i));
  }

  REQUIRE(d_out == h_out);
}
catch (std::bad_alloc& e)
{
  std::cerr << "Caught bad_alloc: " << e.what() << std::endl;
}

C2H_TEST("DeviceCopy::Batched works for a very large range", "[copy]")
try
{
  using data_t        = uint64_t;
  using byte_offset_t = uint64_t;
  using buffer_size_t = uint64_t;

  byte_offset_t large_target_copy_size = static_cast<byte_offset_t>(std::numeric_limits<uint32_t>::max()) + (32 << 20);
  byte_offset_t num_items              = large_target_copy_size;

  // Input iterator for the items of a single range
  auto input_data_it = thrust::make_counting_iterator(data_t{42});

  // Prepare helper to check results
  auto check_result_helper = detail::large_problem_test_helper(num_items);
  auto check_result_it     = check_result_helper.get_flagging_output_iterator(input_data_it);

  // Run test
  const auto num_buffers = 1;
  auto d_buffer_srcs     = thrust::make_constant_iterator(input_data_it);
  auto d_buffer_dsts     = thrust::make_constant_iterator(check_result_it);
  auto d_buffer_sizes    = thrust::make_constant_iterator(num_items);
  copy_batched(d_buffer_srcs, d_buffer_dsts, d_buffer_sizes, num_buffers);

  // Verify result
  check_result_helper.check_all_results_correct();
}
catch (std::bad_alloc& e)
{
  std::cerr << "Caught bad_alloc: " << e.what() << std::endl;
}

C2H_TEST("DeviceCopy::Batched works for non-trivial ctors", "[copy]")
{
  using iterator = c2h::device_vector<object_with_non_trivial_ctor>::iterator;

  constexpr std::int32_t num_buffers = 3;
  c2h::device_vector<object_with_non_trivial_ctor> in(num_buffers, object_with_non_trivial_ctor(99));
  c2h::device_vector<object_with_non_trivial_ctor> out(num_buffers);

  c2h::device_vector<iterator> in_iter{in.begin(), in.begin() + 1, in.begin() + 2};
  c2h::device_vector<iterator> out_iter{out.begin(), out.begin() + 1, out.begin() + 2};

  auto sizes = thrust::make_constant_iterator(1);

  copy_batched(in_iter.begin(), out_iter.begin(), sizes, num_buffers);

  REQUIRE(in == out);
}
