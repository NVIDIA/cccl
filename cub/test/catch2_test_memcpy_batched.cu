// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_memcpy.cuh>
#include <cub/util_macro.cuh>

#include <thrust/copy.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/fill.h>
#include <thrust/gather.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/reduce.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>
#include <thrust/transform.h>

#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.h>

// %PARAM% TEST_LAUNCH lid 0:1:2

DECLARE_LAUNCH_WRAPPER(cub::DeviceMemcpy::Batched, memcpy_batched);

/**
 * @brief Function object that takes an offset and returns an iterator at the given
 * offset relative to a fixed base iterator.
 */
template <typename IteratorT>
struct offset_to_ptr_op
{
  IteratorT base_it;

  template <typename T>
  __host__ __device__ __forceinline__ IteratorT operator()(T offset) const
  {
    return base_it + offset;
  }
};

/**
 * @brief Used for generating a shuffled but cohesive sequence of output-buffer offsets for the
 * sequence of input-buffers.
 */
template <typename BufferOffsetT, typename ByteOffsetT, typename BufferSizeT>
auto get_shuffled_buffer_offsets(const c2h::device_vector<BufferSizeT>& buffer_sizes, c2h::seed_t seed)
  -> c2h::device_vector<ByteOffsetT>
{
  auto const num_buffers = static_cast<BufferOffsetT>(buffer_sizes.size());

  // We're remapping the i-th buffer to pmt_idxs[i]
  c2h::device_vector<BufferOffsetT> pmt_idxs(num_buffers);
  const auto buffer_index_it = thrust::make_counting_iterator(BufferOffsetT{0});
  thrust::shuffle_copy(
    buffer_index_it,
    buffer_index_it + num_buffers,
    pmt_idxs.begin(),
    thrust::default_random_engine(static_cast<std::uint32_t>(seed.get())));

  c2h::device_vector<ByteOffsetT> permuted_offsets(num_buffers);
  auto permuted_buffer_sizes_it = thrust::make_permutation_iterator(buffer_sizes.begin(), pmt_idxs.begin());
  thrust::exclusive_scan(permuted_buffer_sizes_it, permuted_buffer_sizes_it + num_buffers, permuted_offsets.begin());

  c2h::device_vector<BufferOffsetT> scatter_idxs(num_buffers);
  thrust::scatter(buffer_index_it, buffer_index_it + num_buffers, pmt_idxs.cbegin(), scatter_idxs.begin());

  // Gather the permuted offsets for shuffled buffer offsets
  c2h::device_vector<ByteOffsetT> buffer_offsets(num_buffers);
  thrust::gather(scatter_idxs.cbegin(), scatter_idxs.cend(), permuted_offsets.cbegin(), buffer_offsets.begin());
  return buffer_offsets;
}

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
  auto buffer_size_range = GENERATE_COPY(
    table<int, int>(
      {{0, 1},
       {1, 2},
       {0, 32},
       {1, 1024},
       {1, 32 * 1024},
       {128 * 1024, 256 * 1024},
       {target_copy_size, target_copy_size}}),
    take(4,
         map(
           [](const std::vector<int>& chunk) {
             int lhs = chunk[0];
             int rhs = chunk[1];
             // Optionally ensure lhs < rhs, for example:
             return (lhs < rhs) ? std::make_tuple(lhs, rhs) : std::make_tuple(rhs, lhs);
           },
           chunk(2, random(1, 1000000)))));

  const auto min_buffer_size       = static_cast<buffer_size_t>(std::get<0>(buffer_size_range));
  const auto max_buffer_size       = static_cast<buffer_size_t>(std::get<1>(buffer_size_range));
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
  c2h::device_vector<std::uint8_t> d_in(num_total_bytes);
  c2h::device_vector<std::uint8_t> d_out(num_total_bytes, 42);
  c2h::gen(C2H_SEED(1), d_in);

  // Prepare host-side input data for verification
  c2h::host_vector<std::uint8_t> h_in(d_in);
  c2h::host_vector<std::uint8_t> h_out(num_total_bytes);
  c2h::host_vector<buffer_size_t> h_buffer_sizes(d_buffer_sizes);
  c2h::host_vector<byte_offset_t> h_src_offsets(d_buffer_src_offsets);
  c2h::host_vector<byte_offset_t> h_dst_offsets(d_buffer_dst_offsets);

  // Prepare d_buffer_srcs
  offset_to_ptr_op<src_ptr_t> src_transform_op{static_cast<src_ptr_t>(thrust::raw_pointer_cast(d_in.data()))};
  thrust::transform_iterator<offset_to_ptr_op<src_ptr_t>, byte_offset_t*> d_buffer_srcs(
    thrust::raw_pointer_cast(d_buffer_src_offsets.data()), src_transform_op);

  // Prepare d_buffer_dsts
  offset_to_ptr_op<dst_ptr_t> dst_transform_op{static_cast<dst_ptr_t>(thrust::raw_pointer_cast(d_out.data()))};
  thrust::transform_iterator<offset_to_ptr_op<dst_ptr_t>, byte_offset_t*> d_buffer_dsts(
    thrust::raw_pointer_cast(d_buffer_dst_offsets.data()), dst_transform_op);

  // Invoke device-side algorithm
  memcpy_batched(d_buffer_srcs, d_buffer_dsts, d_buffer_sizes.begin(), num_buffers);

  // Prepare CPU-side result for verification
  for (buffer_offset_t i = 0; i < num_buffers; i++)
  {
    std::memcpy(thrust::raw_pointer_cast(h_out.data()) + h_dst_offsets[i],
                thrust::raw_pointer_cast(h_in.data()) + h_src_offsets[i],
                h_buffer_sizes[i]);
  }

  REQUIRE(d_out == h_out);
}
catch (std::bad_alloc& e)
{
  std::cerr << "Caught bad_alloc: " << e.what() << std::endl;
}

C2H_TEST("DeviceMemcpy::Batched works for a very large buffer", "[memcpy]")
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
