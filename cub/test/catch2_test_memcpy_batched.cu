// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_memcpy.cuh>
#include <cub/util_macro.cuh>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/reduce.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>
#include <thrust/transform.h>

#include <cuda/cmath>

#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.h>

// %PARAM% TEST_LAUNCH lid 0:1:2

DECLARE_LAUNCH_WRAPPER(cub::DeviceMemcpy::Batched, memcpy_batched);

enum class test_data_gen_mode
{
  // Random offsets into a data segment
  random,

  // Buffer i+1 cohesively follows buffer i in memory
  consecutive
};

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
template <typename BufferOffsetT, typename BufferSizeT, typename ByteOffsetT>
void get_shuffled_buffer_offsets(const c2h::device_vector<BufferSizeT>& buffer_sizes,
                                 c2h::device_vector<ByteOffsetT>& buffer_offsets,
                                 c2h::seed_t seed)
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
  thrust::gather(scatter_idxs.cbegin(), scatter_idxs.cend(), permuted_offsets.cbegin(), buffer_offsets.begin());
}

C2H_TEST("The batched memcopy used by DeviceMemcpy works", "[memcpy]")
try
{
  constexpr test_data_gen_mode input_gen  = test_data_gen_mode::random;
  constexpr test_data_gen_mode output_gen = test_data_gen_mode::random;

  using src_ptr_t = const uint8_t*;
  using dst_ptr_t = uint8_t*;

  // The most granular type being copied. Buffer's will be aligned and their size be an integer
  // multiple of this type
  using atomic_copy_t = uint8_t;

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
      {std::make_tuple(0, 1),
       std::make_tuple(1, 2),
       std::make_tuple(0, 32),
       std::make_tuple(1, 1024),
       std::make_tuple(1, 32 * 1024),
       std::make_tuple(128 * 1024, 256 * 1024),
       std::make_tuple(target_copy_size, target_copy_size)}),
    take(4,
         map(
           [](const std::vector<int>& chunk) {
             int lhs = chunk[0];
             int rhs = chunk[1];
             // Optionally ensure lhs < rhs, for example:
             return (lhs < rhs) ? std::make_tuple(lhs, rhs) : std::make_tuple(rhs, lhs);
           },
           chunk(2, random(1, 1000000)))));

  const auto min_buffer_size =
    static_cast<buffer_size_t>(::cuda::round_up(std::get<0>(buffer_size_range), sizeof(atomic_copy_t)));
  const auto max_buffer_size = static_cast<buffer_size_t>(
    ::cuda::round_up(std::get<1>(buffer_size_range), static_cast<buffer_size_t>(sizeof(atomic_copy_t))));
  double average_buffer_size    = (min_buffer_size + max_buffer_size) / 2.0;
  const auto target_num_buffers = static_cast<buffer_offset_t>(target_copy_size / average_buffer_size);
  const auto num_buffers        = static_cast<buffer_offset_t>(target_copy_size / average_buffer_size);

  c2h::device_vector<buffer_size_t> d_buffer_sizes(num_buffers);
  c2h::device_vector<byte_offset_t> d_buffer_src_offsets(num_buffers);
  c2h::device_vector<byte_offset_t> d_buffer_dst_offsets(num_buffers);

  // Generate the buffer sizes: Make sure buffer sizes are a multiple of the most granular unit (one AtomicT) being
  // copied (round down)
  c2h::gen(C2H_SEED(2), d_buffer_sizes, min_buffer_size, max_buffer_size);
  using thrust::placeholders::_1;
  thrust::transform(d_buffer_sizes.cbegin(),
                    d_buffer_sizes.cend(),
                    d_buffer_sizes.begin(),
                    (_1 / sizeof(atomic_copy_t) * sizeof(atomic_copy_t)));
  byte_offset_t num_total_bytes = thrust::reduce(d_buffer_sizes.cbegin(), d_buffer_sizes.cend());

  // Compute the total bytes to be copied
  if constexpr (input_gen == test_data_gen_mode::consecutive)
  {
    thrust::exclusive_scan(d_buffer_sizes.cbegin(), d_buffer_sizes.cend(), d_buffer_src_offsets.begin());
  }
  if constexpr (output_gen == test_data_gen_mode::consecutive)
  {
    thrust::exclusive_scan(d_buffer_sizes.cbegin(), d_buffer_sizes.cend(), d_buffer_dst_offsets.begin());
  }

  // Shuffle input buffer source-offsets
  if constexpr (input_gen == test_data_gen_mode::random)
  {
    get_shuffled_buffer_offsets<buffer_offset_t>(d_buffer_sizes, d_buffer_src_offsets, C2H_SEED(1));
  }

  // Shuffle input buffer source-offsets
  if (output_gen == test_data_gen_mode::random)
  {
    get_shuffled_buffer_offsets<buffer_offset_t>(d_buffer_sizes, d_buffer_dst_offsets, C2H_SEED(1));
  }

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
