// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_memcpy.cuh>
#include <cub/util_macro.cuh>

#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scatter.h>
#include <thrust/shuffle.h>

#include <c2h/catch2_test_helper.h>

#pragma once

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

template <typename IteratorT, typename ValueT>
struct prepend_n_constants_op
{
  IteratorT base_it;
  ValueT value_for_first_n;
  ::cuda::std::size_t num_items_to_skip;

  __host__ __device__ __forceinline__ auto operator()(::cuda::std::size_t offset) const
  {
    return offset < num_items_to_skip ? value_for_first_n : base_it[offset - num_items_to_skip];
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
