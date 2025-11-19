// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/agent/agent_segmented_scan.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>

#include <thrust/tabulate.h>

#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "catch2_test_device_reduce.cuh"
#include "catch2_test_device_scan.cuh"
#include <c2h/catch2_test_helper.h>

namespace impl
{
template <typename UnsignedIntegralT>
using pair_t = cuda::std::pair<UnsignedIntegralT, UnsignedIntegralT>;

// bicyclid monoid operator is associative and non-commutative
template <typename UnsignedIntegralT>
struct bicyclic_monoid_op
{
  static_assert(cuda::std::is_integral_v<UnsignedIntegralT>);
  static_assert(cuda::std::is_unsigned_v<UnsignedIntegralT>);

  using pair_t = pair_t<UnsignedIntegralT>;
  using min_t  = cuda::minimum<>;

  pair_t __host__ __device__ operator()(pair_t v1, pair_t v2)
  {
    auto [m, n] = v1;
    auto [r, s] = v2;
    auto min_nr = min_t{}(n, r);
    return {m + r - min_nr, s + n - min_nr};
  }
};

template <typename UnsignedIntegralT>
struct populate_input
{
  static_assert(cuda::std::is_integral_v<UnsignedIntegralT>);
  static_assert(cuda::std::is_unsigned_v<UnsignedIntegralT>);

  using pair_t = pair_t<UnsignedIntegralT>;

  __host__ __device__ pair_t operator()(size_t id) const
  {
    static constexpr pair_t short_seq[] = {
      {0, 1}, {2, 3}, {4, 1}, {2, 5}, {7, 1}, {1, 1}, {0, 4}, {3, 1}, {1, 2}, {3, 2}, {4, 5}, {3, 5},
      {1, 9}, {0, 1}, {0, 1}, {0, 1}, {1, 0}, {1, 0}, {1, 0}, {2, 2}, {2, 2}, {0, 0}, {1, 1}, {2, 3},
      {2, 4}, {4, 3}, {1, 3}, {0, 3}, {1, 1}, {5, 1}, {2, 3}, {4, 7}, {2, 6}, {8, 3}, {1, 0}, {0, 8}};

    static constexpr size_t nelems = sizeof(short_seq) / sizeof(pair_t);

    return short_seq[id % nelems];
  }
};
}; // namespace impl

namespace
{
template <int BlockThreads,
          int ItemsPerThread,
          cub::BlockLoadAlgorithm LoadAlgorithm,
          cub::CacheLoadModifier LoadModifier,
          cub::BlockStoreAlgorithm StoreAlgorithm,
          cub::BlockScanAlgorithm ScanAlgorithm>
struct agent_policy_t
{
  static constexpr int BLOCK_THREADS                        = BlockThreads;
  static constexpr int ITEMS_PER_THREAD                     = ItemsPerThread;
  static constexpr cub::BlockLoadAlgorithm load_algorithm   = LoadAlgorithm;
  static constexpr cub::CacheLoadModifier load_modifier     = LoadModifier;
  static constexpr cub::BlockStoreAlgorithm store_algorithm = StoreAlgorithm;
  static constexpr cub::BlockScanAlgorithm scan_algorithm   = ScanAlgorithm;
};

template <int _BLOCK_THREADS, int _ITEMS_PER_THREAD>
struct policy_wrapper
{
  using segmented_scan_policy_t =
    agent_policy_t<_BLOCK_THREADS,
                   _ITEMS_PER_THREAD,
                   cub::BLOCK_LOAD_WARP_TRANSPOSE,
                   cub::LOAD_DEFAULT,
                   cub::BLOCK_STORE_WARP_TRANSPOSE,
                   cub::BLOCK_SCAN_WARP_SCANS>;
};

template <int _BLOCK_THREADS, int _ITEMS_PER_THREAD>
struct ChainedPolicy
{
  using ActivePolicy = policy_wrapper<_BLOCK_THREADS, _ITEMS_PER_THREAD>;
};

template <typename ChainedPolicyT,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename BeginOffsetIteratorInputT,
          typename EndOffsetIteratorInputT,
          typename BeginOffsetIteratorOutputT,
          typename OffsetT,
          typename ScanOpT,
          typename InitValueT,
          typename AccumT,
          bool ForceInclusive,
          typename ActualInitValueT = typename InitValueT::value_type>
__launch_bounds__(int(ChainedPolicyT::ActivePolicy::segmented_scan_policy_t::BLOCK_THREADS)) __global__
  void device_segmented_scan_kernel_one_segment_per_block(
    InputIteratorT d_in,
    OutputIteratorT d_out,
    BeginOffsetIteratorInputT begin_offset_d_in,
    EndOffsetIteratorInputT end_offset_d_in,
    BeginOffsetIteratorOutputT begin_offset_d_out,
    OffsetT n_segments,
    ScanOpT scan_op,
    InitValueT init_value)
{
  using segmented_scan_policy_t = typename ChainedPolicyT::ActivePolicy::segmented_scan_policy_t;

  using agent_segmented_scan_t = cub::detail::segmented_scan::agent_segmented_scan<
    segmented_scan_policy_t,
    InputIteratorT,
    OutputIteratorT,
    OffsetT,
    ScanOpT,
    ActualInitValueT,
    AccumT,
    ForceInclusive>;

  __shared__ typename agent_segmented_scan_t::TempStorage temp_storage;

  const ActualInitValueT _init_value = init_value;

  const auto segment_id = blockIdx.x;

  _CCCL_ASSERT(segment_id < n_segments,
               "device_segmented_scan_kernel launch configuration results in access violation");

  const OffsetT inp_begin_offset = begin_offset_d_in[segment_id];
  const OffsetT inp_end_offset   = end_offset_d_in[segment_id];
  const OffsetT out_begin_offset = begin_offset_d_out[segment_id];

  agent_segmented_scan_t(temp_storage, d_in, d_out, scan_op, _init_value)
    .consume_range(inp_begin_offset, inp_end_offset, out_begin_offset);
}

template <typename ChainedPolicyT,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename BeginOffsetIteratorInputT,
          typename EndOffsetIteratorInputT,
          typename BeginOffsetIteratorOutputT,
          typename OffsetT,
          typename ScanOpT,
          typename InitValueT,
          typename AccumT,
          bool ForceInclusive,
          typename ActualInitValueT = typename InitValueT::value_type>
__launch_bounds__(int(ChainedPolicyT::ActivePolicy::segmented_scan_policy_t::BLOCK_THREADS)) __global__
  void device_segmented_scan_kernel_two_segments_per_block(
    InputIteratorT d_in,
    OutputIteratorT d_out,
    BeginOffsetIteratorInputT begin_offset_d_in,
    EndOffsetIteratorInputT end_offset_d_in,
    BeginOffsetIteratorOutputT begin_offset_d_out,
    OffsetT n_segments,
    ScanOpT scan_op,
    InitValueT init_value)
{
  using segmented_scan_policy_t = typename ChainedPolicyT::ActivePolicy::segmented_scan_policy_t;

  constexpr int num_segments_per_block = 2;
  using agent_segmented_scan_t         = cub::detail::segmented_scan::agent_segmented_scan_multiple_segments_per_block<
            segmented_scan_policy_t,
            InputIteratorT,
            OutputIteratorT,
            OffsetT,
            ScanOpT,
            ActualInitValueT,
            AccumT,
            num_segments_per_block,
            ForceInclusive>;

  __shared__ typename agent_segmented_scan_t::TempStorage temp_storage;

  const ActualInitValueT _init_value = init_value;

  const auto segment_id = blockIdx.x;

  _CCCL_ASSERT(2 * segment_id + 1 < n_segments,
               "device_segmented_scan_kernel launch configuration results in access violation");

  OffsetT inp_begin_offsets[2] = {begin_offset_d_in[2 * segment_id], begin_offset_d_in[2 * segment_id + 1]};
  OffsetT inp_end_offsets[2]   = {end_offset_d_in[2 * segment_id], end_offset_d_in[2 * segment_id + 1]};
  OffsetT out_begin_offsets[2] = {begin_offset_d_out[2 * segment_id], begin_offset_d_out[2 * segment_id + 1]};

  agent_segmented_scan_t(temp_storage, d_in, d_out, scan_op, _init_value)
    .consume_ranges(inp_begin_offsets, inp_end_offsets, out_begin_offsets);
}

template <typename ChainedPolicyT,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename BeginOffsetIteratorInputT,
          typename EndOffsetIteratorInputT,
          typename BeginOffsetIteratorOutputT,
          typename OffsetT,
          typename ScanOpT,
          typename InitValueT,
          typename AccumT,
          bool ForceInclusive,
          typename ActualInitValueT = typename InitValueT::value_type>
__launch_bounds__(int(ChainedPolicyT::ActivePolicy::segmented_scan_policy_t::BLOCK_THREADS)) __global__
  void device_segmented_scan_kernel_three_segments_per_block(
    InputIteratorT d_in,
    OutputIteratorT d_out,
    BeginOffsetIteratorInputT begin_offset_d_in,
    EndOffsetIteratorInputT end_offset_d_in,
    BeginOffsetIteratorOutputT begin_offset_d_out,
    OffsetT n_segments,
    ScanOpT scan_op,
    InitValueT init_value)
{
  using segmented_scan_policy_t = typename ChainedPolicyT::ActivePolicy::segmented_scan_policy_t;

  constexpr int num_segments_per_block = 3;
  using agent_segmented_scan_t         = cub::detail::segmented_scan::agent_segmented_scan_multiple_segments_per_block<
            segmented_scan_policy_t,
            InputIteratorT,
            OutputIteratorT,
            OffsetT,
            ScanOpT,
            ActualInitValueT,
            AccumT,
            num_segments_per_block,
            ForceInclusive>;

  __shared__ typename agent_segmented_scan_t::TempStorage temp_storage;

  const ActualInitValueT _init_value = init_value;

  const auto segment_id = blockIdx.x;

  _CCCL_ASSERT(3 * segment_id < n_segments,
               "device_segmented_scan_kernel launch configuration results in access violation");

  OffsetT inp_begin_offsets[3];
  OffsetT inp_end_offsets[3];
  OffsetT out_begin_offsets[3];

  if (num_segments_per_block * blockIdx.x + num_segments_per_block - 1 < n_segments)
  {
#pragma unroll
    for (int i = 0; i < num_segments_per_block; ++i)
    {
      inp_begin_offsets[i] = begin_offset_d_in[num_segments_per_block * segment_id + i];
      inp_end_offsets[i]   = end_offset_d_in[num_segments_per_block * segment_id + i];
      out_begin_offsets[i] = begin_offset_d_out[num_segments_per_block * segment_id + i];
    }
    agent_segmented_scan_t(temp_storage, d_in, d_out, scan_op, _init_value)
      .consume_ranges(inp_begin_offsets, inp_end_offsets, out_begin_offsets);
  }
  else
  {
    int tail_size = n_segments - num_segments_per_block * blockIdx.x;
    for (int i = 0; i < tail_size; ++i)
    {
      inp_begin_offsets[i] = begin_offset_d_in[num_segments_per_block * segment_id + i];
      inp_end_offsets[i]   = end_offset_d_in[num_segments_per_block * segment_id + i];
      out_begin_offsets[i] = begin_offset_d_out[num_segments_per_block * segment_id + i];
    }
    agent_segmented_scan_t(temp_storage, d_in, d_out, scan_op, _init_value)
      .consume_ranges(inp_begin_offsets, inp_end_offsets, out_begin_offsets, tail_size);
  }
}
} // namespace

C2H_TEST("cub::detail::segmented_scan::agent_segmented_scan works with one segment per block",
         "[agent_single_segment_per_block][segmented][scan]")
{
  using op_t   = impl::bicyclic_monoid_op<unsigned>;
  using pair_t = typename op_t::pair_t;

  unsigned num_items = 128 * 16;
  c2h::device_vector<unsigned> offsets{0, num_items / 4, num_items / 2, num_items - (num_items / 4), num_items};
  size_t num_segments = offsets.size() - 1;

  c2h::device_vector<pair_t> input(num_items);
  thrust::tabulate(input.begin(), input.end(), impl::populate_input<unsigned>{});
  c2h::device_vector<pair_t> output(input.size());

  pair_t* d_input     = thrust::raw_pointer_cast(input.data());
  pair_t* d_output    = thrust::raw_pointer_cast(output.data());
  unsigned* d_offsets = thrust::raw_pointer_cast(offsets.data());

  device_segmented_scan_kernel_one_segment_per_block<
    ChainedPolicy<128, 4>,
    pair_t*,
    pair_t*,
    unsigned*,
    unsigned*,
    unsigned*,
    unsigned,
    op_t,
    cub::NullType,
    pair_t,
    true><<<num_segments, 128>>>(
    d_input, d_output, d_offsets, d_offsets + 1, d_offsets, static_cast<unsigned>(num_segments), op_t{}, cub::NullType{});

  REQUIRE(cudaSuccess == cudaGetLastError());

  // transfer to host_vector is synchronizing
  c2h::host_vector<pair_t> h_output(output);
  c2h::host_vector<pair_t> h_input(input);
  c2h::host_vector<pair_t> h_expected(input.size());
  c2h::host_vector<unsigned> h_offsets(offsets);

  for (unsigned segment_id = 0; segment_id < num_segments; ++segment_id)
  {
    compute_inclusive_scan_reference(
      h_input.begin() + h_offsets[segment_id],
      h_input.begin() + h_offsets[segment_id + 1],
      h_expected.begin() + h_offsets[segment_id],
      op_t{},
      pair_t{0, 0});
  }

  REQUIRE(h_expected == h_output);
}

C2H_TEST("cub::detail::segmented_scan::agent_segmented_scan works with two segments per block",
         "[agent_multiple_segments_per_block][segmented][scan]")
{
  using op_t   = impl::bicyclic_monoid_op<unsigned>;
  using pair_t = typename op_t::pair_t;

  unsigned num_items = 128 * 4 * 4;
  c2h::device_vector<unsigned> offsets{0, num_items / 4, num_items / 2, num_items - (num_items / 4), num_items};
  size_t num_segments = offsets.size() - 1;

  c2h::device_vector<pair_t> input(num_items);
  thrust::tabulate(input.begin(), input.end(), impl::populate_input<unsigned>{});
  c2h::device_vector<pair_t> output(input.size());

  pair_t* d_input     = thrust::raw_pointer_cast(input.data());
  pair_t* d_output    = thrust::raw_pointer_cast(output.data());
  unsigned* d_offsets = thrust::raw_pointer_cast(offsets.data());

  device_segmented_scan_kernel_two_segments_per_block<
    ChainedPolicy<128, 4>,
    pair_t*,
    pair_t*,
    unsigned*,
    unsigned*,
    unsigned*,
    unsigned,
    op_t,
    cub::NullType,
    pair_t,
    false><<<num_segments / 2, 128>>>(
    d_input, d_output, d_offsets, d_offsets + 1, d_offsets, static_cast<unsigned>(num_segments), op_t{}, cub::NullType{});

  REQUIRE(cudaSuccess == cudaGetLastError());

  // transfer to host_vector is synchronizing
  c2h::host_vector<pair_t> h_output(output);
  c2h::host_vector<pair_t> h_input(input);
  c2h::host_vector<pair_t> h_expected(input.size());
  c2h::host_vector<unsigned> h_offsets(offsets);

  for (unsigned segment_id = 0; segment_id < num_segments; ++segment_id)
  {
    compute_inclusive_scan_reference(
      h_input.begin() + h_offsets[segment_id],
      h_input.begin() + h_offsets[segment_id + 1],
      h_expected.begin() + h_offsets[segment_id],
      op_t{},
      pair_t{0, 0});
  }

  REQUIRE(h_expected == h_output);
}

C2H_TEST("cub::detail::segmented_scan::agent_segmented_scan works with three segments per block and tail",
         "[agent_multiple_segments_per_block][segmented][scan]")
{
  using op_t   = impl::bicyclic_monoid_op<unsigned>;
  using pair_t = typename op_t::pair_t;

  unsigned num_items = 128 * 4 * 4;
  c2h::device_vector<unsigned> offsets{0, num_items / 4, num_items / 2, num_items - (num_items / 4), num_items};
  size_t num_segments = offsets.size() - 1;

  c2h::device_vector<pair_t> input(num_items);
  thrust::tabulate(input.begin(), input.end(), impl::populate_input<unsigned>{});
  c2h::device_vector<pair_t> output(input.size());

  pair_t* d_input     = thrust::raw_pointer_cast(input.data());
  pair_t* d_output    = thrust::raw_pointer_cast(output.data());
  unsigned* d_offsets = thrust::raw_pointer_cast(offsets.data());

  device_segmented_scan_kernel_three_segments_per_block<
    ChainedPolicy<128, 4>,
    pair_t*,
    pair_t*,
    unsigned*,
    unsigned*,
    unsigned*,
    unsigned,
    op_t,
    cub::NullType,
    pair_t,
    false><<<cuda::ceil_div(num_segments, 3), 128>>>(
    d_input, d_output, d_offsets, d_offsets + 1, d_offsets, static_cast<unsigned>(num_segments), op_t{}, cub::NullType{});

  REQUIRE(cudaSuccess == cudaGetLastError());

  // transfer to host_vector is synchronizing
  c2h::host_vector<pair_t> h_output(output);
  c2h::host_vector<pair_t> h_input(input);
  c2h::host_vector<pair_t> h_expected(input.size());
  c2h::host_vector<unsigned> h_offsets(offsets);

  for (unsigned segment_id = 0; segment_id < num_segments; ++segment_id)
  {
    compute_inclusive_scan_reference(
      h_input.begin() + h_offsets[segment_id],
      h_input.begin() + h_offsets[segment_id + 1],
      h_expected.begin() + h_offsets[segment_id],
      op_t{},
      pair_t{0, 0});
  }

  REQUIRE(h_expected == h_output);
}
