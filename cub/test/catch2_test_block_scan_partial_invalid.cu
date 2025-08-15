// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/block/block_scan.cuh>

#include <cuda/functional>

#include <climits>

#include "catch2_test_block_scan_partial_helper.cuh"
#include "thread_reduce/catch2_test_thread_reduce_helper.cuh"
#include <c2h/catch2_test_helper.h>

template <scan_mode Mode>
struct merge_op_t
{
  bool* error_flag_ptr;
  template <int ItemsPerThread, class BlockScanT>
  __device__ void operator()(BlockScanT& scan, segment (&thread_data)[ItemsPerThread], int valid_items) const
  {
    if constexpr (Mode == scan_mode::exclusive)
    {
      scan.ExclusiveScanPartialTile(thread_data, thread_data, merge_segments_op{error_flag_ptr}, valid_items);
    }
    else
    {
      scan.InclusiveScanPartialTile(thread_data, thread_data, merge_segments_op{error_flag_ptr}, valid_items);
    }
  }
};

template <scan_mode Mode>
struct merge_single_op_t
{
  bool* error_flag_ptr;

  template <class BlockScanT>
  __device__ void operator()(BlockScanT& scan, segment& thread_data, int valid_items) const
  {
    merge_segments_op merge_op{error_flag_ptr};
    if constexpr (Mode == scan_mode::exclusive)
    {
      scan.ExclusiveScanPartialTile(thread_data, thread_data, merge_op, valid_items);
    }
    else
    {
      scan.InclusiveScanPartialTile(thread_data, thread_data, merge_op, valid_items);
    }
  }
};

template <scan_mode Mode>
struct merge_init_value_op_t
{
  segment initial_value;
  bool* error_flag_ptr;

  template <int ItemsPerThread, class BlockScanT>
  __device__ void operator()(BlockScanT& scan, segment (&thread_data)[ItemsPerThread], int valid_items) const
  {
    merge_segments_op merge_op{error_flag_ptr};
    if constexpr (Mode == scan_mode::exclusive)
    {
      scan.ExclusiveScanPartialTile(thread_data, thread_data, initial_value, merge_op, valid_items);
    }
    else
    {
      scan.InclusiveScanPartialTile(thread_data, thread_data, initial_value, merge_op, valid_items);
    }
  }
};

template <scan_mode Mode>
struct merge_init_value_aggregate_op_t
{
  int m_target_thread_id;
  segment initial_value;
  segment* m_d_block_aggregate;
  bool* error_flag_ptr;

  template <int ItemsPerThread, class BlockScanT>
  __device__ void operator()(BlockScanT& scan, segment (&thread_data)[ItemsPerThread], int valid_items) const
  {
    segment block_aggregate{};
    merge_segments_op merge_op{error_flag_ptr};

    if constexpr (Mode == scan_mode::exclusive)
    {
      scan.ExclusiveScanPartialTile(thread_data, thread_data, initial_value, merge_op, valid_items, block_aggregate);
    }
    else
    {
      scan.InclusiveScanPartialTile(thread_data, thread_data, initial_value, merge_op, valid_items, block_aggregate);
    }

    const int tid = cub::RowMajorTid(blockDim.x, blockDim.y, blockDim.z);

    if (tid == m_target_thread_id)
    {
      *m_d_block_aggregate = block_aggregate;
    }
  }
};

template <scan_mode Mode>
struct merge_aggregate_op_t
{
  int m_target_thread_id;
  segment* m_d_block_aggregate;
  bool* error_flag_ptr;

  template <int ItemsPerThread, class BlockScanT>
  __device__ void operator()(BlockScanT& scan, segment (&thread_data)[ItemsPerThread], int valid_items) const
  {
    segment block_aggregate{};
    merge_segments_op merge_op{error_flag_ptr};

    if constexpr (Mode == scan_mode::exclusive)
    {
      scan.ExclusiveScanPartialTile(thread_data, thread_data, merge_op, valid_items, block_aggregate);
    }
    else
    {
      scan.InclusiveScanPartialTile(thread_data, thread_data, merge_op, valid_items, block_aggregate);
    }

    const int tid = static_cast<int>(cub::RowMajorTid(blockDim.x, blockDim.y, blockDim.z));

    if (tid == m_target_thread_id)
    {
      *m_d_block_aggregate = block_aggregate;
    }
  }
};

template <scan_mode Mode>
struct merge_prefix_op_t
{
  segment m_prefix;
  bool* error_flag_ptr;

  struct block_prefix_op_t
  {
    int linear_tid;
    segment prefix;
    int valid_items;
    bool* error_flag_ptr;

    __device__ segment operator()(segment block_aggregate)
    {
      segment retval = (linear_tid == 0) ? prefix : segment{};
      if (linear_tid == 0 && valid_items > 0)
      {
        prefix = merge_segments_op{error_flag_ptr}(prefix, block_aggregate);
      }
      return retval;
    }
  };

  template <int ItemsPerThread, class BlockScanT>
  __device__ void operator()(BlockScanT& scan, segment (&thread_data)[ItemsPerThread], int valid_items) const
  {
    const int tid = static_cast<int>(cub::RowMajorTid(blockDim.x, blockDim.y, blockDim.z));
    block_prefix_op_t prefix_op{tid, m_prefix, valid_items, error_flag_ptr};
    merge_segments_op merge_op{error_flag_ptr};

    if constexpr (Mode == scan_mode::exclusive)
    {
      scan.ExclusiveScanPartialTile(thread_data, thread_data, merge_op, valid_items, prefix_op);
    }
    else
    {
      scan.InclusiveScanPartialTile(thread_data, thread_data, merge_op, valid_items, prefix_op);
    }
  }
};

// %PARAM% ALGO_TYPE alg 0:1:2
// %PARAM% TEST_MODE mode 0:1

using invalid_types          = c2h::type_list<segment>;
using block_dim_x            = c2h::enum_type_list<int, 17, 32, 65, 96>;
using block_dim_yz           = c2h::enum_type_list<int, 1, 2>;
using items_per_thread       = c2h::enum_type_list<int, 1, 9>;
using single_item_per_thread = c2h::enum_type_list<int, 1>;
using algorithms =
  c2h::enum_type_list<cub::BlockScanAlgorithm,
                      cub::BlockScanAlgorithm::BLOCK_SCAN_RAKING,
                      cub::BlockScanAlgorithm::BLOCK_SCAN_WARP_SCANS,
                      cub::BlockScanAlgorithm::BLOCK_SCAN_RAKING_MEMOIZE>;
using algorithm = c2h::enum_type_list<cub::BlockScanAlgorithm, c2h::get<ALGO_TYPE, algorithms>::value>;

#if TEST_MODE == 0
using modes = c2h::enum_type_list<scan_mode, scan_mode::inclusive>;
#else
using modes = c2h::enum_type_list<scan_mode, scan_mode::exclusive>;
#endif

using int_gen_t = Catch::Generators::GeneratorWrapper<int>;

template <typename Params>
int_gen_t valid_items_fixed_vals() noexcept
{
  const int items_per_warp           = cub::detail::warp_threads * Params::items_per_thread;
  const int items_per_raking_segment = cub::BlockRakingLayout<typename Params::type, Params::tile_size>::SEGMENT_LENGTH;
  const int items_per_segment =
    Params::algorithm == cub::BlockScanAlgorithm::BLOCK_SCAN_WARP_SCANS ? items_per_warp : items_per_raking_segment;

  using namespace Catch::Generators;
  return values(
    {-1,
     0,
     1,
     items_per_segment - 1,
     items_per_segment,
     items_per_segment + 1,
     Params::tile_size,
     Params::tile_size + 1});
}

int_gen_t valid_items_rand_below() noexcept
{
  using namespace Catch::Generators;
  return take(1, random(cuda::std::numeric_limits<int>::min(), -2));
}

template <typename Params>
int_gen_t valid_items_rand_inside() noexcept
{
  using namespace Catch::Generators;
  return take(1, random(2, cuda::std::max(Params::tile_size - 1, 2)));
}

template <typename Params>
int_gen_t valid_items_rand_above() noexcept
{
  using namespace Catch::Generators;
  return take(1, random(Params::tile_size + 2, cuda::std::numeric_limits<int>::max()));
}

C2H_TEST("Partial block scan (single) does not apply op to invalid items",
         "[scan][block]",
         invalid_types,
         block_dim_x,
         block_dim_yz,
         single_item_per_thread,
         algorithm,
         modes)
{
  using params = params_t<TestType>;

  const int valid_items = GENERATE_COPY(
    valid_items_rand_below(),
    valid_items_rand_inside<params>(),
    valid_items_rand_above<params>(),
    valid_items_fixed_vals<params>());
  const int bounded_valid_items = cuda::std::clamp(valid_items, 0, params::tile_size);
  CAPTURE(params::tile_size, valid_items);
  c2h::device_vector<segment> d_out(params::tile_size);
  c2h::device_vector<segment> d_in(params::tile_size);
  const auto in_it = cuda::make_transform_iterator(
    thrust::make_zip_iterator(cuda::counting_iterator<segment::offset_t>{1},
                              cuda::counting_iterator<segment::offset_t>{2}),
    tuple_to_segment_op{});
  thrust::copy(in_it, in_it + bounded_valid_items, d_in.begin());

  c2h::device_vector<bool> error_flag(1);
  block_scan_single<params::algorithm, params::block_dim_x, params::block_dim_y, params::block_dim_z>(
    d_in, d_out, merge_single_op_t<params::mode>{thrust::raw_pointer_cast(error_flag.data())}, valid_items);
  REQUIRE(false == error_flag.front());

  c2h::host_vector<segment> h_out = d_in;
  host_scan(params::mode, h_out, merge_segments_op{}, valid_items, segment{1, 1});

  if constexpr (params::mode == scan_mode::exclusive)
  {
    //! With no initial value, the output computed for *thread*\ :sub:`0` is undefined.
    h_out[0] = d_out[0];
  }
  REQUIRE(h_out == d_out);
}

C2H_TEST("Partial block scan (multi) does not apply op to invalid elements",
         "[scan][block]",
         invalid_types,
         block_dim_x,
         block_dim_yz,
         items_per_thread,
         algorithm,
         modes)
{
  using params = params_t<TestType>;

  const int valid_items = GENERATE_COPY(
    valid_items_rand_below(),
    valid_items_rand_inside<params>(),
    valid_items_rand_above<params>(),
    valid_items_fixed_vals<params>());
  const int bounded_valid_items = cuda::std::clamp(valid_items, 0, params::tile_size);
  c2h::device_vector<segment> d_out(params::tile_size);
  c2h::device_vector<segment> d_in(params::tile_size);
  const auto in_it = cuda::make_transform_iterator(
    thrust::make_zip_iterator(cuda::counting_iterator<segment::offset_t>{1},
                              cuda::counting_iterator<segment::offset_t>{2}),
    tuple_to_segment_op{});
  thrust::copy(in_it, in_it + bounded_valid_items, d_in.begin());

  c2h::device_vector<bool> error_flag(1);
  block_scan<params::algorithm, params::items_per_thread, params::block_dim_x, params::block_dim_y, params::block_dim_z>(
    d_in, d_out, merge_op_t<params::mode>{thrust::raw_pointer_cast(error_flag.data())}, valid_items);
  REQUIRE(false == error_flag.front());

  c2h::host_vector<segment> h_out = d_in;
  host_scan(params::mode, h_out, merge_segments_op{}, valid_items, segment{1, 1});

  if constexpr (params::mode == scan_mode::exclusive)
  {
    //! With no initial value, the output computed for *thread*\ :sub:`0` is undefined.
    h_out[0] = d_out[0];
  }

  REQUIRE(h_out == d_out);
}

C2H_TEST("Partial block scan (multi) does not apply op to invalid elements and returns valid block aggregate",
         "[scan][block]",
         invalid_types,
         block_dim_x,
         block_dim_yz,
         items_per_thread,
         algorithm,
         modes)
{
  using params = params_t<TestType>;

  const int target_thread_id = GENERATE_COPY(take(2, random(0, params::threads_in_block - 1)));

  const int valid_items = GENERATE_COPY(
    valid_items_rand_below(),
    valid_items_rand_inside<params>(),
    valid_items_rand_above<params>(),
    valid_items_fixed_vals<params>());
  const int bounded_valid_items = cuda::std::clamp(valid_items, 0, params::tile_size);
  c2h::device_vector<segment> d_block_aggregate(1);
  c2h::device_vector<segment> d_out(params::tile_size);
  c2h::device_vector<segment> d_in(params::tile_size);
  const auto in_it = cuda::make_transform_iterator(
    thrust::make_zip_iterator(cuda::counting_iterator<segment::offset_t>{1},
                              cuda::counting_iterator<segment::offset_t>{2}),
    tuple_to_segment_op{});
  thrust::copy(in_it, in_it + bounded_valid_items, d_in.begin());

  c2h::device_vector<bool> error_flag(1);
  block_scan<params::algorithm, params::items_per_thread, params::block_dim_x, params::block_dim_y, params::block_dim_z>(
    d_in,
    d_out,
    merge_aggregate_op_t<params::mode>{
      target_thread_id, thrust::raw_pointer_cast(d_block_aggregate.data()), thrust::raw_pointer_cast(error_flag.data())},
    valid_items);
  REQUIRE(false == error_flag.front());

  c2h::host_vector<segment> h_out = d_in;
  segment block_aggregate         = host_scan(params::mode, h_out, merge_segments_op{}, valid_items, segment{1, 1});

  if constexpr (params::mode == scan_mode::exclusive)
  {
    //! With no initial value, the output computed for *thread*\ :sub:`0` is undefined.
    h_out[0] = d_out[0];
  }
  REQUIRE(h_out == d_out);
  if (valid_items > 0)
  {
    REQUIRE(block_aggregate == d_block_aggregate[0]);
  }
}

C2H_TEST("Partial block scan (multi) does not apply op to invalid elements and works with initial value",
         "[scan][block]",
         invalid_types,
         block_dim_x,
         block_dim_yz,
         items_per_thread,
         algorithm,
         modes)
{
  using params = params_t<TestType>;

  const int valid_items = GENERATE_COPY(
    valid_items_rand_below(),
    valid_items_rand_inside<params>(),
    valid_items_rand_above<params>(),
    valid_items_fixed_vals<params>());
  const int bounded_valid_items = cuda::std::clamp(valid_items, 0, params::tile_size);
  c2h::device_vector<segment> d_out(params::tile_size);
  c2h::device_vector<segment> d_in(params::tile_size);
  const auto in_it = cuda::make_transform_iterator(
    thrust::make_zip_iterator(cuda::counting_iterator<segment::offset_t>{1},
                              cuda::counting_iterator<segment::offset_t>{2}),
    tuple_to_segment_op{});
  thrust::copy(in_it, in_it + bounded_valid_items, d_in.begin());

  const segment initial_value = segment{0, 1};

  c2h::device_vector<bool> error_flag(1);
  block_scan<params::algorithm, params::items_per_thread, params::block_dim_x, params::block_dim_y, params::block_dim_z>(
    d_in,
    d_out,
    merge_init_value_op_t<params::mode>{initial_value, thrust::raw_pointer_cast(error_flag.data())},
    valid_items);
  REQUIRE(false == error_flag.front());

  c2h::host_vector<segment> h_out = d_in;
  host_scan(params::mode, h_out, merge_segments_op{}, valid_items, initial_value);

  REQUIRE(h_out == d_out);
}

C2H_TEST("Partial block scan (multi) with initial value does not apply op to invalid elements and returns valid block "
         "aggregate",
         "[scan][block]",
         invalid_types,
         block_dim_x,
         block_dim_yz,
         items_per_thread,
         algorithm,
         modes)
{
  using params = params_t<TestType>;

  const int valid_items = GENERATE_COPY(
    valid_items_rand_below(),
    valid_items_rand_inside<params>(),
    valid_items_rand_above<params>(),
    valid_items_fixed_vals<params>());
  const int bounded_valid_items = cuda::std::clamp(valid_items, 0, params::tile_size);
  c2h::device_vector<segment> d_out(params::tile_size);
  c2h::device_vector<segment> d_in(params::tile_size);
  const auto in_it = cuda::make_transform_iterator(
    thrust::make_zip_iterator(cuda::counting_iterator<segment::offset_t>{1},
                              cuda::counting_iterator<segment::offset_t>{2}),
    tuple_to_segment_op{});
  thrust::copy(in_it, in_it + bounded_valid_items, d_in.begin());

  const segment initial_value = segment{0, 1};

  const int target_thread_id = GENERATE_COPY(take(2, random(0, params::threads_in_block - 1)));
  CAPTURE(valid_items, initial_value, target_thread_id, params::tile_size);

  c2h::device_vector<segment> d_block_aggregate(1);

  c2h::device_vector<bool> error_flag(1);
  block_scan<params::algorithm, params::items_per_thread, params::block_dim_x, params::block_dim_y, params::block_dim_z>(
    d_in,
    d_out,
    merge_init_value_aggregate_op_t<params::mode>{
      target_thread_id,
      initial_value,
      thrust::raw_pointer_cast(d_block_aggregate.data()),
      thrust::raw_pointer_cast(error_flag.data())},
    valid_items);
  REQUIRE(false == error_flag.front());

  c2h::host_vector<segment> h_out = d_in;
  segment h_block_aggregate       = host_scan(params::mode, h_out, merge_segments_op{}, valid_items, initial_value);

  REQUIRE(h_out == d_out);
  if (valid_items > 0)
  {
    REQUIRE(h_block_aggregate == d_block_aggregate[0]);
  }
}

C2H_TEST("Partial block scan (multi) supports prefix op and does not apply op to invalid items",
         "[scan][block]",
         invalid_types,
         block_dim_x,
         block_dim_yz,
         items_per_thread,
         algorithm,
         modes)
{
  using params = params_t<TestType>;

  const segment prefix = segment{0, 1};

  const int valid_items = GENERATE_COPY(
    valid_items_rand_below(),
    valid_items_rand_inside<params>(),
    valid_items_rand_above<params>(),
    valid_items_fixed_vals<params>());
  const int bounded_valid_items = cuda::std::clamp(valid_items, 0, params::tile_size);
  CAPTURE(params::tile_size, valid_items);
  c2h::device_vector<segment> d_out(params::tile_size);
  c2h::device_vector<segment> d_in(params::tile_size);
  const auto in_it = cuda::make_transform_iterator(
    thrust::make_zip_iterator(cuda::counting_iterator<segment::offset_t>{1},
                              cuda::counting_iterator<segment::offset_t>{2}),
    tuple_to_segment_op{});
  thrust::copy(in_it, in_it + bounded_valid_items, d_in.begin());

  c2h::device_vector<bool> error_flag(1);
  block_scan<params::algorithm, params::items_per_thread, params::block_dim_x, params::block_dim_y, params::block_dim_z>(
    d_in, d_out, merge_prefix_op_t<params::mode>{prefix, thrust::raw_pointer_cast(error_flag.data())}, valid_items);
  REQUIRE(false == error_flag.front());

  c2h::host_vector<segment> h_out = d_in;
  host_scan(params::mode, h_out, merge_segments_op{}, valid_items, prefix);

  REQUIRE(h_out == d_out);
}
