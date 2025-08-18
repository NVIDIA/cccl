// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/block/block_scan.cuh>

#include <cuda/functional>

#include <climits>

#include "catch2_test_block_scan_partial_helper.cuh"
#include <c2h/catch2_test_helper.h>

template <scan_mode Mode>
struct sum_op_t
{
  template <int ItemsPerThread, class BlockScanT, class T>
  __device__ void operator()(BlockScanT& scan, T (&thread_data)[ItemsPerThread], int valid_items) const
  {
    if constexpr (Mode == scan_mode::exclusive)
    {
      scan.ExclusiveScanPartialTile(thread_data, thread_data, cuda::std::plus<>{}, valid_items);
    }
    else
    {
      scan.InclusiveScanPartialTile(thread_data, thread_data, cuda::std::plus<>{}, valid_items);
    }
  }
};

template <scan_mode Mode>
struct sum_single_op_t
{
  template <class BlockScanT, class T>
  __device__ void operator()(BlockScanT& scan, T& thread_data, int valid_items) const
  {
    if constexpr (Mode == scan_mode::exclusive)
    {
      scan.ExclusiveScanPartialTile(thread_data, thread_data, cuda::std::plus<>{}, valid_items);
    }
    else
    {
      scan.InclusiveScanPartialTile(thread_data, thread_data, cuda::std::plus<>{}, valid_items);
    }
  }
};

template <class T, scan_mode Mode>
struct min_init_value_op_t
{
  T initial_value;
  template <int ItemsPerThread, class BlockScanT>
  __device__ void operator()(BlockScanT& scan, T (&thread_data)[ItemsPerThread], int valid_items) const
  {
    if constexpr (Mode == scan_mode::exclusive)
    {
      scan.ExclusiveScanPartialTile(thread_data, thread_data, initial_value, cuda::minimum<>{}, valid_items);
    }
    else
    {
      scan.InclusiveScanPartialTile(thread_data, thread_data, initial_value, cuda::minimum<>{}, valid_items);
    }
  }
};

template <scan_mode Mode>
struct min_op_t
{
  template <int ItemsPerThread, class BlockScanT>
  __device__ void operator()(BlockScanT& scan, int (&thread_data)[ItemsPerThread], int valid_items) const
  {
    if constexpr (Mode == scan_mode::exclusive)
    {
      scan.ExclusiveScanPartialTile(thread_data, thread_data, ::cuda::minimum<>{}, valid_items);
    }
    else
    {
      scan.InclusiveScanPartialTile(thread_data, thread_data, ::cuda::minimum<>{}, valid_items);
    }
  }
};

template <class T, scan_mode Mode>
struct min_init_value_aggregate_op_t
{
  int m_target_thread_id;
  T initial_value;
  T* m_d_block_aggregate;

  template <int ItemsPerThread, class BlockScanT>
  __device__ void operator()(BlockScanT& scan, T (&thread_data)[ItemsPerThread], int valid_items) const
  {
    T block_aggregate{};

    if constexpr (Mode == scan_mode::exclusive)
    {
      scan.ExclusiveScanPartialTile(
        thread_data, thread_data, initial_value, ::cuda::minimum<>{}, valid_items, block_aggregate);
    }
    else
    {
      scan.InclusiveScanPartialTile(
        thread_data, thread_data, initial_value, ::cuda::minimum<>{}, valid_items, block_aggregate);
    }

    const int tid = cub::RowMajorTid(blockDim.x, blockDim.y, blockDim.z);

    if (tid == m_target_thread_id)
    {
      *m_d_block_aggregate = block_aggregate;
    }
  }
};

template <class T, scan_mode Mode>
struct sum_aggregate_op_t
{
  int m_target_thread_id;
  T* m_d_block_aggregate;

  template <int ItemsPerThread, class BlockScanT>
  __device__ void operator()(BlockScanT& scan, T (&thread_data)[ItemsPerThread], int valid_items) const
  {
    T block_aggregate{};

    if constexpr (Mode == scan_mode::exclusive)
    {
      scan.ExclusiveScanPartialTile(thread_data, thread_data, cuda::std::plus<>{}, valid_items, block_aggregate);
    }
    else
    {
      scan.InclusiveScanPartialTile(thread_data, thread_data, cuda::std::plus<>{}, valid_items, block_aggregate);
    }

    const int tid = static_cast<int>(cub::RowMajorTid(blockDim.x, blockDim.y, blockDim.z));

    if (tid == m_target_thread_id)
    {
      *m_d_block_aggregate = block_aggregate;
    }
  }
};

template <class T, scan_mode Mode>
struct sum_prefix_op_t
{
  T m_prefix;

  struct block_prefix_op_t
  {
    int linear_tid;
    T prefix;

    __device__ block_prefix_op_t(int linear_tid, T prefix)
        : linear_tid(linear_tid)
        , prefix(prefix)
    {}

    __device__ T operator()(T block_aggregate)
    {
      T retval = (linear_tid == 0) ? prefix : T{};
      prefix   = prefix + block_aggregate;
      return retval;
    }
  };

  template <int ItemsPerThread, class BlockScanT>
  __device__ void operator()(BlockScanT& scan, T (&thread_data)[ItemsPerThread], int valid_items) const
  {
    const int tid = static_cast<int>(cub::RowMajorTid(blockDim.x, blockDim.y, blockDim.z));
    block_prefix_op_t prefix_op{tid, m_prefix};

    if constexpr (Mode == scan_mode::exclusive)
    {
      scan.ExclusiveScanPartialTile(thread_data, thread_data, cuda::std::plus<>{}, valid_items, prefix_op);
    }
    else
    {
      scan.InclusiveScanPartialTile(thread_data, thread_data, cuda::std::plus<>{}, valid_items, prefix_op);
    }
  }
};

template <class T, scan_mode Mode>
struct min_prefix_op_t
{
  T m_prefix;
  static constexpr T min_identity = ::cuda::std::numeric_limits<T>::max();

  struct block_prefix_op_t
  {
    int linear_tid;
    T prefix;

    __device__ block_prefix_op_t(int linear_tid, T prefix)
        : linear_tid(linear_tid)
        , prefix(prefix)
    {}

    __device__ T operator()(T block_aggregate)
    {
      T retval = (linear_tid == 0) ? prefix : min_identity;
      prefix   = ::cuda::minimum<>{}(prefix, block_aggregate);
      return retval;
    }
  };

  template <int ItemsPerThread, class BlockScanT>
  __device__ void operator()(BlockScanT& scan, T (&thread_data)[ItemsPerThread], int valid_items) const
  {
    const int tid = static_cast<int>(cub::RowMajorTid(blockDim.x, blockDim.y, blockDim.z));
    block_prefix_op_t prefix_op{tid, m_prefix};

    if constexpr (Mode == scan_mode::exclusive)
    {
      scan.ExclusiveScanPartialTile(thread_data, thread_data, ::cuda::minimum<>{}, valid_items, prefix_op);
    }
    else
    {
      scan.InclusiveScanPartialTile(thread_data, thread_data, ::cuda::minimum<>{}, valid_items, prefix_op);
    }
  }
};

// %PARAM% ALGO_TYPE alg 0:1:2
// %PARAM% TEST_MODE mode 0:1

using types = c2h::type_list<std::uint8_t, std::uint16_t, std::int32_t, std::int64_t>;
// FIXME(bgruber): uchar3 fails the test, see #3835
using vec_types = c2h::type_list<
#if _CCCL_CTK_AT_LEAST(13, 0)
  ulonglong4_16a,
#else // _CCCL_CTK_AT_LEAST(13, 0)
  ulonglong4,
#endif // _CCCL_CTK_AT_LEAST(13, 0)
  /*uchar3,*/ short2>;
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

template <typename T, int tile_size>
int_gen_t valid_items_fixed_vals(cub::BlockScanAlgorithm algo, int items_per_thread) noexcept
{
  const int items_per_warp           = cub::detail::warp_threads * items_per_thread;
  const int items_per_raking_segment = cub::BlockRakingLayout<T, tile_size>::SEGMENT_LENGTH;
  const int items_per_segment =
    algo == cub::BlockScanAlgorithm::BLOCK_SCAN_WARP_SCANS ? items_per_warp : items_per_raking_segment;

  using namespace Catch::Generators;
  return values({-1, 0, 1, items_per_segment - 1, items_per_segment, items_per_segment + 1, tile_size, tile_size + 1});
}
int_gen_t valid_items_rand_below() noexcept
{
  using namespace Catch::Generators;
  return take(1, random(cuda::std::numeric_limits<int>::min(), -2));
}
int_gen_t valid_items_rand_inside(int tile_size) noexcept
{
  using namespace Catch::Generators;
  return take(1, random(2, cuda::std::max(tile_size - 1, 2)));
}
int_gen_t valid_items_rand_above(int tile_size) noexcept
{
  using namespace Catch::Generators;
  return take(1, random(tile_size + 2, cuda::std::numeric_limits<int>::max()));
}

C2H_TEST("Partial block scan works with custom sum",
         "[scan][block]",
         types,
         block_dim_x,
         block_dim_yz,
         items_per_thread,
         algorithm,
         modes)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  const int valid_items = GENERATE_COPY(
    valid_items_rand_below(),
    valid_items_rand_inside(params::tile_size),
    valid_items_rand_above(params::tile_size),
    valid_items_fixed_vals<type, params::tile_size>(params::algo, params::items_per_thread));
  c2h::device_vector<type> d_out(params::tile_size);
  c2h::device_vector<type> d_in(params::tile_size);
  c2h::gen(C2H_SEED(10), d_in);

  block_scan<params::algo, params::items_per_thread, params::block_dim_x, params::block_dim_y, params::block_dim_z>(
    d_in, d_out, sum_op_t<params::mode>{}, valid_items);

  c2h::host_vector<type> h_out = d_in;
  host_scan(params::mode, h_out, std::plus<type>{}, valid_items);

  REQUIRE_APPROX_EQ(h_out, d_out);
}

C2H_TEST("Partial block scan works with custom sum single",
         "[scan][block]",
         types,
         block_dim_x,
         block_dim_yz,
         single_item_per_thread,
         algorithm,
         modes)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  const int valid_items = GENERATE_COPY(
    valid_items_rand_below(),
    valid_items_rand_inside(params::tile_size),
    valid_items_rand_above(params::tile_size),
    valid_items_fixed_vals<type, params::tile_size>(params::algo, params::items_per_thread));
  c2h::device_vector<type> d_out(params::tile_size);
  c2h::device_vector<type> d_in(params::tile_size);
  c2h::gen(C2H_SEED(10), d_in);

  block_scan_single<params::algo, params::block_dim_x, params::block_dim_y, params::block_dim_z>(
    d_in, d_out, sum_single_op_t<params::mode>{}, valid_items);

  c2h::host_vector<type> h_out = d_in;
  host_scan(params::mode, h_out, std::plus<type>{}, valid_items);

  if constexpr (params::mode == scan_mode::exclusive)
  {
    // Undefined
    h_out[0] = d_out[0];
  }
  REQUIRE_APPROX_EQ(h_out, d_out);
}

C2H_TEST("Partial block scan works with vec types", "[scan][block]", vec_types, algorithm, modes)
{
  constexpr int items_per_thread         = 3;
  constexpr int block_dim_x              = 256;
  constexpr int block_dim_y              = 1;
  constexpr int block_dim_z              = 1;
  constexpr int tile_size                = items_per_thread * block_dim_x * block_dim_y * block_dim_z;
  constexpr cub::BlockScanAlgorithm algo = c2h::get<1, TestType>::value;
  constexpr scan_mode mode               = c2h::get<2, TestType>::value;

  using type = typename c2h::get<0, TestType>;

  const int valid_items = GENERATE_COPY(
    valid_items_rand_below(),
    valid_items_rand_inside(tile_size),
    valid_items_rand_above(tile_size),
    valid_items_fixed_vals<type, tile_size>(algo, items_per_thread));
  c2h::device_vector<type> d_out(tile_size);
  c2h::device_vector<type> d_in(tile_size);
  c2h::gen(C2H_SEED(10), d_in);
  CAPTURE(valid_items, c2h::type_name<type>(), d_in);

  block_scan<algo, items_per_thread, block_dim_x, block_dim_y, block_dim_z>(d_in, d_out, sum_op_t<mode>{}, valid_items);

  c2h::host_vector<type> h_out = d_in;
  host_scan(mode, h_out, std::plus<type>{}, valid_items);

  if constexpr (mode == scan_mode::exclusive)
  {
    // Undefined
    h_out[0] = d_out[0];
  }
  REQUIRE(h_out == d_out);
}

C2H_TEST("Partial block scan works with custom types", "[scan][block]", algorithm, modes)
{
  constexpr int items_per_thread         = 3;
  constexpr int block_dim_x              = 256;
  constexpr int block_dim_y              = 1;
  constexpr int block_dim_z              = 1;
  constexpr int tile_size                = items_per_thread * block_dim_x * block_dim_y * block_dim_z;
  constexpr cub::BlockScanAlgorithm algo = c2h::get<0, TestType>::value;
  constexpr scan_mode mode               = c2h::get<1, TestType>::value;

  using type = c2h::custom_type_t<c2h::accumulateable_t, c2h::equal_comparable_t>;

  const int valid_items = GENERATE_COPY(
    valid_items_rand_below(),
    valid_items_rand_inside(tile_size),
    valid_items_rand_above(tile_size),
    valid_items_fixed_vals<type, tile_size>(algo, items_per_thread));
  c2h::device_vector<type> d_out(tile_size);
  c2h::device_vector<type> d_in(tile_size);
  c2h::gen(C2H_SEED(10), d_in);

  block_scan<algo, items_per_thread, block_dim_x, block_dim_y, block_dim_z>(d_in, d_out, sum_op_t<mode>{}, valid_items);

  c2h::host_vector<type> h_out = d_in;
  host_scan(mode, h_out, std::plus<type>{}, valid_items);

  if constexpr (mode == scan_mode::exclusive)
  {
    // Undefined
    h_out[0] = d_out[0];
  }
  REQUIRE(h_out == d_out);
}

C2H_TEST("Partial block scan returns valid block aggregate", "[scan][block]", algorithm, modes, block_dim_yz)
{
  constexpr int items_per_thread         = 3;
  constexpr int block_dim_x              = 64;
  constexpr int block_dim_y              = c2h::get<2, TestType>::value;
  constexpr int block_dim_z              = block_dim_y;
  constexpr int threads_in_block         = block_dim_x * block_dim_y * block_dim_z;
  constexpr int tile_size                = items_per_thread * threads_in_block;
  constexpr cub::BlockScanAlgorithm algo = c2h::get<0, TestType>::value;
  constexpr scan_mode mode               = c2h::get<1, TestType>::value;

  using type = c2h::custom_type_t<c2h::accumulateable_t, c2h::equal_comparable_t>;

  const int target_thread_id = GENERATE_COPY(take(2, random(0, threads_in_block - 1)));

  const int valid_items = GENERATE_COPY(
    valid_items_rand_below(),
    valid_items_rand_inside(tile_size),
    valid_items_rand_above(tile_size),
    valid_items_fixed_vals<type, tile_size>(algo, items_per_thread));
  c2h::device_vector<type> d_block_aggregate(1);
  c2h::device_vector<type> d_out(tile_size);
  c2h::device_vector<type> d_in(tile_size);
  c2h::gen(C2H_SEED(10), d_in);

  block_scan<algo, items_per_thread, block_dim_x, block_dim_y, block_dim_z>(
    d_in,
    d_out,
    sum_aggregate_op_t<type, mode>{target_thread_id, thrust::raw_pointer_cast(d_block_aggregate.data())},
    valid_items);

  c2h::host_vector<type> h_out = d_in;
  type block_aggregate         = host_scan(mode, h_out, std::plus<type>{}, valid_items);

  if constexpr (mode == scan_mode::exclusive)
  {
    // Undefined
    h_out[0] = d_out[0];
  }
  REQUIRE(h_out == d_out);
  if (valid_items > 0)
  {
    REQUIRE(block_aggregate == d_block_aggregate[0]);
  }
}

C2H_TEST("Partial block scan supports prefix op", "[scan][block]", algorithm, modes, block_dim_yz)
{
  constexpr int items_per_thread         = 3;
  constexpr int block_dim_x              = 64;
  constexpr int block_dim_y              = c2h::get<2, TestType>::value;
  constexpr int block_dim_z              = block_dim_y;
  constexpr int threads_in_block         = block_dim_x * block_dim_y * block_dim_z;
  constexpr int tile_size                = items_per_thread * threads_in_block;
  constexpr cub::BlockScanAlgorithm algo = c2h::get<0, TestType>::value;
  constexpr scan_mode mode               = c2h::get<1, TestType>::value;

  using type = int;

  const type prefix = GENERATE_COPY(take(2, random(0, tile_size)));

  const int valid_items = GENERATE_COPY(
    valid_items_rand_below(),
    valid_items_rand_inside(tile_size),
    valid_items_rand_above(tile_size),
    valid_items_fixed_vals<type, tile_size>(algo, items_per_thread));
  c2h::device_vector<type> d_out(tile_size);
  c2h::device_vector<type> d_in(tile_size);
  c2h::gen(C2H_SEED(10), d_in);

  block_scan<algo, items_per_thread, block_dim_x, block_dim_y, block_dim_z>(
    d_in, d_out, sum_prefix_op_t<type, mode>{prefix}, valid_items);

  c2h::host_vector<type> h_out = d_in;
  host_scan(mode, h_out, std::plus<type>{}, valid_items, prefix);

  REQUIRE(h_out == d_out);
}

C2H_TEST("Partial block scan supports custom scan op", "[scan][block]", algorithm, modes, block_dim_yz)
{
  constexpr int items_per_thread         = 3;
  constexpr int block_dim_x              = 64;
  constexpr int block_dim_y              = c2h::get<2, TestType>::value;
  constexpr int block_dim_z              = block_dim_y;
  constexpr int threads_in_block         = block_dim_x * block_dim_y * block_dim_z;
  constexpr int tile_size                = items_per_thread * threads_in_block;
  constexpr cub::BlockScanAlgorithm algo = c2h::get<0, TestType>::value;
  constexpr scan_mode mode               = c2h::get<1, TestType>::value;

  using type = int;

  const int valid_items = GENERATE_COPY(
    valid_items_rand_below(),
    valid_items_rand_inside(tile_size),
    valid_items_rand_above(tile_size),
    valid_items_fixed_vals<type, tile_size>(algo, items_per_thread));
  c2h::device_vector<type> d_out(tile_size);
  c2h::device_vector<type> d_in(tile_size);
  c2h::gen(C2H_SEED(10), d_in);

  block_scan<algo, items_per_thread, block_dim_x, block_dim_y, block_dim_z>(d_in, d_out, min_op_t<mode>{}, valid_items);

  c2h::host_vector<type> h_out = d_in;
  host_scan(
    mode,
    h_out,
    [](type l, type r) {
      return std::min(l, r);
    },
    valid_items,
    INT_MAX);

  if constexpr (mode == scan_mode::exclusive)
  {
    //! With no initial value, the output computed for *thread*\ :sub:`0` is undefined.
    d_out.erase(d_out.begin());
    h_out.erase(h_out.begin());
  }

  REQUIRE(h_out == d_out);
}

C2H_TEST("Partial block custom op scan works with initial value", "[scan][block]", algorithm, modes, block_dim_yz)
{
  constexpr int items_per_thread         = 3;
  constexpr int block_dim_x              = 64;
  constexpr int block_dim_y              = c2h::get<2, TestType>::value;
  constexpr int block_dim_z              = block_dim_y;
  constexpr int threads_in_block         = block_dim_x * block_dim_y * block_dim_z;
  constexpr int tile_size                = items_per_thread * threads_in_block;
  constexpr cub::BlockScanAlgorithm algo = c2h::get<0, TestType>::value;
  constexpr scan_mode mode               = c2h::get<1, TestType>::value;

  using type = int;

  const int valid_items = GENERATE_COPY(
    valid_items_rand_below(),
    valid_items_rand_inside(tile_size),
    valid_items_rand_above(tile_size),
    valid_items_fixed_vals<type, tile_size>(algo, items_per_thread));
  c2h::device_vector<type> d_out(tile_size);
  c2h::device_vector<type> d_in(tile_size);
  c2h::gen(C2H_SEED(10), d_in);

  const type initial_value = static_cast<type>(GENERATE_COPY(take(2, random(0, tile_size))));

  block_scan<algo, items_per_thread, block_dim_x, block_dim_y, block_dim_z>(
    d_in, d_out, min_init_value_op_t<type, mode>{initial_value}, valid_items);

  c2h::host_vector<type> h_out = d_in;
  host_scan(
    mode,
    h_out,
    [](type l, type r) {
      return std::min(l, r);
    },
    valid_items,
    initial_value);

  REQUIRE(h_out == d_out);
}

C2H_TEST("Partial block custom op scan with initial value returns valid block aggregate",
         "[scan][block]",
         algorithm,
         modes,
         block_dim_yz)
{
  constexpr int items_per_thread         = 3;
  constexpr int block_dim_x              = 64;
  constexpr int block_dim_y              = c2h::get<2, TestType>::value;
  constexpr int block_dim_z              = block_dim_y;
  constexpr int threads_in_block         = block_dim_x * block_dim_y * block_dim_z;
  constexpr int tile_size                = items_per_thread * threads_in_block;
  constexpr cub::BlockScanAlgorithm algo = c2h::get<0, TestType>::value;
  constexpr scan_mode mode               = c2h::get<1, TestType>::value;

  using type = int;

  const int valid_items = GENERATE_COPY(
    valid_items_rand_below(),
    valid_items_rand_inside(tile_size),
    valid_items_rand_above(tile_size),
    valid_items_fixed_vals<type, tile_size>(algo, items_per_thread));
  c2h::device_vector<type> d_out(tile_size);
  c2h::device_vector<type> d_in(tile_size);
  c2h::gen(C2H_SEED(10), d_in, 0, 1);

  const type initial_value = static_cast<type>(GENERATE_COPY(take(2, random(0, tile_size))));

  const int target_thread_id = GENERATE_COPY(take(2, random(0, threads_in_block - 1)));
  CAPTURE(valid_items, initial_value, target_thread_id, tile_size, d_in);

  c2h::device_vector<type> d_block_aggregate(1);

  block_scan<algo, items_per_thread, block_dim_x, block_dim_y, block_dim_z>(
    d_in,
    d_out,
    min_init_value_aggregate_op_t<type, mode>{
      target_thread_id, initial_value, thrust::raw_pointer_cast(d_block_aggregate.data())},
    valid_items);

  c2h::host_vector<type> h_out = d_in;
  type h_block_aggregate       = host_scan(
    mode,
    h_out,
    [](type l, type r) {
      return std::min(l, r);
    },
    valid_items,
    initial_value);

  REQUIRE(h_out == d_out);
  if (valid_items > 0)
  {
    REQUIRE(h_block_aggregate == d_block_aggregate[0]);
  }
}

C2H_TEST("Partial block scan supports prefix op and custom scan op", "[scan][block]", algorithm, modes, block_dim_yz)
{
  constexpr int items_per_thread         = 3;
  constexpr int block_dim_x              = 64;
  constexpr int block_dim_y              = c2h::get<2, TestType>::value;
  constexpr int block_dim_z              = block_dim_y;
  constexpr int threads_in_block         = block_dim_x * block_dim_y * block_dim_z;
  constexpr int tile_size                = items_per_thread * threads_in_block;
  constexpr cub::BlockScanAlgorithm algo = c2h::get<0, TestType>::value;
  constexpr scan_mode mode               = c2h::get<1, TestType>::value;

  using type = int;

  const type prefix = GENERATE_COPY(take(2, random(0, tile_size)));

  const int valid_items = GENERATE_COPY(
    valid_items_rand_below(),
    valid_items_rand_inside(tile_size),
    valid_items_rand_above(tile_size),
    valid_items_fixed_vals<type, tile_size>(algo, items_per_thread));
  c2h::device_vector<type> d_out(tile_size);
  c2h::device_vector<type> d_in(tile_size);
  c2h::gen(C2H_SEED(10), d_in);

  block_scan<algo, items_per_thread, block_dim_x, block_dim_y, block_dim_z>(
    d_in, d_out, min_prefix_op_t<type, mode>{prefix}, valid_items);

  c2h::host_vector<type> h_out = d_in;
  host_scan(
    mode,
    h_out,
    [](type a, type b) {
      return std::min(a, b);
    },
    valid_items,
    prefix);

  REQUIRE(h_out == d_out);
}
