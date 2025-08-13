// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/util_macro.cuh>
#include <cub/warp/warp_scan.cuh>

#include <cuda/functional>
#include <cuda/std/limits>

#include "catch2_test_warp_scan_partial_helper.cuh"
#include <c2h/catch2_test_helper.h>

using types                = c2h::type_list<std::uint8_t, std::uint16_t, std::int32_t, std::int64_t>;
using logical_warp_threads = c2h::enum_type_list<int, 32, 16, 9, 2>;
using modes                = c2h::enum_type_list<scan_mode, scan_mode::exclusive, scan_mode::inclusive>;

using vec_types = c2h::type_list<
#if _CCCL_CTK_AT_LEAST(13, 0)
  ulonglong4_16a,
#else // _CCCL_CTK_AT_LEAST(13, 0)
  ulonglong4,
#endif // _CCCL_CTK_AT_LEAST(13, 0)
  uchar3,
  short2>;

constexpr int num_seeds = 3;

template <scan_mode Mode>
struct sum_op_t
{
  template <class WarpScanT, class T>
  __device__ void operator()(WarpScanT& scan, T& thread_data, int valid_items) const
  {
    if constexpr (Mode == scan_mode::exclusive)
    {
      scan.ExclusiveScanPartial(thread_data, thread_data, cuda::std::plus<>{}, valid_items);
    }
    else
    {
      scan.InclusiveScanPartial(thread_data, thread_data, cuda::std::plus<>{}, valid_items);
    }
  }
};

template <class T, scan_mode Mode>
struct sum_aggregate_op_t
{
  int m_target_thread_id;
  T* m_d_warp_aggregate;

  template <int LogicalWarpThreads>
  __device__ void operator()(cub::WarpScan<T, LogicalWarpThreads>& scan, T& thread_data, int valid_items) const
  {
    T warp_aggregate{};

    if constexpr (Mode == scan_mode::exclusive)
    {
      scan.ExclusiveScanPartial(thread_data, thread_data, cuda::std::plus<>{}, valid_items, warp_aggregate);
    }
    else
    {
      scan.InclusiveScanPartial(thread_data, thread_data, cuda::std::plus<>{}, valid_items, warp_aggregate);
    }

    const int tid = cub::RowMajorTid(blockDim.x, blockDim.y, blockDim.z);

    if (tid % LogicalWarpThreads == m_target_thread_id)
    {
      m_d_warp_aggregate[tid / LogicalWarpThreads] = warp_aggregate;
    }
  }
};

template <scan_mode Mode>
struct min_op_t
{
  template <class T, class WarpScanT>
  __device__ void operator()(WarpScanT& scan, T& thread_data, int valid_items) const
  {
    if constexpr (Mode == scan_mode::exclusive)
    {
      scan.ExclusiveScanPartial(thread_data, thread_data, cuda::minimum<>{}, valid_items);
    }
    else
    {
      scan.InclusiveScanPartial(thread_data, thread_data, cuda::minimum<>{}, valid_items);
    }
  }
};

template <class T, scan_mode Mode>
struct min_aggregate_op_t
{
  int m_target_thread_id;
  T* m_d_warp_aggregate;

  template <int LogicalWarpThreads>
  __device__ void operator()(cub::WarpScan<T, LogicalWarpThreads>& scan, T& thread_data, int valid_items) const
  {
    T warp_aggregate{};

    if constexpr (Mode == scan_mode::exclusive)
    {
      scan.ExclusiveScanPartial(thread_data, thread_data, cuda::minimum<>{}, valid_items, warp_aggregate);
    }
    else
    {
      scan.InclusiveScanPartial(thread_data, thread_data, cuda::minimum<>{}, valid_items, warp_aggregate);
    }

    const int tid = cub::RowMajorTid(blockDim.x, blockDim.y, blockDim.z);

    if (tid % LogicalWarpThreads == m_target_thread_id)
    {
      m_d_warp_aggregate[tid / LogicalWarpThreads] = warp_aggregate;
    }
  }
};

template <class T, scan_mode Mode>
struct min_init_value_op_t
{
  T initial_value;
  template <class WarpScanT>
  __device__ void operator()(WarpScanT& scan, T& thread_data, int valid_items) const
  {
    if constexpr (Mode == scan_mode::exclusive)
    {
      scan.ExclusiveScanPartial(thread_data, thread_data, initial_value, cuda::minimum<>{}, valid_items);
    }
    else
    {
      scan.InclusiveScanPartial(thread_data, thread_data, initial_value, cuda::minimum<>{}, valid_items);
    }
  }
};

template <class T, scan_mode Mode>
struct min_init_value_aggregate_op_t
{
  int m_target_thread_id;
  T initial_value;
  T* m_d_warp_aggregate;

  template <int LogicalWarpThreads>
  __device__ void operator()(cub::WarpScan<T, LogicalWarpThreads>& scan, T& thread_data, int valid_items) const
  {
    T warp_aggregate{};

    if constexpr (Mode == scan_mode::exclusive)
    {
      scan.ExclusiveScanPartial(thread_data, thread_data, initial_value, cuda::minimum<>{}, valid_items, warp_aggregate);
    }
    else
    {
      scan.InclusiveScanPartial(thread_data, thread_data, initial_value, cuda::minimum<>{}, valid_items, warp_aggregate);
    }

    const int tid = cub::RowMajorTid(blockDim.x, blockDim.y, blockDim.z);

    if (tid % LogicalWarpThreads == m_target_thread_id)
    {
      m_d_warp_aggregate[tid / LogicalWarpThreads] = warp_aggregate;
    }
  }
};

struct min_scan_op_t
{
  template <class T, class WarpScanT>
  __device__ void
  operator()(WarpScanT& scan, T& thread_data, T& inclusive_output, T& exclusive_output, int valid_items) const
  {
    scan.ScanPartial(thread_data, inclusive_output, exclusive_output, cuda::minimum<>{}, valid_items);
  }
};

template <class T>
struct min_init_value_scan_op_t
{
  T initial_value;
  template <class WarpScanT>
  __device__ void
  operator()(WarpScanT& scan, T& thread_data, T& inclusive_output, T& exclusive_output, int valid_items) const
  {
    scan.ScanPartial(thread_data, inclusive_output, exclusive_output, initial_value, cuda::minimum<>{}, valid_items);
  }
};

C2H_TEST("Partial warp scan works with sum", "[scan][warp]", types, logical_warp_threads, modes)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  const int valid_items = GENERATE_COPY(
    take(1, random(2, cuda::std::max(2, params::logical_warp_threads))),
    take(1, random(params::logical_warp_threads + 2, cuda::std::numeric_limits<int>::max())),
    values({-1, 0, 1, params::logical_warp_threads, params::logical_warp_threads + 1}));
  CAPTURE(valid_items, params::mode, params::logical_warp_threads, c2h::type_name<type>());
  c2h::device_vector<type> d_out(params::tile_size, thrust::no_init);
  c2h::device_vector<type> d_in(params::tile_size, thrust::no_init);
  c2h::gen(C2H_SEED(num_seeds), d_in);

  warp_scan<params::logical_warp_threads, params::total_warps>(d_in, d_out, sum_op_t<params::mode>{}, valid_items);

  c2h::host_vector<type> h_out = d_in;

  compute_host_reference(params::mode, h_out, params::logical_warp_threads, std::plus<type>{}, valid_items);
  // From the documentation -
  // Computes an exclusive prefix scan using the specified binary scan functor across the calling warp. Because no
  // initial value is supplied, the output computed for warp-lane0 is undefined.

  // When comparing device output, the corresponding undefined data points need to be fixed

  if constexpr (params::mode == scan_mode::exclusive)
  {
    for (size_t i = 0; i < h_out.size(); i += params::logical_warp_threads)
    {
      d_out[i] = h_out[i];
    }
  }
  REQUIRE_APPROX_EQ(h_out, d_out);
}

C2H_TEST("Partial warp scan works with vec_types", "[scan][warp]", vec_types, logical_warp_threads, modes)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  const int valid_items = GENERATE_COPY(
    take(1, random(2, cuda::std::max(2, params::logical_warp_threads))),
    take(1, random(params::logical_warp_threads + 2, cuda::std::numeric_limits<int>::max())),
    take(1, random(cuda::std::numeric_limits<int>::min(), -2)),
    values({-1, 0, 1, params::logical_warp_threads, params::logical_warp_threads + 1}));
  CAPTURE(valid_items, params::mode, params::logical_warp_threads, c2h::type_name<type>());
  c2h::device_vector<type> d_out(params::tile_size, thrust::no_init);
  c2h::device_vector<type> d_in(params::tile_size, thrust::no_init);
  c2h::gen(C2H_SEED(num_seeds), d_in);

  warp_scan<params::logical_warp_threads, params::total_warps>(d_in, d_out, sum_op_t<params::mode>{}, valid_items);

  c2h::host_vector<type> h_out = d_in;

  compute_host_reference(params::mode, h_out, params::logical_warp_threads, std::plus<type>{}, valid_items);
  // From the documentation -
  // Computes an exclusive prefix scan using the specified binary scan functor across the calling warp. Because no
  // initial value is supplied, the output computed for warp-lane0 is undefined.

  // When comparing device output, the corresponding undefined data points need to be fixed

  if constexpr (params::mode == scan_mode::exclusive)
  {
    for (size_t i = 0; i < h_out.size(); i += params::logical_warp_threads)
    {
      d_out[i] = h_out[i];
    }
  }
  REQUIRE(h_out == d_out);
}

C2H_TEST("Partial warp scan works with custom types",
         "[scan][warp]",
         c2h::type_list<c2h::custom_type_t<c2h::accumulateable_t, c2h::equal_comparable_t>>,
         logical_warp_threads,
         modes)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  const int valid_items = GENERATE_COPY(
    take(1, random(2, cuda::std::max(2, params::logical_warp_threads))),
    take(1, random(params::logical_warp_threads + 2, cuda::std::numeric_limits<int>::max())),
    take(1, random(cuda::std::numeric_limits<int>::min(), -2)),
    values({-1, 0, 1, params::logical_warp_threads, params::logical_warp_threads + 1}));
  CAPTURE(valid_items, params::mode, params::logical_warp_threads, c2h::type_name<type>());
  c2h::device_vector<type> d_out(params::tile_size);
  c2h::device_vector<type> d_in(params::tile_size);
  c2h::gen(C2H_SEED(num_seeds), d_in);

  warp_scan<params::logical_warp_threads, params::total_warps>(d_in, d_out, sum_op_t<params::mode>{}, valid_items);

  c2h::host_vector<type> h_out = d_in;

  compute_host_reference(params::mode, h_out, params::logical_warp_threads, std::plus<type>{}, valid_items);
  // From the documentation -
  // Computes an exclusive prefix scan using the specified binary scan functor across the calling warp. Because no
  // initial value is supplied, the output computed for warp-lane0 is undefined.

  // When comparing device output, the corresponding undefined data points need to be fixed

  if constexpr (params::mode == scan_mode::exclusive)
  {
    for (size_t i = 0; i < h_out.size(); i += params::logical_warp_threads)
    {
      d_out[i] = h_out[i];
    }
  }
  REQUIRE(h_out == d_out);
}

C2H_TEST("Partial warp scan returns valid warp aggregate",
         "[scan][warp]",
         c2h::type_list<c2h::custom_type_t<c2h::accumulateable_t, c2h::equal_comparable_t>>,
         logical_warp_threads,
         modes)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  const int valid_items = GENERATE_COPY(
    take(1, random(2, cuda::std::max(2, params::logical_warp_threads))),
    take(1, random(params::logical_warp_threads + 2, cuda::std::numeric_limits<int>::max())),
    take(1, random(cuda::std::numeric_limits<int>::min(), -2)),
    values({-1, 0, 1, params::logical_warp_threads, params::logical_warp_threads + 1}));
  CAPTURE(valid_items, params::mode, params::logical_warp_threads, c2h::type_name<type>());
  c2h::device_vector<type> d_warp_aggregates(params::total_warps);
  c2h::device_vector<type> d_out(params::tile_size);
  c2h::device_vector<type> d_in(params::tile_size);
  c2h::gen(C2H_SEED(num_seeds), d_in);
  CAPTURE(d_in);

  const int target_thread_id = GENERATE_COPY(take(2, random(0, params::logical_warp_threads - 1)));

  warp_scan<params::logical_warp_threads, params::total_warps>(
    d_in,
    d_out,
    sum_aggregate_op_t<type, params::mode>{target_thread_id, thrust::raw_pointer_cast(d_warp_aggregates.data())},
    valid_items);

  c2h::host_vector<type> h_out = d_in;

  auto h_warp_aggregates =
    compute_host_reference(params::mode, h_out, params::logical_warp_threads, std::plus<type>{}, valid_items);
  // From the documentation -
  // Computes an exclusive prefix scan using the specified binary scan functor across the calling warp. Because no
  // initial value is supplied, the output computed for warp-lane0 is undefined.

  // When comparing device output, the corresponding undefined data points need to be fixed

  if constexpr (params::mode == scan_mode::exclusive)
  {
    for (size_t i = 0; i < h_out.size(); i += params::logical_warp_threads)
    {
      d_out[i] = h_out[i];
    }
  }
  REQUIRE(h_out == d_out);
  if (valid_items > 0)
  {
    REQUIRE(h_warp_aggregates == d_warp_aggregates);
  }
}

// TODO : Do we need all the types?
C2H_TEST("Partial warp scan works with custom scan op", "[scan][warp]", types, logical_warp_threads, modes)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  const int valid_items = GENERATE_COPY(
    take(1, random(2, cuda::std::max(2, params::logical_warp_threads))),
    take(1, random(params::logical_warp_threads + 2, cuda::std::numeric_limits<int>::max())),
    take(1, random(cuda::std::numeric_limits<int>::min(), -2)),
    values({-1, 0, 1, params::logical_warp_threads, params::logical_warp_threads + 1}));
  CAPTURE(valid_items, params::mode, params::logical_warp_threads, c2h::type_name<type>());
  c2h::device_vector<type> d_out(params::tile_size, thrust::no_init);
  c2h::device_vector<type> d_in(params::tile_size, thrust::no_init);
  c2h::gen(C2H_SEED(num_seeds), d_in);

  warp_scan<params::logical_warp_threads, params::total_warps>(d_in, d_out, min_op_t<params::mode>{}, valid_items);

  c2h::host_vector<type> h_out = d_in;

  compute_host_reference(
    params::mode,
    h_out,
    params::logical_warp_threads,
    [](type l, type r) {
      return std::min(l, r);
    },
    valid_items,
    cuda::std::numeric_limits<type>::max());

  // From the documentation -
  // Computes an exclusive prefix scan using the specified binary scan functor across the calling warp. Because no
  // initial value is supplied, the output computed for warp-lane0 is undefined.

  // When comparing device output, the corresponding undefined data points need to be fixed

  if constexpr (params::mode == scan_mode::exclusive)
  {
    for (size_t i = 0; i < h_out.size(); i += params::logical_warp_threads)
    {
      d_out[i] = h_out[i];
    }
  }
  REQUIRE_APPROX_EQ(h_out, d_out);
}

C2H_TEST("Partial warp custom op scan returns valid warp aggregate", "[scan][warp]", types, logical_warp_threads, modes)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  const int valid_items = GENERATE_COPY(
    take(1, random(2, cuda::std::max(2, params::logical_warp_threads))),
    take(1, random(params::logical_warp_threads + 2, cuda::std::numeric_limits<int>::max())),
    take(1, random(cuda::std::numeric_limits<int>::min(), -2)),
    values({-1, 0, 1, params::logical_warp_threads, params::logical_warp_threads + 1}));
  CAPTURE(valid_items, params::mode, params::logical_warp_threads, c2h::type_name<type>());
  c2h::device_vector<type> d_warp_aggregates(params::total_warps);
  c2h::device_vector<type> d_out(params::tile_size, thrust::no_init);
  c2h::device_vector<type> d_in(params::tile_size, thrust::no_init);
  c2h::gen(C2H_SEED(num_seeds), d_in);

  const int target_thread_id = GENERATE_COPY(take(2, random(0, params::logical_warp_threads - 1)));

  warp_scan<params::logical_warp_threads, params::total_warps>(
    d_in,
    d_out,
    min_aggregate_op_t<type, params::mode>{target_thread_id, thrust::raw_pointer_cast(d_warp_aggregates.data())},
    valid_items);

  c2h::host_vector<type> h_out = d_in;

  auto h_warp_aggregates = compute_host_reference(
    params::mode,
    h_out,
    params::logical_warp_threads,
    [](type l, type r) {
      return std::min(l, r);
    },
    valid_items,
    cuda::std::numeric_limits<type>::max());

  // From the documentation -
  // Computes an exclusive prefix scan using the specified binary scan functor across the calling warp. Because no
  // initial value is supplied, the output computed for warp-lane0 is undefined.

  // When comparing device output, the corresponding undefined data points need to be fixed

  if constexpr (params::mode == scan_mode::exclusive)
  {
    for (size_t i = 0; i < h_out.size(); i += params::logical_warp_threads)
    {
      d_out[i] = h_out[i];
    }
  }
  REQUIRE(h_out == d_out);
  if (valid_items > 0)
  {
    REQUIRE(h_warp_aggregates == d_warp_aggregates);
  }
}

C2H_TEST("Partial warp custom op scan works with initial value", "[scan][warp]", types, logical_warp_threads, modes)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  const int valid_items = GENERATE_COPY(
    take(1, random(2, cuda::std::max(2, params::logical_warp_threads))),
    take(1, random(params::logical_warp_threads + 2, cuda::std::numeric_limits<int>::max())),
    take(1, random(cuda::std::numeric_limits<int>::min(), -2)),
    values({-1, 0, 1, params::logical_warp_threads, params::logical_warp_threads + 1}));
  CAPTURE(valid_items, params::mode, params::logical_warp_threads, c2h::type_name<type>());
  c2h::device_vector<type> d_out(params::tile_size, thrust::no_init);
  c2h::device_vector<type> d_in(params::tile_size, thrust::no_init);
  c2h::gen(C2H_SEED(num_seeds), d_in);

  const type initial_value = static_cast<type>(GENERATE_COPY(take(2, random(0, params::tile_size))));
  CAPTURE(d_in, initial_value);

  warp_scan<params::logical_warp_threads, params::total_warps>(
    d_in, d_out, min_init_value_op_t<type, params::mode>{initial_value}, valid_items);

  c2h::host_vector<type> h_out = d_in;

  compute_host_reference(
    params::mode,
    h_out,
    params::logical_warp_threads,
    [](type l, type r) {
      return std::min(l, r);
    },
    valid_items,
    initial_value);

  REQUIRE_APPROX_EQ(h_out, d_out);
}

C2H_TEST("Partial warp custom op scan with initial value returns valid warp aggregate",
         "[scan][warp]",
         types,
         logical_warp_threads,
         modes)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  const int valid_items = GENERATE_COPY(
    take(1, random(2, cuda::std::max(2, params::logical_warp_threads))),
    take(1, random(params::logical_warp_threads + 2, cuda::std::numeric_limits<int>::max())),
    take(1, random(cuda::std::numeric_limits<int>::min(), -2)),
    values({-1, 0, 1, params::logical_warp_threads, params::logical_warp_threads + 1}));
  c2h::device_vector<type> d_warp_aggregates(params::total_warps);
  c2h::device_vector<type> d_out(params::tile_size, thrust::no_init);
  c2h::device_vector<type> d_in(params::tile_size, thrust::no_init);
  c2h::gen(C2H_SEED(num_seeds), d_in);

  const int target_thread_id = GENERATE_COPY(take(2, random(0, params::logical_warp_threads - 1)));
  const type initial_value   = static_cast<type>(GENERATE_COPY(take(2, random(0, params::tile_size))));
  CAPTURE(valid_items, params::mode, params::logical_warp_threads, c2h::type_name<type>(), initial_value, d_in);

  warp_scan<params::logical_warp_threads, params::total_warps>(
    d_in,
    d_out,
    min_init_value_aggregate_op_t<type, params::mode>{
      target_thread_id, initial_value, thrust::raw_pointer_cast(d_warp_aggregates.data())},
    valid_items);

  c2h::host_vector<type> h_out = d_in;

  auto h_warp_aggregates = compute_host_reference(
    params::mode,
    h_out,
    params::logical_warp_threads,
    [](type l, type r) {
      return std::min(l, r);
    },
    valid_items,
    initial_value);

  REQUIRE(h_out == d_out);
  if (valid_items > 0)
  {
    REQUIRE(h_warp_aggregates == d_warp_aggregates);
  }
}

C2H_TEST("Partial warp combination scan works with custom scan op", "[scan][warp]", logical_warp_threads)
{
  constexpr int logical_warp_threads = c2h::get<0, TestType>();
  constexpr int total_warps          = total_warps_t<logical_warp_threads>::value();
  constexpr int tile_size            = logical_warp_threads * total_warps;
  using type                         = int;

  const int valid_items = GENERATE_COPY(
    take(1, random(2, cuda::std::max(2, logical_warp_threads))),
    take(1, random(logical_warp_threads + 2, cuda::std::numeric_limits<int>::max())),
    take(1, random(cuda::std::numeric_limits<int>::min(), -2)),
    values({-1, 0, 1, logical_warp_threads, logical_warp_threads + 1}));
  const int bounded_valid_items = cuda::std::clamp(valid_items, 0, logical_warp_threads);
  const type filler =
    GENERATE_COPY(take(1, random(cuda::std::numeric_limits<type>::lowest(), cuda::std::numeric_limits<type>::max())));
  CAPTURE(valid_items, logical_warp_threads, filler, c2h::type_name<type>());
  c2h::device_vector<type> d_inclusive_out(tile_size, thrust::no_init);
  c2h::device_vector<type> d_exclusive_out(tile_size, thrust::no_init);
  c2h::device_vector<type> d_in(tile_size, thrust::no_init);
  c2h::gen(C2H_SEED(num_seeds), d_in);

  warp_combine_scan<logical_warp_threads, total_warps>(
    d_in, d_inclusive_out, d_exclusive_out, min_scan_op_t{}, valid_items, filler);

  c2h::host_vector<type> h_exclusive_out = d_in;
  c2h::host_vector<type> h_inclusive_out = d_in;
  for (int i = 0; i < tile_size; i += logical_warp_threads)
  {
    thrust::fill(
      h_exclusive_out.begin() + i + bounded_valid_items, h_exclusive_out.begin() + i + logical_warp_threads, filler);
    thrust::fill(
      h_inclusive_out.begin() + i + bounded_valid_items, h_inclusive_out.begin() + i + logical_warp_threads, filler);
  }

  compute_host_reference(
    scan_mode::exclusive,
    h_exclusive_out,
    logical_warp_threads,
    [](type l, type r) {
      return std::min(l, r);
    },
    valid_items,
    cuda::std::numeric_limits<type>::max());

  compute_host_reference(
    scan_mode::inclusive,
    h_inclusive_out,
    logical_warp_threads,
    [](type l, type r) {
      return std::min(l, r);
    },
    valid_items,
    cuda::std::numeric_limits<type>::max());

  // According to WarpScan::Scan documentation -
  // Because no initial value is supplied, the exclusive_output computed for warp-lane0 is undefined.

  // When comparing device output, the corresponding undefined data points need to be fixed

  for (size_t i = 0; i < h_exclusive_out.size(); i += logical_warp_threads)
  {
    d_exclusive_out[i] = h_exclusive_out[i];
  }

  REQUIRE(h_inclusive_out == d_inclusive_out);
  REQUIRE(h_exclusive_out == d_exclusive_out);
}

C2H_TEST("Partial warp combination custom scan works with initial value", "[scan][warp]", logical_warp_threads)
{
  constexpr int logical_warp_threads = c2h::get<0, TestType>();
  constexpr int total_warps          = total_warps_t<logical_warp_threads>::value();
  constexpr int tile_size            = logical_warp_threads * total_warps;
  using type                         = int;

  const int valid_items = GENERATE_COPY(
    take(1, random(2, cuda::std::max(2, logical_warp_threads))),
    take(1, random(logical_warp_threads + 2, cuda::std::numeric_limits<int>::max())),
    take(1, random(cuda::std::numeric_limits<int>::min(), -2)),
    values({-1, 0, 1, logical_warp_threads, logical_warp_threads + 1}));
  const int bounded_valid_items = cuda::std::clamp(valid_items, 0, logical_warp_threads);
  const type filler =
    GENERATE_COPY(take(1, random(cuda::std::numeric_limits<type>::lowest(), cuda::std::numeric_limits<type>::max())));
  CAPTURE(valid_items, logical_warp_threads, filler, c2h::type_name<type>());
  c2h::device_vector<type> d_inclusive_out(tile_size, thrust::no_init);
  c2h::device_vector<type> d_exclusive_out(tile_size, thrust::no_init);
  c2h::device_vector<type> d_in(tile_size, thrust::no_init);
  c2h::gen(C2H_SEED(num_seeds), d_in);

  const type initial_value = GENERATE_COPY(take(2, random(0, total_warps * logical_warp_threads)));
  CAPTURE(d_in, initial_value);

  warp_combine_scan<logical_warp_threads, total_warps>(
    d_in, d_inclusive_out, d_exclusive_out, min_init_value_scan_op_t<type>{initial_value}, valid_items, filler);

  c2h::host_vector<type> h_exclusive_out = d_in;
  c2h::host_vector<type> h_inclusive_out = d_in;
  for (size_t i = 0; i < tile_size; i += logical_warp_threads)
  {
    thrust::fill(
      h_exclusive_out.begin() + i + bounded_valid_items, h_exclusive_out.begin() + i + logical_warp_threads, filler);
    thrust::fill(
      h_inclusive_out.begin() + i + bounded_valid_items, h_inclusive_out.begin() + i + logical_warp_threads, filler);
  }

  compute_host_reference(
    scan_mode::exclusive,
    h_exclusive_out,
    logical_warp_threads,
    [](type l, type r) {
      return std::min(l, r);
    },
    valid_items,
    initial_value);

  compute_host_reference(
    scan_mode::inclusive,
    h_inclusive_out,
    logical_warp_threads,
    [](type l, type r) {
      return std::min(l, r);
    },
    valid_items,
    initial_value);

  REQUIRE(h_inclusive_out == d_inclusive_out);
  REQUIRE(h_exclusive_out == d_exclusive_out);
}
