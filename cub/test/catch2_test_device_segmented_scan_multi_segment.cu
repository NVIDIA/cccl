// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/agent/agent_segmented_scan.cuh>
#include <cub/agent/agent_thread_segmented_scan.cuh>
#include <cub/agent/agent_warp_segmented_scan.cuh>
#include <cub/device/dispatch/dispatch_segmented_scan.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>

#include <thrust/generate.h>
#include <thrust/tabulate.h>

#include <cstdint>
#include <type_traits>

#include "catch2_test_device_scan.cuh"
#include <c2h/catch2_test_helper.h>

namespace impl
{
template <typename UnsignedIntegralT>
using pair_t = cuda::std::pair<UnsignedIntegralT, UnsignedIntegralT>;

// bicyclic monoid operator is associative and non-commutative
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
struct populate_bicyclic_monoid_input
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
} // namespace impl

namespace
{
using integral_types = c2h::type_list<std::int32_t, std::int64_t, std::uint32_t, std::uint64_t>;

using itp_list =
  c2h::type_list<std::integral_constant<int, 1>,
                 std::integral_constant<int, 2>,
                 std::integral_constant<int, 3>,
                 std::integral_constant<int, 8>>;

template <int BlockThreads, int ItemsPerThread, int MaxSegmentsPerBlock, int MaxSegmentsPerWarp, typename AccumT>
struct policy_hub_t
{
  struct policy_t : cub::ChainedPolicy<300, policy_t, policy_t>
  {
    using block_segmented_scan_policy_t = cub::detail::segmented_scan::agent_segmented_scan_policy_t<
      BlockThreads,
      ItemsPerThread,
      AccumT,
      cub::BLOCK_LOAD_WARP_TRANSPOSE,
      cub::LOAD_DEFAULT,
      cub::BLOCK_STORE_WARP_TRANSPOSE,
      cub::BLOCK_SCAN_WARP_SCANS,
      MaxSegmentsPerBlock>;

    using warp_segmented_scan_policy_t = cub::detail::segmented_scan::agent_warp_segmented_scan_policy_t<
      BlockThreads,
      ItemsPerThread,
      AccumT,
      cub::WARP_LOAD_TRANSPOSE,
      cub::LOAD_DEFAULT,
      cub::WARP_STORE_TRANSPOSE,
      MaxSegmentsPerWarp>;

    using thread_segmented_scan_policy_t = cub::detail::segmented_scan::
      agent_thread_segmented_scan_policy_t<BlockThreads, ItemsPerThread, AccumT, cub::LOAD_DEFAULT>;
  };

  using MaxPolicy = policy_t;
};

template <typename DispatchT, typename OffsetT, typename InputT, typename OutputT, typename ScanOpT, typename InitValueT>
void run_dispatch_scan(
  cub::detail::segmented_scan::worker worker_choice,
  const c2h::device_vector<OffsetT>& offsets,
  const c2h::device_vector<OffsetT>& out_offsets,
  const c2h::device_vector<InputT>& input,
  c2h::device_vector<OutputT>& output,
  ScanOpT scan_op,
  InitValueT init_value,
  int segments_per_worker)
{
  const auto n_segments = static_cast<OffsetT>(offsets.size() - 1);

  const auto d_input       = thrust::raw_pointer_cast(input.data());
  auto d_output            = thrust::raw_pointer_cast(output.data());
  const auto d_offsets     = thrust::raw_pointer_cast(offsets.data());
  const auto d_out_offsets = thrust::raw_pointer_cast(out_offsets.data());

  size_t temp_storage_bytes = 0;
  DispatchT::dispatch(
    nullptr,
    temp_storage_bytes,
    d_input,
    d_output,
    n_segments,
    d_offsets,
    d_offsets + 1,
    d_out_offsets,
    scan_op,
    init_value,
    segments_per_worker,
    worker_choice,
    0);

  c2h::device_vector<cuda::std::uint8_t> temp_storage(temp_storage_bytes);
  cudaError_t code = DispatchT::dispatch(
    temp_storage.data().get(),
    temp_storage_bytes,
    d_input,
    d_output,
    n_segments,
    d_offsets,
    d_offsets + 1,
    d_out_offsets,
    scan_op,
    init_value,
    segments_per_worker,
    worker_choice,
    0);

  REQUIRE(code == cudaSuccess);
}

template <typename DispatchT, typename OffsetT, typename InputT, typename OutputT, typename ScanOpT, typename InitValueT>
void run_dispatch_scan(
  cub::detail::segmented_scan::worker worker_choice,
  const c2h::device_vector<OffsetT>& offsets,
  const c2h::device_vector<InputT>& input,
  c2h::device_vector<OutputT>& output,
  ScanOpT scan_op,
  InitValueT init_value,
  int segments_per_worker)
{
  run_dispatch_scan<DispatchT, OffsetT, InputT, OutputT, ScanOpT, InitValueT>(
    worker_choice, offsets, offsets, input, output, scan_op, init_value, segments_per_worker);
}

template <typename ValueT>
struct init_op
{
  using value_t = ValueT;

  template <typename Tp>
  __host__ __device__ value_t operator()(Tp a) const
  {
    using Up = typename cuda::std::make_unsigned<Tp>::type;
    const Up m{63};
    return static_cast<value_t>(static_cast<Up>(a) % m);
  }
};

template <typename ValueT>
struct numeric_op
{
  using value_t = ValueT;
  __host__ __device__ value_t operator()(value_t a, value_t b) const
  {
    using Up = typename cuda::std::make_unsigned<value_t>::type;
    const Up m{63};
    Up r_a = static_cast<Up>(a) % m;
    Up r_b = static_cast<Up>(b) % m;
    return (r_a + r_b) % m;
  }
};

template <typename ValueT>
struct constant_value_op
{
  using value_t = ValueT;

  value_t value;

  __host__ __device__ value_t operator()() const
  {
    return value;
  }
};
} // namespace

C2H_TEST("segmented inclusive scan works correctly for pairs with noncommutative op",
         "[multi_segment][segmented][scan]")
{
  using op_t     = impl::bicyclic_monoid_op<unsigned>;
  using pair_t   = typename op_t::pair_t;
  using offset_t = unsigned;

  constexpr int block_size             = 128;
  constexpr int items_per_thread       = 4;
  constexpr int max_segments_per_block = 256;
  constexpr int max_segments_per_warp  = 64;
  using policy_t = policy_hub_t<block_size, items_per_thread, max_segments_per_block, max_segments_per_warp, pair_t>;

  unsigned num_items = block_size * items_per_thread * 101 + 1;
  c2h::device_vector<unsigned> offsets{0, num_items / 4, num_items / 2, num_items - (num_items / 4), num_items};
  size_t num_segments = offsets.size() - 1;

  c2h::device_vector<pair_t> input(num_items);
  thrust::tabulate(input.begin(), input.end(), impl::populate_bicyclic_monoid_input<unsigned>{});
  c2h::device_vector<pair_t> output(input.size());

  using inclusive_scan_dispatch_t = cub::detail::segmented_scan::dispatch_segmented_scan<
    const pair_t*,
    pair_t*,
    const offset_t*,
    const offset_t*,
    const offset_t*,
    op_t,
    cub::NullType,
    pair_t,
    cub::ForceInclusive::No,
    offset_t,
    policy_t>;

  c2h::host_vector<pair_t> h_input(input);
  c2h::host_vector<pair_t> h_expected(input.size());
  c2h::host_vector<offset_t> h_offsets(offsets);

  op_t op{};
  pair_t h_init{0, 0};

  for (offset_t segment_id = 0; segment_id < num_segments; ++segment_id)
  {
    compute_inclusive_scan_reference(
      h_input.begin() + h_offsets[segment_id],
      h_input.begin() + h_offsets[segment_id + 1],
      h_expected.begin() + h_offsets[segment_id],
      op,
      h_init);
  }

  const int one_segment_per_worker = 1;

  cub::NullType d_no_init{};

  SECTION("worker-block, one segment per worker")
  {
    run_dispatch_scan<inclusive_scan_dispatch_t>(
      cub::detail::segmented_scan::worker::block, offsets, input, output, op, d_no_init, one_segment_per_worker);

    c2h::host_vector<pair_t> h_output(output);

    REQUIRE(h_expected == h_output);
  }

  SECTION("worker-warp, one segment per worker")
  {
    run_dispatch_scan<inclusive_scan_dispatch_t>(
      cub::detail::segmented_scan::worker::warp, offsets, input, output, op, d_no_init, one_segment_per_worker);

    c2h::host_vector<pair_t> h_output(output);

    REQUIRE(h_expected == h_output);
  }

  SECTION("worker-thread, one segment per worker")
  {
    run_dispatch_scan<inclusive_scan_dispatch_t>(
      cub::detail::segmented_scan::worker::thread, offsets, input, output, op, d_no_init, one_segment_per_worker);

    c2h::host_vector<pair_t> h_output(output);

    REQUIRE(h_expected == h_output);
  }

  const int two_segments_per_worker = 2;

  SECTION("worker-block, two segments per worker")
  {
    run_dispatch_scan<inclusive_scan_dispatch_t>(
      cub::detail::segmented_scan::worker::block, offsets, input, output, op, d_no_init, two_segments_per_worker);

    c2h::host_vector<pair_t> h_output(output);

    REQUIRE(h_expected == h_output);
  }

  SECTION("worker-warp, two segments per worker")
  {
    run_dispatch_scan<inclusive_scan_dispatch_t>(
      cub::detail::segmented_scan::worker::warp, offsets, input, output, op, d_no_init, two_segments_per_worker);

    c2h::host_vector<pair_t> h_output(output);

    REQUIRE(h_expected == h_output);
  }

  SECTION("worker-thread, two segments per worker")
  {
    run_dispatch_scan<inclusive_scan_dispatch_t>(
      cub::detail::segmented_scan::worker::thread, offsets, input, output, op, d_no_init, two_segments_per_worker);

    c2h::host_vector<pair_t> h_output(output);

    REQUIRE(h_expected == h_output);
  }
}

C2H_TEST("segmented exclusive scan works for integer types", "[multi_segment][segmented][scan]", integral_types)
{
  using value_t  = c2h::get<0, TestType>;
  using op_t     = numeric_op<value_t>;
  using offset_t = unsigned int;

  constexpr unsigned num_segments      = 7;
  constexpr unsigned items_per_segment = 128 * 4 * 33;
  constexpr unsigned num_items         = num_segments * items_per_segment;

  c2h::host_vector<unsigned> h_offsets(num_segments + 1);
  for (unsigned i = 0; i <= num_segments; ++i)
  {
    h_offsets[i] = i * items_per_segment;
  }

  c2h::device_vector<unsigned> offsets = h_offsets;
  c2h::device_vector<value_t> input(num_items);
  thrust::tabulate(input.begin(), input.end(), init_op<value_t>{});
  c2h::device_vector<value_t> output(input.size());

  constexpr int block_size             = 128;
  constexpr int items_per_thread       = 11;
  constexpr int max_segments_per_block = 256;
  constexpr int max_segments_per_warp  = 64;
  using policy_t = policy_hub_t<block_size, items_per_thread, max_segments_per_block, max_segments_per_warp, value_t>;

  using d_init_t                  = cub::detail::InputValue<value_t>;
  using exclusive_scan_dispatch_t = cub::detail::segmented_scan::dispatch_segmented_scan<
    const value_t*,
    value_t*,
    const offset_t*,
    const offset_t*,
    const offset_t*,
    op_t,
    d_init_t,
    value_t,
    cub::ForceInclusive::No,
    offset_t,
    policy_t>;

  c2h::host_vector<value_t> h_input(input);
  c2h::host_vector<value_t> h_expected(input.size());

  op_t op{};
  value_t h_init{3};

  for (unsigned segment_id = 0; segment_id < num_segments; ++segment_id)
  {
    compute_exclusive_scan_reference(
      h_input.begin() + h_offsets[segment_id],
      h_input.begin() + h_offsets[segment_id + 1],
      h_expected.begin() + h_offsets[segment_id],
      h_init,
      op);
  }

  const int segments_per_worker = 2;

  d_init_t d_init_v{h_init};

  SECTION("worker block")
  {
    run_dispatch_scan<exclusive_scan_dispatch_t>(
      cub::detail::segmented_scan::worker::block, offsets, input, output, op, d_init_v, segments_per_worker);

    c2h::host_vector<value_t> h_output(output);
    REQUIRE(h_expected == h_output);
  }

  SECTION("worker warp")
  {
    run_dispatch_scan<exclusive_scan_dispatch_t>(
      cub::detail::segmented_scan::worker::warp, offsets, input, output, op, d_init_v, segments_per_worker);

    c2h::host_vector<value_t> h_output(output);
    REQUIRE(h_expected == h_output);
  }

  SECTION("worker thread")
  {
    run_dispatch_scan<exclusive_scan_dispatch_t>(
      cub::detail::segmented_scan::worker::thread, offsets, input, output, op, d_init_v, segments_per_worker);

    c2h::host_vector<value_t> h_output(output);
    REQUIRE(h_expected == h_output);
  }
}

C2H_TEST("Segmented inclusive scan works correctly for integer types",
         "[multi_segment][segmented][scan]",
         integral_types)
{
  using value_t  = c2h::get<0, TestType>;
  using op_t     = numeric_op<value_t>;
  using offset_t = unsigned int;

  constexpr int block_size             = 128;
  constexpr int items_per_thread       = 4;
  constexpr int max_segments_per_block = 256;
  constexpr int max_segments_per_warp  = 64;
  using policy_t = policy_hub_t<block_size, items_per_thread, max_segments_per_block, max_segments_per_warp, value_t>;

  unsigned num_items = block_size * items_per_thread * 132;
  c2h::device_vector<unsigned> offsets{0, num_items / 4, num_items / 2, num_items - (num_items / 4), num_items};
  size_t num_segments = offsets.size() - 1;

  c2h::device_vector<value_t> input(num_items);
  thrust::tabulate(input.begin(), input.end(), init_op<value_t>{});
  c2h::device_vector<value_t> output(input.size());

  using inclusive_scan_dispatch_t = cub::detail::segmented_scan::dispatch_segmented_scan<
    const value_t*,
    value_t*,
    const offset_t*,
    const offset_t*,
    const offset_t*,
    op_t,
    cub::NullType,
    value_t,
    cub::ForceInclusive::No,
    offset_t,
    policy_t>;

  c2h::host_vector<value_t> h_input(input);
  c2h::host_vector<value_t> h_expected(input.size());
  c2h::host_vector<unsigned> h_offsets(offsets);

  op_t op{};
  value_t h_init{0};

  for (unsigned segment_id = 0; segment_id < num_segments; ++segment_id)
  {
    compute_inclusive_scan_reference(
      h_input.begin() + h_offsets[segment_id],
      h_input.begin() + h_offsets[segment_id + 1],
      h_expected.begin() + h_offsets[segment_id],
      op,
      h_init);
  }

  const int segments_per_worker = 4;

  cub::NullType d_no_init{};

  SECTION("worker-block")
  {
    run_dispatch_scan<inclusive_scan_dispatch_t>(
      cub::detail::segmented_scan::worker::block, offsets, input, output, op, d_no_init, segments_per_worker);

    c2h::host_vector<value_t> h_output(output);
    REQUIRE(h_expected == h_output);
  }

  SECTION("worker-warp")
  {
    run_dispatch_scan<inclusive_scan_dispatch_t>(
      cub::detail::segmented_scan::worker::warp, offsets, input, output, op, d_no_init, segments_per_worker);

    c2h::host_vector<value_t> h_output(output);
    REQUIRE(h_expected == h_output);
  }

  SECTION("worker-thread")
  {
    run_dispatch_scan<inclusive_scan_dispatch_t>(
      cub::detail::segmented_scan::worker::thread, offsets, input, output, op, d_no_init, segments_per_worker);

    c2h::host_vector<value_t> h_output(output);
    REQUIRE(h_expected == h_output);
  }
}

C2H_TEST("Segmented inclusive scan with init works for integer types",
         "[multi_segment][segmented][scan]",
         integral_types)
{
  using value_t  = c2h::get<0, TestType>;
  using op_t     = numeric_op<value_t>;
  using offset_t = unsigned;

  constexpr unsigned num_segments      = 11;
  constexpr unsigned items_per_segment = 128 * 4 * 33;
  constexpr unsigned num_items         = num_segments * items_per_segment;

  c2h::host_vector<unsigned> h_offsets(num_segments + 1);
  for (unsigned i = 0; i <= num_segments; ++i)
  {
    h_offsets[i] = i * items_per_segment;
  }

  c2h::device_vector<unsigned> offsets = h_offsets;
  c2h::device_vector<value_t> input(num_items);
  thrust::tabulate(input.begin(), input.end(), init_op<value_t>{});
  c2h::device_vector<value_t> output(input.size());

  constexpr int block_size             = 128;
  constexpr int items_per_thread       = 4;
  constexpr int max_segments_per_block = 256;
  constexpr int max_segments_per_warp  = 64;
  using policy_t = policy_hub_t<block_size, items_per_thread, max_segments_per_block, max_segments_per_warp, value_t>;

  using d_init_t                       = cub::detail::InputValue<value_t>;
  using inclusive_init_scan_dispatch_t = cub::detail::segmented_scan::dispatch_segmented_scan<
    const value_t*,
    value_t*,
    const offset_t*,
    const offset_t*,
    const offset_t*,
    op_t,
    d_init_t,
    value_t,
    cub::ForceInclusive::Yes,
    offset_t,
    policy_t>;

  c2h::host_vector<value_t> h_input(input);
  c2h::host_vector<value_t> h_expected(input.size());

  op_t op{};
  value_t h_init{3};

  for (unsigned segment_id = 0; segment_id < num_segments; ++segment_id)
  {
    compute_inclusive_scan_reference(
      h_input.begin() + h_offsets[segment_id],
      h_input.begin() + h_offsets[segment_id + 1],
      h_expected.begin() + h_offsets[segment_id],
      op,
      h_init);
  }

  d_init_t d_init_v{h_init};
  const int segments_per_worker = 2;

  // pre-condition to ensure that incomplete tail tile case is tested
  REQUIRE(num_segments % segments_per_worker != 0);

  SECTION("worker block")
  {
    run_dispatch_scan<inclusive_init_scan_dispatch_t>(
      cub::detail::segmented_scan::worker::block, offsets, input, output, op, d_init_v, segments_per_worker);

    c2h::host_vector<value_t> h_output(output);
    REQUIRE(h_expected == h_output);
  }

  SECTION("worker warp")
  {
    run_dispatch_scan<inclusive_init_scan_dispatch_t>(
      cub::detail::segmented_scan::worker::warp, offsets, input, output, op, d_init_v, segments_per_worker);

    c2h::host_vector<value_t> h_output(output);
    REQUIRE(h_expected == h_output);
  }

  SECTION("worker thread")
  {
    run_dispatch_scan<inclusive_init_scan_dispatch_t>(
      cub::detail::segmented_scan::worker::thread, offsets, input, output, op, d_init_v, segments_per_worker);

    c2h::host_vector<value_t> h_output(output);
    REQUIRE(h_expected == h_output);
  }
}

C2H_TEST("Segmented inclusive scan skips empty segments", "[multi_segment][segmented][scan]", itp_list)
{
  using op_t     = cuda::std::plus<>;
  using value_t  = unsigned int;
  using offset_t = unsigned int;

  constexpr int block_size             = 128;
  constexpr int items_per_thread       = c2h::get<0, TestType>{};
  constexpr int max_segments_per_block = 256;
  constexpr int max_segments_per_warp  = 64;
  using policy_t = policy_hub_t<block_size, items_per_thread, max_segments_per_block, max_segments_per_warp, value_t>;

  const auto canary = value_t{0xDEADBEEF};

  const offset_t gap = 4;
  c2h::device_vector<offset_t> offsets{{0, 4, 17, 17, 63, 63, 99, 127, 127, 133, 150}};
  c2h::device_vector<offset_t> out_offsets{
    {0,
     4,
     17,
     17 + gap,
     63 + gap,
     63 + 2 * gap,
     99 + 2 * gap,
     127 + 2 * gap,
     127 + 3 * gap,
     133 + 3 * gap,
     150 + 3 * gap}};

  const size_t num_segments = offsets.size() - 1;
  const unsigned num_items  = offsets.back();

  const auto num_output = out_offsets.back();

  c2h::device_vector<value_t> input(num_items);
  c2h::device_vector<value_t> output(num_output);

  thrust::tabulate(input.begin(), input.end(), cuda::std::identity{});

  constexpr int segments_per_worker = 2;

  op_t op{};
  value_t h_init_v{0};
  cub::NullType d_no_init{};

  c2h::host_vector<value_t> h_input(input);
  c2h::host_vector<offset_t> h_offsets(offsets);
  c2h::host_vector<offset_t> h_out_offsets(out_offsets);

  c2h::host_vector<value_t> h_expected(output.size(), canary);

  for (offset_t segment_id = 0; segment_id < num_segments; ++segment_id)
  {
    if (h_offsets[segment_id] == h_offsets[segment_id + 1])
    {
      continue;
    }
    compute_inclusive_scan_reference(
      h_input.begin() + h_offsets[segment_id],
      h_input.begin() + h_offsets[segment_id + 1],
      h_expected.begin() + h_out_offsets[segment_id],
      op,
      h_init_v);
  }

  using inclusive_scan_dispatch_t = cub::detail::segmented_scan::dispatch_segmented_scan<
    const value_t*,
    value_t*,
    const offset_t*,
    const offset_t*,
    const offset_t*,
    op_t,
    cub::NullType,
    value_t,
    cub::ForceInclusive::No,
    offset_t,
    policy_t>;

  SECTION("worker block")
  {
    thrust::generate(output.begin(), output.end(), constant_value_op<value_t>{canary});
    run_dispatch_scan<inclusive_scan_dispatch_t>(
      cub::detail::segmented_scan::worker::block,
      offsets,
      out_offsets,
      input,
      output,
      op,
      d_no_init,
      segments_per_worker);

    const c2h::host_vector<value_t> h_output(output);
    REQUIRE(h_output == h_expected);
  }

  SECTION("worker warp")
  {
    thrust::generate(output.begin(), output.end(), constant_value_op<value_t>{canary});
    run_dispatch_scan<inclusive_scan_dispatch_t>(
      cub::detail::segmented_scan::worker::warp, offsets, out_offsets, input, output, op, d_no_init, segments_per_worker);

    const c2h::host_vector<value_t> h_output(output);
    REQUIRE(h_output == h_expected);
  }

  SECTION("worker thread")
  {
    thrust::generate(output.begin(), output.end(), constant_value_op<value_t>{canary});
    run_dispatch_scan<inclusive_scan_dispatch_t>(
      cub::detail::segmented_scan::worker::thread,
      offsets,
      out_offsets,
      input,
      output,
      op,
      d_no_init,
      segments_per_worker);

    const c2h::host_vector<value_t> h_output(output);
    REQUIRE(h_output == h_expected);
  }
}
