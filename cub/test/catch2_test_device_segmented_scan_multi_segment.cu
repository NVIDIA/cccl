// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/dispatch/dispatch_segmented_scan.cuh>

#include <cuda/buffer>
#include <cuda/cmath>
#include <cuda/devices>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/cstddef>
#include <cuda/std/functional>
#include <cuda/std/initializer_list>
#include <cuda/std/type_traits> // std::integral_constant
#include <cuda/std/utility>
#include <cuda/stream>

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>

#include "catch2_test_device_segmented_scan_utils.cuh"
#include "catch2_test_launch_helper.h"
#include "cub_test_macros_lightweight.h"
#include <c2h/checked_memory_resource.cuh>

// %PARAM% TEST_LAUNCH lid 0:1:2

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
    auto [m, n]       = v1;
    auto [r, s]       = v2;
    const auto min_nr = min_t{}(n, r);
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
using segmented_scan_test::copy_to_host;
using segmented_scan_test::current_device;
using segmented_scan_test::enqueue_copy_to_device;
using segmented_scan_test::make_device_buffer_from_host;
using segmented_scan_test::make_host_buffer;
using segmented_scan_test::make_tabulated_host_buffer;
using segmented_scan_test::require_ranges_equal;

template <typename T>
constexpr unsigned int get_max_elems()
{
  constexpr unsigned int max_input_bytes = (static_cast<unsigned int>(1) << 19);
  constexpr auto elem_bytes              = static_cast<unsigned int>(sizeof(T));
  return cuda::ceil_div(max_input_bytes, elem_bytes);
}

using integral_types = c2h::type_list<std::int32_t, std::int64_t, std::uint32_t, std::uint64_t>;

using itp_list =
  c2h::type_list<cuda::std::integral_constant<int, 1>,
                 cuda::std::integral_constant<int, 2>,
                 cuda::std::integral_constant<int, 3>,
                 cuda::std::integral_constant<int, 8>>;

template <int ThreadsPerBlock, int ItemsPerThread, int MaxSegmentsPerBlock>
struct policy_selector_t
{
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability) const
    -> cub::SegmentedScanPolicy
  {
    return cub::SegmentedScanPolicy{cub::SegmentedScanBlockPolicy{
      ThreadsPerBlock,
      ItemsPerThread,
      cub::BLOCK_LOAD_WARP_TRANSPOSE,
      cub::LOAD_DEFAULT,
      cub::BLOCK_STORE_WARP_TRANSPOSE,
      cub::BLOCK_SCAN_WARP_SCANS,
      MaxSegmentsPerBlock}};
  }
};

DECLARE_TMPL_LAUNCH_WRAPPER(
  cub::detail::segmented_scan::dispatch,
  dispatch_segmented_scan,
  ESCAPE_LIST(
    cub::ForceInclusive EnforceInclusive,
    typename InputIteratorT,
    typename OutputIteratorT,
    typename BeginOffsetIteratorInputT,
    typename EndOffsetIteratorInputT,
    typename BeginOffsetIteratorOutputT,
    typename ScanOpT,
    typename InitValueT,
    typename AccumT,
    typename OffsetT,
    typename PolicySelector),
  ESCAPE_LIST(
    EnforceInclusive,
    InputIteratorT,
    OutputIteratorT,
    BeginOffsetIteratorInputT,
    EndOffsetIteratorInputT,
    BeginOffsetIteratorOutputT,
    ScanOpT,
    InitValueT,
    AccumT,
    OffsetT,
    PolicySelector));

template <typename DispatchT, typename OffsetT, typename InputT, typename OutputT, typename ScanOpT, typename InitValueT>
void run_dispatch_scan(
  DispatchT dispatch_fn,
  [[maybe_unused]] cuda::stream_ref stream,
  cub::detail::segmented_scan::worker worker_choice,
  const cuda::device_buffer<OffsetT>& offsets,
  const cuda::device_buffer<OffsetT>& out_offsets,
  const cuda::device_buffer<InputT>& input,
  cuda::device_buffer<OutputT>& output,
  ScanOpT scan_op,
  InitValueT init_value,
  int segments_per_worker)
{
  const auto n_segments = static_cast<OffsetT>(offsets.size() - 1);

  const auto d_input       = input.data();
  const auto d_output      = output.data();
  const auto d_offsets     = offsets.data();
  const auto d_out_offsets = out_offsets.data();

  dispatch_fn(
    d_input,
    d_output,
    n_segments,
    d_offsets,
    d_offsets + 1,
    d_out_offsets,
    scan_op,
    init_value,
    segments_per_worker,
    worker_choice
#if TEST_LAUNCH == 0
    ,
    stream.get()
#elif TEST_LAUNCH == 1
    ,
    nullptr // Host stream handles are invalid for device-side launches.
#endif // TEST_LAUNCH == 0 / TEST_LAUNCH == 1
  );
}

template <typename DispatchT, typename OffsetT, typename InputT, typename OutputT, typename ScanOpT, typename InitValueT>
void run_dispatch_scan(
  DispatchT dispatch_fn,
  [[maybe_unused]] cuda::stream_ref stream,
  cub::detail::segmented_scan::worker worker_choice,
  const cuda::device_buffer<OffsetT>& offsets,
  const cuda::device_buffer<InputT>& input,
  cuda::device_buffer<OutputT>& output,
  ScanOpT scan_op,
  InitValueT init_value,
  int segments_per_worker)
{
  run_dispatch_scan(
    dispatch_fn, stream, worker_choice, offsets, offsets, input, output, scan_op, init_value, segments_per_worker);
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
} // namespace

CUB_TEST("segmented inclusive scan works correctly for pairs with noncommutative op",
         "[multi_segment][segmented][scan]",
         CUB_SMALL)
{
  using op_t     = impl::bicyclic_monoid_op<unsigned int>;
  using pair_t   = typename op_t::pair_t;
  using offset_t = unsigned int;

  // WAR for MSVC which incorrectly deduces that these variables are declared, but unused
  [[maybe_unused]] static constexpr int items_per_thread       = 4;
  [[maybe_unused]] static constexpr int block_size             = 128;
  [[maybe_unused]] static constexpr int max_segments_per_block = 256;

  using policy_t = policy_selector_t<block_size, items_per_thread, max_segments_per_block>;

  const auto device = current_device();
  auto copy_stream  = cuda::stream{device};

  const unsigned num_items = block_size * items_per_thread * 101 + 1;
  const auto h_offsets     = make_host_buffer<offset_t>(
    copy_stream,
    device,
    cuda::std::initializer_list<offset_t>{0, num_items / 4, num_items / 2, num_items - (num_items / 4), num_items});
  const std::size_t num_segments = h_offsets.size() - 1;

  auto inclusive_scan_dispatch = [](auto&&... args) {
    dispatch_segmented_scan<
      cub::ForceInclusive::No,
      const pair_t*,
      pair_t*,
      const offset_t*,
      const offset_t*,
      const offset_t*,
      op_t,
      cub::NullType,
      pair_t,
      offset_t,
      policy_t>(std::forward<decltype(args)>(args)...);
  };

  const auto h_input = make_tabulated_host_buffer<pair_t>(
    copy_stream, device, num_items, impl::populate_bicyclic_monoid_input<unsigned int>{});
  auto h_expected = make_host_buffer<pair_t>(copy_stream, device, h_input.size(), cuda::no_init);

  auto offsets = make_device_buffer_from_host(copy_stream, device, h_offsets);
  auto input   = make_device_buffer_from_host(copy_stream, device, h_input);
  auto output  = c2h::make_device_buffer<pair_t>(copy_stream, device, h_input.size(), cuda::no_init);
  copy_stream.sync();

  op_t op{};
  pair_t h_init{0, 0};

  for (offset_t segment_id = 0; segment_id < num_segments; ++segment_id)
  {
    compute_inclusive_scan_reference(
      h_input.begin() + h_offsets[segment_id],
      h_input.begin() + h_offsets[segment_id + 1], // NOLINT(bugprone-misplaced-widening-cast)
      h_expected.begin() + h_offsets[segment_id],
      op,
      h_init);
  }

  const int one_segment_per_worker = 1;

  cub::NullType d_no_init{};

  SECTION("worker-block, one segment per worker")
  {
    run_dispatch_scan(
      inclusive_scan_dispatch,
      copy_stream,
      cub::detail::segmented_scan::worker::block,
      offsets,
      input,
      output,
      op,
      d_no_init,
      one_segment_per_worker);

    auto h_output = make_host_buffer<pair_t>(copy_stream, device, output.size(), cuda::no_init);
    copy_to_host(copy_stream, output, h_output);

    require_ranges_equal(h_expected, h_output);
  }

  const int two_segments_per_worker = 2;

  SECTION("worker-block, two segments per worker")
  {
    run_dispatch_scan(
      inclusive_scan_dispatch,
      copy_stream,
      cub::detail::segmented_scan::worker::block,
      offsets,
      input,
      output,
      op,
      d_no_init,
      two_segments_per_worker);

    auto h_output = make_host_buffer<pair_t>(copy_stream, device, output.size(), cuda::no_init);
    copy_to_host(copy_stream, output, h_output);

    require_ranges_equal(h_expected, h_output);
  }
}

CUB_TEST(
  "segmented exclusive scan works for integer types", "[multi_segment][segmented][scan]", CUB_SMALL, integral_types)
{
  using value_t  = c2h::get<0, TestType>;
  using op_t     = numeric_op<value_t>;
  using offset_t = unsigned int;

  const auto device = current_device();
  auto copy_stream  = cuda::stream{device};

  constexpr auto max_nelems = get_max_elems<value_t>();

  // repeat the test for multiple values of num_segments
  const unsigned int num_segments      = GENERATE(7, 9, 13, 129);
  const unsigned int items_per_segment = cuda::ceil_div(max_nelems, num_segments);
  const unsigned int num_items         = num_segments * items_per_segment;

  CAPTURE(num_segments, num_items, items_per_segment, cuda::std::is_signed_v<value_t>);

  auto h_offsets = make_host_buffer<offset_t>(copy_stream, device, num_segments + 1, cuda::no_init);
  for (unsigned i = 0; i <= num_segments; ++i)
  {
    h_offsets[i] = i * items_per_segment;
  }

  const auto h_input = make_tabulated_host_buffer<value_t>(copy_stream, device, num_items, init_op<value_t>{});

  auto offsets = make_device_buffer_from_host(copy_stream, device, h_offsets);
  auto input   = make_device_buffer_from_host(copy_stream, device, h_input);
  auto output  = c2h::make_device_buffer<value_t>(copy_stream, device, h_input.size(), cuda::no_init);
  copy_stream.sync();

  // WAR for MSVC which incorrectly deduces that these variables are declared, but unused
  [[maybe_unused]] static constexpr int items_per_thread       = 11;
  [[maybe_unused]] static constexpr int block_size             = 128;
  [[maybe_unused]] static constexpr int max_segments_per_block = 16;

  using policy_t = policy_selector_t<block_size, items_per_thread, max_segments_per_block>;

  using d_init_t               = cub::detail::InputValue<value_t>;
  auto exclusive_scan_dispatch = [](auto&&... args) {
    dispatch_segmented_scan<
      cub::ForceInclusive::No,
      const value_t*,
      value_t*,
      const offset_t*,
      const offset_t*,
      const offset_t*,
      op_t,
      d_init_t,
      value_t,
      offset_t,
      policy_t>(std::forward<decltype(args)>(args)...);
  };

  auto h_expected = make_host_buffer<value_t>(copy_stream, device, h_input.size(), cuda::no_init);

  op_t op{};
  value_t h_init{3};

  for (unsigned segment_id = 0; segment_id < num_segments; ++segment_id)
  {
    compute_exclusive_scan_reference(
      h_input.begin() + h_offsets[segment_id],
      h_input.begin() + h_offsets[segment_id + 1], // NOLINT(bugprone-misplaced-widening-cast)
      h_expected.begin() + h_offsets[segment_id],
      h_init,
      op);
  }

  const int segments_per_worker = 2;

  d_init_t d_init_v{h_init};

  SECTION("worker block")
  {
    run_dispatch_scan(
      exclusive_scan_dispatch,
      copy_stream,
      cub::detail::segmented_scan::worker::block,
      offsets,
      input,
      output,
      op,
      d_init_v,
      segments_per_worker);

    auto h_output = make_host_buffer<value_t>(copy_stream, device, output.size(), cuda::no_init);
    copy_to_host(copy_stream, output, h_output);
    require_ranges_equal(h_expected, h_output);
  }
}

CUB_TEST("Segmented inclusive scan works correctly for integer types",
         "[multi_segment][segmented][scan]",
         CUB_SMALL,
         integral_types)
{
  using value_t  = c2h::get<0, TestType>;
  using op_t     = numeric_op<value_t>;
  using offset_t = unsigned int;

  const auto device = current_device();
  auto copy_stream  = cuda::stream{device};

  // WAR for MSVC which incorrectly deduces that these variables are declared, but unused
  [[maybe_unused]] static constexpr int items_per_thread       = 4;
  [[maybe_unused]] static constexpr int block_size             = 128;
  [[maybe_unused]] static constexpr int max_segments_per_block = 256;

  using policy_t = policy_selector_t<block_size, items_per_thread, max_segments_per_block>;

  const unsigned num_items = block_size * items_per_thread * 132;
  const auto h_offsets     = make_host_buffer<offset_t>(
    copy_stream,
    device,
    cuda::std::initializer_list<offset_t>{0, num_items / 4, num_items / 2, num_items - (num_items / 4), num_items});
  const std::size_t num_segments = h_offsets.size() - 1;

  const auto h_input = make_tabulated_host_buffer<value_t>(copy_stream, device, num_items, init_op<value_t>{});

  auto offsets = make_device_buffer_from_host(copy_stream, device, h_offsets);
  auto input   = make_device_buffer_from_host(copy_stream, device, h_input);
  auto output  = c2h::make_device_buffer<value_t>(copy_stream, device, h_input.size(), cuda::no_init);
  copy_stream.sync();

  auto inclusive_scan_dispatch = [](auto&&... args) {
    dispatch_segmented_scan<
      cub::ForceInclusive::No,
      const value_t*,
      value_t*,
      const offset_t*,
      const offset_t*,
      const offset_t*,
      op_t,
      cub::NullType,
      value_t,
      offset_t,
      policy_t>(std::forward<decltype(args)>(args)...);
  };

  auto h_expected = make_host_buffer<value_t>(copy_stream, device, h_input.size(), cuda::no_init);

  op_t op{};
  value_t h_init{0};

  for (unsigned segment_id = 0; segment_id < num_segments; ++segment_id)
  {
    compute_inclusive_scan_reference(
      h_input.begin() + h_offsets[segment_id],
      h_input.begin() + h_offsets[segment_id + 1], // NOLINT(bugprone-misplaced-widening-cast)
      h_expected.begin() + h_offsets[segment_id],
      op,
      h_init);
  }

  const int segments_per_worker = 4;

  cub::NullType d_no_init{};

  SECTION("worker-block")
  {
    run_dispatch_scan(
      inclusive_scan_dispatch,
      copy_stream,
      cub::detail::segmented_scan::worker::block,
      offsets,
      input,
      output,
      op,
      d_no_init,
      segments_per_worker);

    auto h_output = make_host_buffer<value_t>(copy_stream, device, output.size(), cuda::no_init);
    copy_to_host(copy_stream, output, h_output);
    require_ranges_equal(h_expected, h_output);
  }
}

CUB_TEST("Segmented inclusive scan with init works for integer types",
         "[multi_segment][segmented][scan]",
         CUB_SMALL,
         integral_types)
{
  using value_t  = c2h::get<0, TestType>;
  using op_t     = numeric_op<value_t>;
  using offset_t = unsigned int;

  const auto device = current_device();
  auto copy_stream  = cuda::stream{device};

  constexpr auto max_nelems = get_max_elems<value_t>();

  const unsigned int num_segments      = GENERATE(7, 11, 13, 129);
  const unsigned int items_per_segment = cuda::ceil_div(max_nelems, num_segments);
  const unsigned int num_items         = num_segments * items_per_segment;

  auto h_offsets = make_host_buffer<offset_t>(copy_stream, device, num_segments + 1, cuda::no_init);
  for (unsigned i = 0; i <= num_segments; ++i)
  {
    h_offsets[i] = i * items_per_segment;
  }

  CAPTURE(num_segments, num_items, items_per_segment, cuda::std::is_signed_v<value_t>);

  const auto h_input = make_tabulated_host_buffer<value_t>(copy_stream, device, num_items, init_op<value_t>{});

  auto offsets = make_device_buffer_from_host(copy_stream, device, h_offsets);
  auto input   = make_device_buffer_from_host(copy_stream, device, h_input);
  auto output  = c2h::make_device_buffer<value_t>(copy_stream, device, h_input.size(), cuda::no_init);
  copy_stream.sync();

  // WAR for MSVC which incorrectly deduces that these variables are declared, but unused
  [[maybe_unused]] static constexpr int items_per_thread       = 4;
  [[maybe_unused]] static constexpr int block_size             = 128;
  [[maybe_unused]] static constexpr int max_segments_per_block = 256;

  using policy_t = policy_selector_t<block_size, items_per_thread, max_segments_per_block>;

  using d_init_t                    = cub::detail::InputValue<value_t>;
  auto inclusive_init_scan_dispatch = [](auto&&... args) {
    dispatch_segmented_scan<
      cub::ForceInclusive::Yes,
      const value_t*,
      value_t*,
      const offset_t*,
      const offset_t*,
      const offset_t*,
      op_t,
      d_init_t,
      value_t,
      offset_t,
      policy_t>(std::forward<decltype(args)>(args)...);
  };

  auto h_expected = make_host_buffer<value_t>(copy_stream, device, h_input.size(), cuda::no_init);

  op_t op{};
  value_t h_init{3};

  for (unsigned segment_id = 0; segment_id < num_segments; ++segment_id)
  {
    compute_inclusive_scan_reference(
      h_input.begin() + h_offsets[segment_id],
      h_input.begin() + h_offsets[segment_id + 1], // NOLINT(bugprone-misplaced-widening-cast)
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
    run_dispatch_scan(
      inclusive_init_scan_dispatch,
      copy_stream,
      cub::detail::segmented_scan::worker::block,
      offsets,
      input,
      output,
      op,
      d_init_v,
      segments_per_worker);

    auto h_output = make_host_buffer<value_t>(copy_stream, device, output.size(), cuda::no_init);
    copy_to_host(copy_stream, output, h_output);
    require_ranges_equal(h_expected, h_output);
  }
}

// Given vector of segment sizes (s1, s2, ..., sn), compute input offsets
// (0, s1, s1 + s2, ..., s1 + s2 + ... + sn)
// Similar for out-offsets, except every non-positive element in segment sizes
// is replaced with `gap`.
template <typename OffsetT>
std::tuple<std::vector<OffsetT>, std::vector<OffsetT>>
make_in_out_offsets(const std::vector<OffsetT>& sizes, OffsetT gap)
{
  std::vector<OffsetT> offsets;

  std::size_t segment_count = sizes.size();

  static constexpr OffsetT zero{0};

  offsets.resize(segment_count + 1);
  offsets[0] = zero;

  cuda::std::plus<> plus_t{};

  compute_inclusive_scan_reference(sizes.begin(), sizes.end(), offsets.begin() + 1, plus_t, zero);

  std::vector<OffsetT> sizes_with_gaps;
  sizes_with_gaps.resize(segment_count);
  for (std::size_t i = 0; i < segment_count; ++i)
  {
    const auto s       = sizes[i];
    sizes_with_gaps[i] = (s > 0) ? s : gap;
  }

  std::vector<OffsetT> offsets_with_gaps;
  offsets_with_gaps.resize(segment_count + 1);
  offsets_with_gaps[0] = zero;
  compute_inclusive_scan_reference(
    sizes_with_gaps.begin(), sizes_with_gaps.end(), offsets_with_gaps.begin() + 1, plus_t, zero);

  return {offsets, offsets_with_gaps};
}

CUB_TEST("Segmented inclusive scan skips empty segments", "[multi_segment][segmented][scan]", CUB_SMALL, itp_list)
{
  using op_t     = cuda::std::plus<>;
  using value_t  = unsigned int;
  using offset_t = unsigned int;

  const auto device = current_device();
  auto copy_stream  = cuda::stream{device};

  [[maybe_unused]] static constexpr int items_per_thread       = c2h::get<0, TestType>::value;
  [[maybe_unused]] static constexpr int block_size             = 128;
  [[maybe_unused]] static constexpr int max_segments_per_block = 256;

  using policy_t = policy_selector_t<block_size, items_per_thread, max_segments_per_block>;

  static constexpr auto canary = value_t{0xDEADBEEF};

  const offset_t gap                        = 4;
  const std::vector<offset_t> segment_sizes = {4, 13, 0, 46, 0, 33, 28, 0, 6, 17, 0, 0, 1, 0, 7};

  const auto [in_offsets_v, out_offsets_v] = make_in_out_offsets(segment_sizes, gap);

  const auto num_segments = static_cast<cuda::std::size_t>(segment_sizes.size());
  const auto num_items    = static_cast<cuda::std::size_t>(in_offsets_v.back());

  const auto num_output = static_cast<cuda::std::size_t>(out_offsets_v.back());

  const auto h_input = make_tabulated_host_buffer<value_t>(copy_stream, device, num_items, cuda::std::identity{});

  auto offsets     = make_device_buffer_from_host(copy_stream, device, in_offsets_v);
  auto out_offsets = make_device_buffer_from_host(copy_stream, device, out_offsets_v);
  auto input       = make_device_buffer_from_host(copy_stream, device, h_input);
  auto output      = c2h::make_device_buffer<value_t>(copy_stream, device, num_output, cuda::no_init);
  copy_stream.sync();

  constexpr int segments_per_worker = 2;

  op_t op{};
  value_t h_init_v{0};
  cub::NullType d_no_init{};

  auto h_expected = make_host_buffer<value_t>(copy_stream, device, output.size(), canary);

  for (cuda::std::size_t segment_id = 0; segment_id < num_segments; ++segment_id)
  {
    if (in_offsets_v[segment_id] >= in_offsets_v[segment_id + 1])
    {
      continue;
    }
    compute_inclusive_scan_reference(
      h_input.begin() + in_offsets_v[segment_id],
      h_input.begin() + in_offsets_v[segment_id + 1],
      h_expected.begin() + out_offsets_v[segment_id],
      op,
      h_init_v);
  }

  auto inclusive_scan_dispatch = [](auto&&... args) {
    dispatch_segmented_scan<
      cub::ForceInclusive::No,
      const value_t*,
      value_t*,
      const offset_t*,
      const offset_t*,
      const offset_t*,
      op_t,
      cub::NullType,
      value_t,
      offset_t,
      policy_t>(std::forward<decltype(args)>(args)...);
  };

  SECTION("worker block")
  {
    const auto h_canaries = make_host_buffer<value_t>(copy_stream, device, output.size(), canary);
    enqueue_copy_to_device(copy_stream, h_canaries, output);
    copy_stream.sync();

    run_dispatch_scan(
      inclusive_scan_dispatch,
      copy_stream,
      cub::detail::segmented_scan::worker::block,
      offsets,
      out_offsets,
      input,
      output,
      op,
      d_no_init,
      segments_per_worker);

    auto h_output = make_host_buffer<value_t>(copy_stream, device, output.size(), cuda::no_init);
    copy_to_host(copy_stream, output, h_output);
    require_ranges_equal(h_output, h_expected);
  }
}

CUB_TEST("Segmented inclusive scan handles end_offset < begin_offset", "[multi_segment][segmented][scan]", CUB_SMALL)
{
  using op_t     = cuda::std::plus<>;
  using value_t  = unsigned int;
  using offset_t = int;

  const auto device = current_device();
  auto copy_stream  = cuda::stream{device};

  [[maybe_unused]] static constexpr int items_per_thread       = 7;
  [[maybe_unused]] static constexpr int block_size             = 128;
  [[maybe_unused]] static constexpr int max_segments_per_block = 256;

  using policy_t = policy_selector_t<block_size, items_per_thread, max_segments_per_block>;

  static constexpr auto canary = value_t{0xDEADBEEF};

  const offset_t gap                        = 4;
  const std::vector<offset_t> segment_sizes = {4, 13, -2, 46, -4, 33, 28, -2, 6, 17, 0, -4, 1, 0, 7};

  CAPTURE(segment_sizes, gap);

  const auto [in_offsets_v, out_offsets_v] = make_in_out_offsets(segment_sizes, gap);

  CAPTURE(in_offsets_v, out_offsets_v);

  for (const auto offset : in_offsets_v)
  {
    REQUIRE(offset >= offset_t{0});
  }

  for (const auto offset : out_offsets_v)
  {
    REQUIRE(offset >= offset_t{0});
  }

  const auto num_segments = segment_sizes.size();
  const auto num_items    = static_cast<cuda::std::size_t>(in_offsets_v.back());

  const auto num_output = static_cast<cuda::std::size_t>(out_offsets_v.back());

  const auto h_input = make_tabulated_host_buffer<value_t>(copy_stream, device, num_items, cuda::std::identity{});

  auto offsets     = make_device_buffer_from_host(copy_stream, device, in_offsets_v);
  auto out_offsets = make_device_buffer_from_host(copy_stream, device, out_offsets_v);
  auto input       = make_device_buffer_from_host(copy_stream, device, h_input);
  auto output      = c2h::make_device_buffer<value_t>(copy_stream, device, num_output, cuda::no_init);
  copy_stream.sync();

  constexpr int segments_per_worker = 2;

  op_t op{};
  value_t h_init_v{0};
  cub::NullType d_no_init{};

  auto h_expected = make_host_buffer<value_t>(copy_stream, device, output.size(), canary);

  for (cuda::std::size_t segment_id = 0; segment_id < num_segments; ++segment_id)
  {
    if (in_offsets_v[segment_id] >= in_offsets_v[segment_id + 1])
    {
      continue;
    }
    compute_inclusive_scan_reference(
      h_input.begin() + in_offsets_v[segment_id],
      h_input.begin() + in_offsets_v[segment_id + 1],
      h_expected.begin() + out_offsets_v[segment_id],
      op,
      h_init_v);
  }

  auto inclusive_scan_dispatch = [](auto&&... args) {
    dispatch_segmented_scan<
      cub::ForceInclusive::No,
      const value_t*,
      value_t*,
      const offset_t*,
      const offset_t*,
      const offset_t*,
      op_t,
      cub::NullType,
      value_t,
      offset_t,
      policy_t>(std::forward<decltype(args)>(args)...);
  };

  SECTION("worker block")
  {
    const auto h_canaries = make_host_buffer<value_t>(copy_stream, device, output.size(), canary);
    enqueue_copy_to_device(copy_stream, h_canaries, output);
    copy_stream.sync();

    run_dispatch_scan(
      inclusive_scan_dispatch,
      copy_stream,
      cub::detail::segmented_scan::worker::block,
      offsets,
      out_offsets,
      input,
      output,
      op,
      d_no_init,
      segments_per_worker);

    auto h_output = make_host_buffer<value_t>(copy_stream, device, output.size(), cuda::no_init);
    copy_to_host(copy_stream, output, h_output);
    require_ranges_equal(h_output, h_expected);
  }
}

template <typename DispatchT,
          typename OffsetT,
          typename InputIterT,
          typename OutputIterT,
          typename ScanOpT,
          typename InitValueT>
void run_dispatch_scan_iterator(
  DispatchT dispatch_fn,
  [[maybe_unused]] cuda::stream_ref stream,
  cub::detail::segmented_scan::worker worker_choice,
  const cuda::device_buffer<OffsetT>& offsets,
  InputIterT input_it,
  OutputIterT output_it,
  ScanOpT scan_op,
  InitValueT init_value,
  int segments_per_worker)
{
  const auto n_segments = static_cast<OffsetT>(offsets.size() - 1);

  const auto d_input   = input_it;
  const auto d_output  = output_it;
  const auto d_offsets = offsets.data();

  dispatch_fn(
    d_input,
    d_output,
    n_segments,
    d_offsets,
    d_offsets + 1,
    d_offsets,
    scan_op,
    init_value,
    segments_per_worker,
    worker_choice
#if TEST_LAUNCH == 0
    ,
    stream.get()
#elif TEST_LAUNCH == 1
    ,
    nullptr // Host stream handles are invalid for device-side launches.
#endif // TEST_LAUNCH == 0 / TEST_LAUNCH == 1
  );
}

CUB_TEST("segmented inclusive scan works correctly with fancy iterators", "[multi_segment][segmented][scan]", CUB_SMALL)
{
  using op_t     = cuda::std::plus<>;
  using value_t  = unsigned int;
  using offset_t = unsigned int;

  const auto device = current_device();
  auto copy_stream  = cuda::stream{device};

  [[maybe_unused]] static constexpr int items_per_thread       = 9;
  [[maybe_unused]] static constexpr int block_size             = 128;
  [[maybe_unused]] static constexpr int max_segments_per_block = 256;

  using policy_t = policy_selector_t<block_size, items_per_thread, max_segments_per_block>;

  constexpr auto max_nelems = get_max_elems<value_t>();

  const unsigned int num_segments      = 255;
  const unsigned int items_per_segment = cuda::ceil_div(max_nelems, num_segments);
  const unsigned int num_items         = num_segments * items_per_segment;

  auto h_offsets = make_host_buffer<offset_t>(copy_stream, device, num_segments + 1, cuda::no_init);
  for (unsigned i = 0; i <= num_segments; ++i)
  {
    h_offsets[i] = i * items_per_segment;
  }

  CAPTURE(num_segments, num_items, items_per_segment, cuda::std::is_signed_v<value_t>);

  auto offsets = make_device_buffer_from_host(copy_stream, device, h_offsets);

  const auto input_it = cuda::make_transform_iterator(cuda::counting_iterator<value_t>(0), init_op<value_t>{});

  auto output    = c2h::make_device_buffer<value_t>(copy_stream, device, num_items, cuda::no_init);
  auto output_it = cuda::make_transform_output_iterator(output.data(), init_op<value_t>{});
  copy_stream.sync();

  using input_it_t  = decltype(input_it);
  using output_it_t = decltype(output_it);

  using d_init_t                    = cub::detail::InputValue<value_t>;
  auto inclusive_init_scan_dispatch = [](auto&&... args) {
    dispatch_segmented_scan<
      cub::ForceInclusive::Yes,
      input_it_t,
      output_it_t,
      const offset_t*,
      const offset_t*,
      const offset_t*,
      op_t,
      d_init_t,
      value_t,
      offset_t,
      policy_t>(std::forward<decltype(args)>(args)...);
  };

  const auto h_input = make_tabulated_host_buffer<value_t>(copy_stream, device, num_items, init_op<value_t>{});
  auto h_expected    = make_host_buffer<value_t>(copy_stream, device, num_items, cuda::no_init);

  op_t op{};
  value_t h_init{3};

  for (unsigned segment_id = 0; segment_id < num_segments; ++segment_id)
  {
    compute_inclusive_scan_reference(
      h_input.begin() + h_offsets[segment_id],
      h_input.begin() + h_offsets[segment_id + 1], // NOLINT(bugprone-misplaced-widening-cast)
      h_expected.begin() + h_offsets[segment_id],
      op,
      h_init);

    for (offset_t offset = h_offsets[segment_id];
         offset < h_offsets[segment_id + 1]; // NOLINT(bugprone-misplaced-widening-cast)
         ++offset)
    {
      h_expected[offset] = init_op<value_t>{}(h_expected[offset]);
    }
  }

  d_init_t d_init_v{h_init};
  const int segments_per_worker = 2;

  SECTION("worker block")
  {
    run_dispatch_scan_iterator(
      inclusive_init_scan_dispatch,
      copy_stream,
      cub::detail::segmented_scan::worker::block,
      offsets,
      input_it,
      output_it,
      op,
      d_init_v,
      segments_per_worker);

    auto h_output = make_host_buffer<value_t>(copy_stream, device, output.size(), cuda::no_init);
    copy_to_host(copy_stream, output, h_output);
    require_ranges_equal(h_expected, h_output);
  }
}
