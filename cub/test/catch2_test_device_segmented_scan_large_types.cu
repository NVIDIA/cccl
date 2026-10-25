// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Should precede any includes
struct stream_registry_factory_t;
#define CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY stream_registry_factory_t

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_segmented_scan.cuh>

#include <cuda/__execution/tune.h>
#include <cuda/std/algorithm>
#include <cuda/std/cstdint>
#include <cuda/std/execution>
#include <cuda/std/functional>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <vector>

#include "catch2_test_device_segmented_scan_schedules.cuh"
#include "catch2_test_launch_helper.h"
#include "cub_test_macros.h"

DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceSegmentedScan::InclusiveSegmentedScan, device_segmented_inclusive_scan);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceSegmentedScan::ExclusiveSegmentedScan, device_segmented_exclusive_scan);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceSegmentedScan::InclusiveSegmentedScanInit, device_segmented_inclusive_scan_init);

// %PARAM% TEST_LAUNCH lid 0:1:2
// %PARAM% TEST_TYPES types 0:1:2

namespace stdexec = cuda::std::execution;

namespace
{
// Unsigned words added elementwise, so sums wrap and the host reference is exact.
template <int Words>
struct u32_array
{
  cuda::std::uint32_t v[Words]{};

  __host__ __device__ friend u32_array operator+(const u32_array& lhs, const u32_array& rhs)
  {
    u32_array result;
    for (int word = 0; word < Words; ++word)
    {
      result.v[word] = lhs.v[word] + rhs.v[word];
    }
    return result;
  }

  friend bool operator==(const u32_array& lhs, const u32_array& rhs)
  {
    for (int word = 0; word < Words; ++word)
    {
      if (lhs.v[word] != rhs.v[word])
      {
        return false;
      }
    }
    return true;
  }
};

template <typename ValueT>
ValueT filled_value(cuda::std::uint32_t seed, cuda::std::uint32_t step)
{
  ValueT result;
  for (int word = 0; word < static_cast<int>(sizeof(ValueT) / sizeof(cuda::std::uint32_t)); ++word)
  {
    const cuda::std::uint32_t mixed = seed * 0x9e3779b1u + static_cast<cuda::std::uint32_t>(word) * step;
    result.v[word]                  = mixed ^ (mixed >> 15);
  }
  return result;
}

// The value of unwritten output items: every bit set, as cudaMemset with 0xff leaves them.
template <typename ValueT>
ValueT canary()
{
  ValueT result;
  for (auto& word : result.v)
  {
    word = ~cuda::std::uint32_t{0};
  }
  return result;
}

enum class arrangement
{
  back_to_back,
  gaps,
  overlapping
};

enum class scan_form
{
  inclusive,
  exclusive,
  inclusive_init
};

template <scan_form... Forms>
struct scan_forms
{};

template <typename OffsetT>
struct segment_layout
{
  c2h::host_vector<OffsetT> begins;
  c2h::host_vector<OffsetT> ends;
  c2h::host_vector<OffsetT> out_begins;
  size_t input_size;
  size_t output_size;
  cuda::std::int64_t num_items;
};

// Segments of about target_items items: empty ones, single items, sizes around a tile, and one of half the items that
// spans many blocks. The odd total leaves a partial last tile. Except back to back, the segments lie in the input in
// reverse order with a gap after each, and one empty segment ends before it begins. Overlapping adds segments that read
// a gap and part of another segment, part of the long segment, and the same items as another segment. The output
// begin offsets place every segment in order with a gap after each.
template <typename OffsetT>
segment_layout<OffsetT>
make_layout(cuda::std::int64_t target_items, cuda::std::int64_t tile_items, arrangement segment_arrangement)
{
  const cuda::std::int64_t pattern[] = {0, 1, tile_items - 1, tile_items, tile_items + 1, 2, 0, 3 * tile_items + 5, 17};
  constexpr size_t pattern_size      = sizeof(pattern) / sizeof(pattern[0]);
  constexpr size_t long_segment      = 3;
  constexpr size_t tile_segment      = 7;

  std::vector<cuda::std::int64_t> sizes{5, 0, 1, target_items / 2};
  cuda::std::int64_t total            = 1 + 5 + target_items / 2;
  const cuda::std::int64_t base_total = target_items - 37;
  for (size_t index = 0; total < base_total; ++index)
  {
    const cuda::std::int64_t size = (cuda::std::min) (pattern[index % pattern_size], base_total - total);
    sizes.push_back(size);
    total += size;
  }
  REQUIRE(sizes.size() > tile_segment);

  const bool back_to_back                 = segment_arrangement == arrangement::back_to_back;
  const cuda::std::int64_t input_gap      = back_to_back ? 0 : 3;
  constexpr cuda::std::int64_t output_gap = 5;
  const size_t base_segments              = sizes.size();
  std::vector<cuda::std::int64_t> begins(base_segments);
  std::vector<cuda::std::int64_t> ends(base_segments);
  cuda::std::int64_t input_end = input_gap;
  for (size_t place = 0; place < base_segments; ++place)
  {
    const size_t segment = back_to_back ? place : base_segments - 1 - place;
    begins[segment]      = input_end;
    ends[segment]        = input_end + sizes[segment];
    input_end += sizes[segment] + input_gap;
  }
  if (!back_to_back)
  {
    ends[1] = begins[1] - 1;
  }
  if (segment_arrangement == arrangement::overlapping)
  {
    const auto add = [&](cuda::std::int64_t begin, cuda::std::int64_t size) {
      begins.push_back(begin);
      ends.push_back(begin + size);
    };
    add(begins[long_segment] - 2, tile_items + 7);
    add(begins[long_segment] + tile_items / 2, 2 * tile_items + 3);
    add(begins[tile_segment], sizes[tile_segment]);
  }

  segment_layout<OffsetT> layout{};
  const size_t num_segments = begins.size();
  layout.begins.resize(num_segments);
  layout.ends.resize(num_segments);
  layout.out_begins.resize(num_segments);
  cuda::std::int64_t output_end = output_gap;
  for (size_t segment = 0; segment < num_segments; ++segment)
  {
    layout.begins[segment]        = static_cast<OffsetT>(begins[segment]);
    layout.ends[segment]          = static_cast<OffsetT>(ends[segment]);
    layout.out_begins[segment]    = static_cast<OffsetT>(output_end);
    const cuda::std::int64_t size = (cuda::std::max) (cuda::std::int64_t{0}, ends[segment] - begins[segment]);
    output_end += size + output_gap;
    layout.num_items += size;
  }
  layout.input_size  = static_cast<size_t>(input_end);
  layout.output_size = static_cast<size_t>(output_end);
  return layout;
}

template <typename ValueT, typename OffsetT>
c2h::host_vector<ValueT> reference_scan(
  scan_form form,
  const segment_layout<OffsetT>& layout,
  bool separate_output,
  const c2h::host_vector<ValueT>& h_in,
  const ValueT& init)
{
  const auto& dest_begins = separate_output ? layout.out_begins : layout.begins;
  c2h::host_vector<ValueT> expected(separate_output ? layout.output_size : layout.input_size, canary<ValueT>());
  for (size_t segment = 0; segment < layout.begins.size(); ++segment)
  {
    const cuda::std::int64_t begin = layout.begins[segment];
    const cuda::std::int64_t dest  = dest_begins[segment];
    ValueT prefix                  = init;
    for (cuda::std::int64_t item = 0; begin + item < layout.ends[segment]; ++item)
    {
      const ValueT& value = h_in[begin + item];
      ValueT& out         = expected[dest + item];
      if (form == scan_form::exclusive)
      {
        out    = prefix;
        prefix = prefix + value;
      }
      else
      {
        prefix = (item == 0 && form == scan_form::inclusive) ? value : prefix + value;
        out    = prefix;
      }
    }
  }
  return expected;
}

template <typename ValueT>
size_t first_mismatch(const c2h::host_vector<ValueT>& actual, const c2h::host_vector<ValueT>& expected)
{
  for (size_t index = 0; index < expected.size(); ++index)
  {
    if (!(actual[index] == expected[index]))
    {
      return index;
    }
  }
  return expected.size();
}

template <typename ValueT, typename OffsetT>
struct scan_problem
{
  const ValueT* d_in;
  const OffsetT* d_begins;
  const OffsetT* d_ends;
  const OffsetT* d_out_begins;
  cuda::std::int64_t num_segments;
  ValueT init;
};

template <scan_form Form, bool SeparateOutput, typename ValueT, typename OffsetT, typename ScheduleEnvT>
void check_scan(const ScheduleEnvT& schedule_env,
                const scan_problem<ValueT, OffsetT>& problem,
                c2h::device_vector<ValueT>& d_out,
                const c2h::host_vector<ValueT>& expected)
{
  constexpr auto enforce_inclusive =
    Form == scan_form::inclusive_init ? cub::ForceInclusive::Yes : cub::ForceInclusive::No;
  ValueT* d_out_ptr            = thrust::raw_pointer_cast(d_out.data());
  const OffsetT* d_dest_begins = SeparateOutput ? problem.d_out_begins : problem.d_begins;
  const auto scan_op           = cuda::std::plus<>{};

  size_t bytes{};
  if constexpr (Form == scan_form::inclusive)
  {
    bytes = schedule_allocation_size<enforce_inclusive, !SeparateOutput>(
      schedule_env,
      problem.d_in,
      d_out_ptr,
      problem.num_segments,
      problem.d_begins,
      problem.d_ends,
      d_dest_begins,
      scan_op,
      cub::NullType{});
  }
  else
  {
    bytes = schedule_allocation_size<enforce_inclusive, !SeparateOutput>(
      schedule_env,
      problem.d_in,
      d_out_ptr,
      problem.num_segments,
      problem.d_begins,
      problem.d_ends,
      d_dest_begins,
      scan_op,
      cub::detail::InputValue<ValueT>{problem.init});
  }

  REQUIRE(cudaSuccess == cudaMemset(d_out_ptr, 0xff, d_out.size() * sizeof(ValueT)));
  const auto env = stdexec::env{expected_allocation_size(bytes), schedule_env};
  if constexpr (Form == scan_form::inclusive && SeparateOutput)
  {
    device_segmented_inclusive_scan(
      problem.d_in,
      d_out_ptr,
      problem.d_begins,
      problem.d_ends,
      problem.d_out_begins,
      problem.num_segments,
      scan_op,
      env);
  }
  else if constexpr (Form == scan_form::inclusive)
  {
    device_segmented_inclusive_scan(
      problem.d_in, d_out_ptr, problem.d_begins, problem.d_ends, problem.num_segments, scan_op, env);
  }
  else if constexpr (Form == scan_form::exclusive && SeparateOutput)
  {
    device_segmented_exclusive_scan(
      problem.d_in,
      d_out_ptr,
      problem.d_begins,
      problem.d_ends,
      problem.d_out_begins,
      problem.num_segments,
      scan_op,
      problem.init,
      env);
  }
  else if constexpr (Form == scan_form::exclusive)
  {
    device_segmented_exclusive_scan(
      problem.d_in, d_out_ptr, problem.d_begins, problem.d_ends, problem.num_segments, scan_op, problem.init, env);
  }
  else if constexpr (SeparateOutput)
  {
    device_segmented_inclusive_scan_init(
      problem.d_in,
      d_out_ptr,
      problem.d_begins,
      problem.d_ends,
      problem.d_out_begins,
      problem.num_segments,
      scan_op,
      problem.init,
      env);
  }
  else
  {
    device_segmented_inclusive_scan_init(
      problem.d_in, d_out_ptr, problem.d_begins, problem.d_ends, problem.num_segments, scan_op, problem.init, env);
  }

  const c2h::host_vector<ValueT> h_out = d_out;
  REQUIRE(first_mismatch(h_out, expected) == expected.size());
}

template <scan_form Form, bool SeparateOutput, typename ValueT, typename OffsetT, typename ScheduleEnvsT, size_t... Schedules>
void check_form(const segment_layout<OffsetT>& layout,
                const c2h::host_vector<ValueT>& h_in,
                const scan_problem<ValueT, OffsetT>& problem,
                const ScheduleEnvsT& schedule_envs,
                cuda::std::index_sequence<Schedules...>)
{
  const c2h::host_vector<ValueT> expected = reference_scan(Form, layout, SeparateOutput, h_in, problem.init);
  // Copied from the host, since a device fill would pass the large value as a kernel parameter.
  c2h::device_vector<ValueT> d_out = expected;
  (check_scan<Form, SeparateOutput>(cuda::std::get<Schedules>(schedule_envs), problem, d_out, expected), ...);
}

template <bool SeparateOutput, typename ValueT, typename OffsetT, typename ScheduleEnvsT, scan_form... Forms>
void check_forms(const segment_layout<OffsetT>& layout,
                 const c2h::host_vector<ValueT>& h_in,
                 const scan_problem<ValueT, OffsetT>& problem,
                 const ScheduleEnvsT& schedule_envs,
                 scan_forms<Forms...>)
{
  constexpr auto schedules = cuda::std::make_index_sequence<cuda::std::tuple_size_v<ScheduleEnvsT>>{};
  (check_form<Forms, SeparateOutput>(layout, h_in, problem, schedule_envs, schedules), ...);
}

// Scans every arrangement in each of the forms under each schedule environment in the tuple that make_schedule_envs
// returns for the layout's item count.
template <typename ValueT, typename OffsetT, typename MakeScheduleEnvsT, typename FormsT>
void check_large_accumulator(
  cuda::std::int64_t target_items, cuda::std::int64_t tile_items, MakeScheduleEnvsT make_schedule_envs, FormsT forms)
{
  for (const auto segment_arrangement : {arrangement::back_to_back, arrangement::gaps, arrangement::overlapping})
  {
    const segment_layout<OffsetT> layout = make_layout<OffsetT>(target_items, tile_items, segment_arrangement);

    c2h::host_vector<ValueT> h_in(layout.input_size);
    for (size_t item = 0; item < h_in.size(); ++item)
    {
      h_in[item] = filled_value<ValueT>(static_cast<cuda::std::uint32_t>(item), 0x85ebca77u);
    }
    const c2h::device_vector<ValueT> d_in          = h_in;
    const c2h::device_vector<OffsetT> d_begins     = layout.begins;
    const c2h::device_vector<OffsetT> d_ends       = layout.ends;
    const c2h::device_vector<OffsetT> d_out_begins = layout.out_begins;

    const scan_problem<ValueT, OffsetT> problem{
      thrust::raw_pointer_cast(d_in.data()),
      thrust::raw_pointer_cast(d_begins.data()),
      thrust::raw_pointer_cast(d_ends.data()),
      thrust::raw_pointer_cast(d_out_begins.data()),
      static_cast<cuda::std::int64_t>(layout.begins.size()),
      filled_value<ValueT>(0x2545f491u, 0x01000193u)};

    const auto schedule_envs = make_schedule_envs(layout.num_items);
    if (segment_arrangement == arrangement::overlapping)
    {
      check_forms<true>(layout, h_in, problem, schedule_envs, forms);
    }
    else
    {
      check_forms<false>(layout, h_in, problem, schedule_envs, forms);
    }
  }
}

// Whether the load-balanced kernel holds ValueT under the default policy, for raw pointers, separate output begin
// offsets when SeparateOutput is set, an initial value of type InitValueT (cub::NullType for none) and work of type
// WorkT.
template <typename ValueT,
          typename OffsetT,
          bool SeparateOutput,
          typename InitValueT = cub::NullType,
          typename WorkT      = cub::detail::segmented_scan::load_balanced_work_t<OffsetT, false>>
inline constexpr bool load_balanced_kernel_holds = cub::detail::segmented_scan::default_load_balanced_policy_fits<
  const ValueT*,
  ValueT*,
  OffsetT,
  WorkT,
  cuda::std::plus<>,
  InitValueT,
  ValueT,
  false,
  cub::detail::segmented_scan::segment_addressing<
    const OffsetT*,
    cuda::std::conditional_t<SeparateOutput, const OffsetT*, cub::detail::segmented_scan::reuse_input_begin>>>;

#if TEST_TYPES != 2
// The default schedule, and load balancing alone and with num_items.
struct untuned_schedules
{
  auto operator()(cuda::std::int64_t num_items) const
  {
    return cuda::std::make_tuple(
      stdexec::env{},
      stdexec::env{cub::segmented_scan_load_balancing},
      stdexec::env{cub::segmented_scan_num_items(num_items), cub::segmented_scan_load_balancing});
  }
};

template <typename ValueT>
constexpr cuda::std::int64_t default_load_balanced_tile_items()
{
  constexpr auto policy =
    cub::detail::segmented_scan::load_balanced_policy_selector_from_types<ValueT>{}(cuda::compute_capability{});
  return cuda::std::int64_t{policy.threads_per_block} * policy.items_per_thread;
}

// Static shared memory of the load-balanced kernel under the policy PolicyGetter returns, by default CUB's, for a scan
// with an initial value, for raw pointers and output begin offsets that are the input's.
template <typename ValueT,
          typename OffsetT,
          bool ForceInclusive,
          typename PolicyGetter = cub::detail::segmented_scan::default_load_balanced_policy_getter<ValueT>>
inline constexpr size_t load_balanced_kernel_smem = cub::detail::segmented_scan::load_balanced_kernel_smem_bytes<
  PolicyGetter,
  const ValueT*,
  ValueT*,
  OffsetT,
  cub::detail::segmented_scan::load_balanced_work_t<OffsetT, false>,
  cuda::std::plus<>,
  ValueT,
  ValueT,
  ForceInclusive,
  cub::detail::segmented_scan::segment_addressing<const OffsetT*, cub::detail::segmented_scan::reuse_input_begin>>;

// Whether the load-balanced kernel's exclusive scan keeps one tail per thread under the policy PolicyGetter returns, by
// default CUB's. Its storage then exceeds that of the inclusive scan with an initial value, which keeps one tail.
template <typename ValueT,
          typename OffsetT,
          typename PolicyGetter = cub::detail::segmented_scan::default_load_balanced_policy_getter<ValueT>>
inline constexpr bool exclusive_scan_keeps_thread_tails =
  (load_balanced_kernel_smem<ValueT,
                             OffsetT,
                             false,
                             PolicyGetter> > load_balanced_kernel_smem<ValueT, OffsetT, true, PolicyGetter>);
#endif // TEST_TYPES != 2
} // namespace

// Tuned launches are checked from the host and under graph capture only, as in the suite's tuning tests.
#if TEST_LAUNCH != 1
template <typename TuningT>
struct tuned_load_balanced_schedules
{
  auto operator()(cuda::std::int64_t num_items) const
  {
    const auto tuning = cuda::execution::tune(TuningT{});
    return cuda::std::make_tuple(
      stdexec::env{cub::segmented_scan_load_balancing, tuning},
      stdexec::env{cub::segmented_scan_num_items(num_items), cub::segmented_scan_load_balancing, tuning});
  }
};

// The default load-balanced policy's threads per block for a 4-byte-aligned accumulator of AccumBytes bytes, without
// instantiating a type of that size
template <int AccumBytes>
inline constexpr int default_load_balanced_threads =
  cub::detail::segmented_scan::load_balanced_policy_selector{static_cast<int>(sizeof(int)) + AccumBytes}(
    cuda::compute_capability{})
    .threads_per_block;
#endif // TEST_LAUNCH != 1

#if TEST_TYPES == 0

using offset_types = c2h::type_list<cuda::std::int32_t, cuda::std::int64_t>;

CUB_TEST("Device segmented scan scans a 64-byte accumulator in every form",
         "[segmented][scan][device]",
         CUB_SMALL,
         offset_types)
{
  using offset_t = c2h::get<0, TestType>;
  using value_t  = u32_array<16>;
  STATIC_REQUIRE(load_balanced_kernel_holds<value_t, offset_t, false>);
  STATIC_REQUIRE(load_balanced_kernel_holds<value_t, offset_t, true>);
  STATIC_REQUIRE(exclusive_scan_keeps_thread_tails<value_t, offset_t>);

  check_large_accumulator<value_t, offset_t>(
    cuda::std::int64_t{1} << 14,
    default_load_balanced_tile_items<value_t>(),
    untuned_schedules{},
    scan_forms<scan_form::inclusive, scan_form::exclusive, scan_form::inclusive_init>{});
}

#endif // TEST_TYPES == 0

#if TEST_TYPES == 1

CUB_TEST("Device segmented scan scans a 384-byte accumulator exclusively with one tail per block",
         "[segmented][scan][device]",
         CUB_SMALL)
{
  // Past the size at which one tail per thread fits the shared memory of the default policy's tile
  using value_t = u32_array<96>;
  STATIC_REQUIRE(load_balanced_kernel_holds<value_t, cuda::std::int32_t, false, value_t>);
  STATIC_REQUIRE(load_balanced_kernel_holds<value_t, cuda::std::int32_t, true, value_t>);
  STATIC_REQUIRE(!exclusive_scan_keeps_thread_tails<value_t, cuda::std::int32_t>);

  check_large_accumulator<value_t, cuda::std::int32_t>(
    cuda::std::int64_t{1} << 14,
    default_load_balanced_tile_items<value_t>(),
    untuned_schedules{},
    scan_forms<scan_form::exclusive>{});
}

#  if TEST_LAUNCH != 1

// With one warp, the look-back storage is larger than the block scan's.
struct one_warp_tuning
{
  _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability) const
    -> cub::SegmentedScanLoadBalancedPolicy
  {
    return cub::SegmentedScanLoadBalancedPolicy{
      32, 1, cub::BLOCK_LOAD_DIRECT, cub::LOAD_DEFAULT, cub::BLOCK_STORE_DIRECT, cub::BLOCK_SCAN_WARP_SCANS};
  }
};

CUB_TEST("Device segmented scan with load balancing scans a 64-byte accumulator with one warp per block",
         "[segmented][scan][device]",
         CUB_SMALL)
{
  // The default policy's shape for a 2,048-byte accumulator
  STATIC_REQUIRE(default_load_balanced_threads<2048> == 32);

  check_large_accumulator<u32_array<16>, cuda::std::int32_t>(
    cuda::std::int64_t{1} << 14,
    32,
    tuned_load_balanced_schedules<one_warp_tuning>{},
    scan_forms<scan_form::inclusive, scan_form::exclusive>{});
}

// Two items per thread over 256 threads.
struct two_item_tuning
{
  _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability) const
    -> cub::SegmentedScanLoadBalancedPolicy
  {
    return cub::SegmentedScanLoadBalancedPolicy{
      256, 2, cub::BLOCK_LOAD_DIRECT, cub::LOAD_DEFAULT, cub::BLOCK_STORE_DIRECT, cub::BLOCK_SCAN_WARP_SCANS};
  }
};

struct two_item_policy_getter
{
  _CCCL_HOST_DEVICE_API constexpr auto operator()() const
  {
    return two_item_tuning{}(::cuda::compute_capability{});
  }
};

CUB_TEST("Device segmented scan with load balancing scans a 192-byte accumulator exclusively with two items per "
         "thread and one tail per block",
         "[segmented][scan][device]",
         CUB_SMALL)
{
  // One tail per thread would take 256 pairs of 196 bytes, past the shared memory limit.
  using value_t = u32_array<48>;
  STATIC_REQUIRE(!exclusive_scan_keeps_thread_tails<value_t, cuda::std::int32_t, two_item_policy_getter>);

  check_large_accumulator<value_t, cuda::std::int32_t>(
    cuda::std::int64_t{1} << 14,
    512,
    tuned_load_balanced_schedules<two_item_tuning>{},
    scan_forms<scan_form::exclusive>{});
}

#  endif // TEST_LAUNCH != 1

#endif // TEST_TYPES == 1

#if TEST_TYPES == 2

// Checks the routing at compile time only, so no kernel is instantiated for the type.
CUB_TEST("Device segmented scan with load balancing routes an accumulator too large for the load-balanced kernel",
         "[segmented][scan][device]",
         CUB_SMALL)
{
  // The size of u32_array<3072>, without operators that would go unreferenced
  struct value_t
  {
    cuda::std::uint32_t v[3072];
  };
  // Load balancing takes one block per segment instead.
  STATIC_REQUIRE(!load_balanced_kernel_holds<value_t, cuda::std::int32_t, false>);
  STATIC_REQUIRE(!load_balanced_kernel_holds<value_t, cuda::std::int32_t, true>);
}

// Checks the routing at compile time only, so no kernel is instantiated for the type.
CUB_TEST("Device segmented scan with load balancing routes an accumulator that only 32-bit work holds",
         "[segmented][scan][device]",
         CUB_SMALL)
{
  // The size of u32_array<3040>, without operators that would go unreferenced
  struct value_t
  {
    cuda::std::uint32_t v[3040];
  };
  // With separate output begin offsets, a supplied total of 2^32 or more takes one block per segment and a smaller one
  // is load balanced.
  STATIC_REQUIRE(load_balanced_kernel_holds<value_t, cuda::std::int32_t, true>);
  STATIC_REQUIRE(
    !load_balanced_kernel_holds<value_t,
                                cuda::std::int32_t,
                                true,
                                cub::NullType,
                                cub::detail::segmented_scan::wide_load_balanced_work_t>);
}

#  if TEST_LAUNCH != 1

// The block exchange of a warp-transpose load or store holds the whole tile.
struct warp_transpose_tuning
{
  _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability) const
    -> cub::SegmentedScanLoadBalancedPolicy
  {
    return cub::SegmentedScanLoadBalancedPolicy{
      96,
      1,
      cub::BLOCK_LOAD_WARP_TRANSPOSE,
      cub::LOAD_DEFAULT,
      cub::BLOCK_STORE_WARP_TRANSPOSE,
      cub::BLOCK_SCAN_WARP_SCANS};
  }
};

CUB_TEST("Device segmented scan with load balancing scans a 64-byte accumulator with a 96-thread warp-transpose policy",
         "[segmented][scan][device]",
         CUB_SMALL)
{
  // The default policy's block size for a 512-byte accumulator
  STATIC_REQUIRE(default_load_balanced_threads<512> == 96);

  check_large_accumulator<u32_array<16>, cuda::std::int32_t>(
    cuda::std::int64_t{1} << 14,
    96,
    tuned_load_balanced_schedules<warp_transpose_tuning>{},
    scan_forms<scan_form::inclusive, scan_form::exclusive>{});
}

#  endif // TEST_LAUNCH != 1

#endif // TEST_TYPES == 2
