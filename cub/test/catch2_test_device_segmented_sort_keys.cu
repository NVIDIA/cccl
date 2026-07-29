// SPDX-FileCopyrightText: Copyright (c) 2011-2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_segmented_sort.cuh>
#include <cub/util_type.cuh>

#include <cuda/iterator>

#include "catch2_radix_sort_helper.cuh"
#include "catch2_segmented_sort_helper.cuh"
#include <c2h/bfloat16.cuh>
#include <c2h/catch2_test_helper.h>
#include <c2h/half.cuh>

// FIXME: Graph launch disabled, algorithm syncs internally. WAR exists for device-launch, figure out how to enable for
// graph launch.
// %PARAM% TEST_LAUNCH lid 0:1
// %PARAM% TEST_TYPES types 0:1

DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedSort::StableSortKeys, stable_sort_keys);

#if TEST_TYPES == 0
using key_types = c2h::type_list<bool, std::uint8_t>;
#elif TEST_TYPES == 1
using key_types =
  c2h::type_list<std::uint64_t
#  if TEST_HALF_T()
                 ,
                 half_t
#  endif // TEST_HALF_T()
#  if TEST_BF_T()
                 ,
                 bfloat16_t
#  endif // TEST_BF_T()
                 >;
#endif

#if TEST_TYPES == 0

C2H_TEST("DeviceSegmentedSortKeys: No segments", "[keys][segmented][sort][device]")
{
  // Type doesn't affect the escape logic, so it should be fine
  // to test only one set of types here.

  using KeyT = std::uint8_t;

  const bool stable_sort     = GENERATE(unstable, stable);
  const bool sort_descending = GENERATE(ascending, descending);
  const bool sort_buffer     = GENERATE(pointers, double_buffer);

  cub::DoubleBuffer<KeyT> keys_buffer(nullptr, nullptr);
  cub::DoubleBuffer<cub::NullType> values_buffer(nullptr, nullptr);
  values_buffer.selector = 1;

  call_cub_segmented_sort_api(
    sort_descending,
    sort_buffer,
    stable_sort,
    static_cast<KeyT*>(nullptr),
    static_cast<KeyT*>(nullptr),
    static_cast<cub::NullType*>(nullptr),
    static_cast<cub::NullType*>(nullptr),
    int{},
    int{},
    nullptr,
    &keys_buffer.selector,
    &values_buffer.selector);

  REQUIRE(keys_buffer.selector == 0);
  REQUIRE(values_buffer.selector == 1);
}

C2H_TEST("DeviceSegmentedSortKeys: Empty segments", "[keys][segmented][sort][device]")
{
  // Type doesn't affect the escape logic, so it should be fine
  // to test only one set of types here.

  using KeyT = std::uint8_t;

  const int num_segments     = GENERATE(take(2, random(1 << 2, 1 << 22)));
  const bool sort_stable     = GENERATE(unstable, stable);
  const bool sort_descending = GENERATE(ascending, descending);
  const bool sort_buffer     = GENERATE(pointers, double_buffer);

  c2h::device_vector<int> offsets(num_segments + 1, int{});
  const int* d_offsets = thrust::raw_pointer_cast(offsets.data());

  cub::DoubleBuffer<KeyT> keys_buffer(nullptr, nullptr);
  cub::DoubleBuffer<cub::NullType> values_buffer(nullptr, nullptr);
  values_buffer.selector = 1;

  call_cub_segmented_sort_api(
    sort_descending,
    sort_buffer,
    sort_stable,
    static_cast<KeyT*>(nullptr),
    static_cast<KeyT*>(nullptr),
    static_cast<cub::NullType*>(nullptr),
    static_cast<cub::NullType*>(nullptr),
    int{},
    num_segments,
    d_offsets,
    &keys_buffer.selector,
    &values_buffer.selector);

  REQUIRE(keys_buffer.selector == 0);
  REQUIRE(values_buffer.selector == 1);
}

#endif // TEST_TYPES == 0

C2H_TEST("DeviceSegmentedSortKeys: Same size segments, derived keys",
         "[keys][segmented][sort][device][skip-cs-racecheck]",
         key_types)
{
  using KeyT = c2h::get<0, TestType>;

  // Use adjust_seed_count to limit the number of passes run during sanitizer tests.
  const int segment_size = GENERATE_COPY(
    take(c2h::adjust_seed_count(2), random(1 << 0, 1 << 5)), //
    take(c2h::adjust_seed_count(2), random(1 << 5, 1 << 10)),
    take(c2h::adjust_seed_count(2), random(1 << 10, 1 << 15)));

  const int segments = GENERATE_COPY( //
    take(c2h::adjust_seed_count(2), random(1 << 0, 1 << 5)), //
    take(c2h::adjust_seed_count(2), random(1 << 5, 1 << 10)));

  test_same_size_segments_derived<KeyT>(segment_size, segments);
}

C2H_TEST("DeviceSegmentedSortKeys: Randomly sized segments, derived keys",
         "[keys][segmented][sort][device][skip-cs-racecheck]",
         key_types)
{
  using KeyT = c2h::get<0, TestType>;

  const int max_items   = 1 << 22;
  const int max_segment = 6000;

  // Use adjust_seed_count to limit the number of passes run during sanitizer tests.
  const int segments = GENERATE_COPY(
    take(c2h::adjust_seed_count(2), random(1 << 0, 1 << 5)), //
    take(c2h::adjust_seed_count(2), random(1 << 5, 1 << 10)),
    take(c2h::adjust_seed_count(2), random(1 << 10, 1 << 15)),
    take(c2h::adjust_seed_count(2), random(1 << 15, 1 << 20)));

  test_random_size_segments_derived<KeyT>(C2H_SEED(1), max_items, max_segment, segments);
}

C2H_TEST("DeviceSegmentedSortKeys: Randomly sized segments, random keys",
         "[keys][segmented][sort][device][skip-cs-initcheck][skip-cs-racecheck]",
         key_types)
{
  using KeyT = c2h::get<0, TestType>;

  const int max_items   = 1 << 22;
  const int max_segment = 6000;

  // Use adjust_seed_count to limit the number of passes run during sanitizer tests.
  const int segments = GENERATE_COPY(take(c2h::adjust_seed_count(2), random(1 << 15, 1 << 20)));

  test_random_size_segments_random<KeyT>(C2H_SEED(1), max_items, max_segment, segments);
}

C2H_TEST("DeviceSegmentedSortKeys: Edge case segments, random keys", "[keys][segmented][sort][device]", key_types)
{
  using KeyT = c2h::get<0, TestType>;
  test_edge_case_segments_random<KeyT>(C2H_SEED(4));
}

C2H_TEST("DeviceSegmentedSortKeys: Unspecified segments, random keys", "[keys][segmented][sort][device]", key_types)
{
  using KeyT = c2h::get<0, TestType>;
  test_unspecified_segments_random<KeyT>(C2H_SEED(4));
}
