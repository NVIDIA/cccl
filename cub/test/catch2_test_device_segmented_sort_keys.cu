/******************************************************************************
 * Copyright (c) 2011-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_segmented_sort.cuh>
#include <cub/util_type.cuh>

#include "catch2_radix_sort_helper.cuh"
#include "catch2_segmented_sort_helper.cuh"
#include <c2h/bfloat16.cuh>
#include <c2h/catch2_test_helper.h>
#include <c2h/half.cuh>

// FIXME: Graph launch disabled, algorithm syncs internally. WAR exists for device-launch, figure out how to enable for
// graph launch.
// %PARAM% TEST_LAUNCH lid 0:1

DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedSort::StableSortKeys, stable_sort_keys);

using key_types =
  c2h::type_list<bool,
                 std::uint8_t,
                 std::uint64_t
#if TEST_HALF_T()
                 ,
                 half_t
#endif // TEST_HALF_T()
#if TEST_BF_T()
                 ,
                 bfloat16_t
#endif // TEST_BF_T()
                 >;

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

C2H_TEST("DeviceSegmentedSortKeys: very large number of segments",
         "[keys][segmented][sort][device][skip-cs-memcheck][skip-cs-racecheck][skip-cs-initcheck]",
         all_offset_types)
try
{
  using key_t                        = cuda::std::uint8_t; // minimize memory footprint to support a wider range of GPUs
  using segment_offset_t             = std::int64_t;
  using offset_t                     = c2h::get<0, TestType>;
  using segment_iterator_t           = segment_index_to_offset_op<offset_t, segment_offset_t>;
  constexpr std::size_t segment_size = 1000000;
  constexpr std::size_t uint32_max   = cuda::std::numeric_limits<std::uint32_t>::max();
  constexpr std::size_t num_items =
    (sizeof(offset_t) == 8) ? uint32_max + (1 << 20) : cuda::std::numeric_limits<offset_t>::max();
  constexpr segment_offset_t num_empty_segments = uint32_max;
  const segment_offset_t num_segments           = num_empty_segments + cuda::ceil_div(num_items, segment_size);
  CAPTURE(c2h::type_name<offset_t>(), num_items, num_segments);

  c2h::device_vector<key_t> in_keys(num_items);
  c2h::device_vector<key_t> out_keys(num_items);

  // Generate input keys
  constexpr auto max_histo_size = 250;
  segmented_verification_helper<key_t> verification_helper{max_histo_size};
  verification_helper.prepare_input_data(in_keys);

  auto offsets = thrust::make_transform_iterator(
    thrust::make_counting_iterator(std::size_t{0}),
    segment_iterator_t{num_empty_segments, num_segments, segment_size, num_items});

  stable_sort_keys(
    thrust::raw_pointer_cast(in_keys.data()),
    thrust::raw_pointer_cast(out_keys.data()),
    static_cast<offset_t>(num_items),
    static_cast<segment_offset_t>(num_segments),
    offsets,
    offsets + 1);

  // Verify the keys are sorted correctly
  verification_helper.verify_sorted(out_keys, offsets + num_empty_segments, num_segments - num_empty_segments);
}
catch (std::bad_alloc& e)
{
  std::cerr << "Skipping segmented sort test, insufficient GPU memory. " << e.what() << "\n";
}

C2H_TEST("DeviceSegmentedSort::SortKeys: very large segments",
         "[keys][segmented][sort][device][skip-cs-memcheck][skip-cs-racecheck][skip-cs-initcheck]",
         all_offset_types)
try
{
  using key_t                      = cuda::std::uint8_t; // minimize memory footprint to support a wider range of GPUs
  using segment_offset_t           = std::int32_t;
  using offset_t                   = c2h::get<0, TestType>;
  constexpr std::size_t uint32_max = cuda::std::numeric_limits<std::uint32_t>::max();
  constexpr int num_key_seeds      = 1;
  constexpr std::size_t num_items =
    (sizeof(offset_t) == 8) ? uint32_max + (1 << 20) : cuda::std::numeric_limits<offset_t>::max();
  const segment_offset_t num_segments = 2;
  CAPTURE(c2h::type_name<offset_t>(), num_items, num_segments);

  c2h::device_vector<key_t> in_keys(num_items);
  c2h::device_vector<key_t> out_keys(num_items);
  c2h::gen(C2H_SEED(num_key_seeds), in_keys);
  c2h::device_vector<offset_t> offsets(num_segments + 1);
  offsets[0] = 0;
  offsets[1] = static_cast<offset_t>(num_items);
  offsets[2] = static_cast<offset_t>(num_items);

  // Prepare information for later verification
  short_key_verification_helper<key_t> verification_helper{};
  verification_helper.prepare_verification_data(in_keys);

  stable_sort_keys(
    thrust::raw_pointer_cast(in_keys.data()),
    thrust::raw_pointer_cast(out_keys.data()),
    static_cast<offset_t>(num_items),
    static_cast<segment_offset_t>(num_segments),
    thrust::raw_pointer_cast(offsets.data()),
    offsets.cbegin() + 1);

  // Verify the keys are sorted correctly
  verification_helper.verify_sorted(out_keys);
}
catch (std::bad_alloc& e)
{
  std::cerr << "Skipping segmented sort test, insufficient GPU memory. " << e.what() << "\n";
}
