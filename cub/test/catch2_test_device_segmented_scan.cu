#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_segmented_scan.cuh>
#include <cub/device/dispatch/dispatch_segmented_scan.cuh>

#include <thrust/host_vector.h>

#include <cstdint>
#include <iostream>
#include <utility>

#include "catch2_test_device_reduce.cuh"
#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.h>
#include <c2h/custom_type.h>
#include <c2h/extended_types.h>

DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedScan::InclusiveSegmentedScan, device_inclusive_segmented_scan);

// %PARAM% TEST_LAUNCH lid 0:1:2
// %PARAM% TEST_TYPES types 0:1:2:3

// List of types to test
using custom_t =
  c2h::custom_type_t<c2h::accumulateable_t,
                     c2h::equal_comparable_t,
                     c2h::lexicographical_less_comparable_t,
                     c2h::lexicographical_greater_comparable_t>;

#if TEST_TYPES == 0
using full_type_list = c2h::type_list<type_pair<std::uint8_t>, type_pair<std::int8_t, std::int32_t>>;
#elif TEST_TYPES == 1
using full_type_list = c2h::type_list<type_pair<std::int32_t>, type_pair<std::int64_t>>;
#elif TEST_TYPES == 2
using full_type_list =
  c2h::type_list<type_pair<uchar3>,
                 type_pair<
#  if _CCCL_CTK_AT_LEAST(13, 0)
                   ulonglong4_16a
#  else // _CCCL_CTK_AT_LEAST(13, 0)
                   ulonglong4
#  endif // _CCCL_CTK_AT_LEAST(13, 0)
                   >>;
#elif TEST_TYPES == 3
// clang-format off
using full_type_list = c2h::type_list<
type_pair<custom_t>
#if TEST_HALF_T()
, type_pair<half_t> // testing half
#endif // TEST_HALF_T()
#if TEST_BF_T()
, type_pair<bfloat16_t> // testing bf16
#endif // TEST_BF_T()
>;
// clang-format on
#endif

using offsets = c2h::type_list<std::int32_t, std::uint32_t>;

C2H_TEST("Device segmented_scan works with all device interfaces", "[segmented][scan][device]", full_type_list, offsets)
{
  using params   = params_t<TestType>;
  using input_t  = typename params::item_t;
  using output_t = typename params::output_t;
  using offset_t = int32_t;

  constexpr offset_t min_items = 2048;
  constexpr offset_t max_items = 1024 * 1024;

  // Generate the input sizes to test for
  const offset_t num_items = GENERATE_COPY(
    take(3, random(min_items, max_items)),
    values({
      min_items,
      max_items,
    }));

  const offset_t small_size  = num_items / 1024;
  const offset_t medium_size = num_items / 128;
  const offset_t large_size  = num_items / 16;

  assert(small_size > 0);

  // Range of segment sizes to generate
  // Note that the segment range [0, 1] may also include one last segment with more than 1 items
  const std::tuple<offset_t, offset_t> seg_size_range =
    GENERATE_COPY(table<offset_t, offset_t>({{0, small_size}, {medium_size, large_size}, {large_size, num_items}}));
  std::cout << "Test seg_size_range: [" << std::get<0>(seg_size_range) << ", " << std::get<1>(seg_size_range) << "]\n";

  // Generate input segments
  c2h::device_vector<offset_t> segment_offsets = c2h::gen_uniform_offsets<offset_t>(
    C2H_SEED(1), num_items, std::get<0>(seg_size_range), std::get<1>(seg_size_range));
  const offset_t num_segments = static_cast<offset_t>(segment_offsets.size() - 1);
  auto d_offsets_it           = thrust::raw_pointer_cast(segment_offsets.data());

  std::cout << "Num segments: " << num_segments << " \n";

  // Generate input data
  c2h::device_vector<input_t> in_items(num_items);
  c2h::gen(C2H_SEED(2), in_items);
  auto d_in_it = thrust::raw_pointer_cast(in_items.data());

  c2h::device_vector<output_t> output_vec(num_items);
  auto d_out_it = thrust::raw_pointer_cast(output_vec.data());

  device_inclusive_segmented_scan(
    d_in_it, d_out_it, num_segments, d_offsets_it, d_offsets_it + 1, d_offsets_it, ::cuda::std::plus<>{});
}
