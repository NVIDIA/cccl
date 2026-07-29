// SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

// Large-memory DeviceRadixSort::SortKeys tests, split out from catch2_test_device_radix_sort_keys.cu
// so they can be run serially (tagged [large-mem]) without serializing the small tests in that file.

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_radix_sort.cuh>
#include <cub/util_type.cuh>

#include <thrust/functional.h>
#include <thrust/memory.h>
#include <thrust/scatter.h>
#include <thrust/transform.h>

#include <cuda/iterator>
#include <cuda/std/type_traits>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <new> // bad_alloc

#include "catch2_large_array_sort_helper.cuh"
#include "catch2_radix_sort_helper.cuh"
#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.h>
#include <c2h/extended_types.h>

// %PARAM% TEST_LAUNCH lid 0:1:2

DECLARE_LAUNCH_WRAPPER(cub::DeviceRadixSort::SortKeys, sort_keys);
DECLARE_LAUNCH_WRAPPER(cub::DeviceRadixSort::SortKeysDescending, sort_keys_descending);

// %PARAM% TEST_KEY_BITS key_bits 8:16:32:64

// The unsigned integer for the given byte count should be first:
#if TEST_KEY_BITS == 8
using key_types            = c2h::type_list<cuda::std::uint8_t, cuda::std::int8_t, bool, char>;
using bit_window_key_types = c2h::type_list<cuda::std::uint8_t, cuda::std::int8_t, char>;
#  define NO_FP_KEY_TYPES
#elif TEST_KEY_BITS == 16
// clang-format off
using key_types = c2h::type_list<
    cuda::std::uint16_t
  , cuda::std::int16_t
#if TEST_HALF_T()
  , half_t
#endif // TEST_HALF_T()
#if TEST_BF_T()
  , bfloat16_t
#endif // TEST_BF_T()
  >;
// clang-format on
using bit_window_key_types = c2h::type_list<cuda::std::uint16_t, cuda::std::int16_t>;
#  define NO_FP_KEY_TYPES
#elif TEST_KEY_BITS == 32
using key_types            = c2h::type_list<cuda::std::uint32_t, cuda::std::int32_t, float>;
using bit_window_key_types = c2h::type_list<cuda::std::uint32_t, cuda::std::int32_t>;
using fp_key_types         = c2h::type_list<float>;
#elif TEST_KEY_BITS == 64
using key_types            = c2h::type_list<cuda::std::uint64_t, cuda::std::int64_t, double>;
using bit_window_key_types = c2h::type_list<cuda::std::uint64_t, cuda::std::int64_t>;
using fp_key_types         = c2h::type_list<double>;
#endif

// Used for tests that just need a single type for testing:
using single_key_type = c2h::type_list<c2h::get<0, key_types>>;

template <typename key_t, typename num_items_t>
void do_large_offset_test(std::size_t num_items)
{
  const bool is_descending = GENERATE(false, true);

  CAPTURE(num_items, is_descending);

  try
  {
    large_array_sort_helper<key_t> arrays;
    arrays.initialize_for_unstable_key_sort(C2H_SEED(1), num_items, is_descending);

    TIME(c2h::cpu_timer timer);

    double_buffer_sort_t action(is_descending);
    action.initialize();
    const num_items_t typed_num_items = static_cast<num_items_t>(num_items);
    launch(action, arrays.keys_buffer, typed_num_items, begin_bit<key_t>(), end_bit<key_t>());

    arrays.keys_buffer.selector = action.selector();
    action.finalize();

    auto& sorted_keys = arrays.keys_buffer.selector == 0 ? arrays.keys_in : arrays.keys_out;

    TIME(timer.print_elapsed_seconds_and_reset("Device sort"));

    arrays.verify_unstable_key_sort(num_items, is_descending, sorted_keys);
  }
  catch ([[maybe_unused]] std::bad_alloc& e)
  {
#ifdef DEBUG_CHECKED_ALLOC_FAILURE
    const std::size_t num_bytes = num_items * sizeof(key_t);
    std::cerr
      << "Skipping radix sort test with " << num_items << " elements (" << num_bytes << " bytes): " << e.what() << "\n";
#endif // DEBUG_CHECKED_ALLOC_FAILURE
    SUCCEED("allocation failure is not a test failure");
  }
}

C2H_TEST("DeviceRadixSort::SortKeys: 32-bit overflow check",
         "[large-mem][large][keys][radix][sort][device][skip-cs-synccheck][skip-cs-initcheck][skip-cs-racecheck]",
         single_key_type)
{
  using key_t       = c2h::get<0, TestType>;
  using num_items_t = std::uint32_t;

  // Test problem sizes near and at the maximum offset value to ensure that internal calculations
  // do not overflow.
  constexpr std::size_t max_offset    = cuda::std::numeric_limits<num_items_t>::max();
  constexpr std::size_t min_num_items = max_offset - 5;
  constexpr std::size_t max_num_items = max_offset;
  const std::size_t num_items         = GENERATE_COPY(min_num_items, max_num_items);

  do_large_offset_test<key_t, num_items_t>(num_items);
}

C2H_TEST("DeviceRadixSort::SortKeys: Large Offsets",
         "[large-mem][large][keys][radix][sort][device][skip-cs-synccheck][skip-cs-initcheck][skip-cs-racecheck]",
         single_key_type)
{
  using key_t       = c2h::get<0, TestType>;
  using num_items_t = std::uint64_t;

  constexpr std::size_t min_num_items = std::size_t{1} << 32;
  constexpr std::size_t max_num_items = std::size_t{1} << 33;
  const std::size_t num_items         = GENERATE_COPY(take(2, random(min_num_items, max_num_items)));

  do_large_offset_test<key_t, num_items_t>(num_items);
}
