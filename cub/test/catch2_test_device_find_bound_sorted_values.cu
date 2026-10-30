// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_find.cuh>
#include <cub/device/dispatch/tuning/tuning_find_bound_sorted_values.cuh>

#include <thrust/sort.h>

#include <cuda/cmath>
#include <cuda/iterator>

#include <algorithm>
#include <cstdint>
#include <limits>

#include "catch2_test_launch_helper.h"
#include "cub_test_macros.h"
#include <c2h/catch2_allocation_failure.h>

// %PARAM% TEST_LAUNCH lid 0:1:2

DECLARE_LAUNCH_WRAPPER(cub::DeviceFind::LowerBoundSortedValues, lower_bound_sorted_values);
DECLARE_LAUNCH_WRAPPER(cub::DeviceFind::UpperBoundSortedValues, upper_bound_sorted_values);

struct std_lower_bound_t
{
  template <typename RangeIteratorT, typename T, typename CompareOpT>
  RangeIteratorT operator()(RangeIteratorT first, RangeIteratorT last, const T& value, CompareOpT comp) const
  {
    return std::lower_bound(first, last, value, comp);
  }
} std_lower_bound;

struct std_upper_bound_t
{
  template <typename RangeIteratorT, typename T, typename CompareOpT>
  RangeIteratorT operator()(RangeIteratorT first, RangeIteratorT last, const T& value, CompareOpT comp) const
  {
    return std::upper_bound(first, last, value, comp);
  }
} std_upper_bound;

using types = c2h::type_list<std::uint8_t, std::int16_t, std::uint32_t, double>;

template <typename Value,
          typename CountT = std::size_t,
          typename Variant,
          typename HostVariant,
          typename CompareOp  = cuda::std::less<Value>,
          typename WrapOutput = cuda::std::identity>
void test_sorted(Variant variant,
                 HostVariant host_variant,
                 cuda::std::type_identity_t<CountT> range_num_items  = 7492,
                 cuda::std::type_identity_t<CountT> values_num_items = 749,
                 CompareOp compare_op                                = {},
                 WrapOutput wrap_output                              = {})
{
  c2h::device_vector<Value> range_d(range_num_items, thrust::default_init);
  c2h::gen(C2H_SEED(1), range_d);
  thrust::sort(c2h::device_policy, range_d.begin(), range_d.end(), compare_op);

  c2h::device_vector<Value> values_d(values_num_items, thrust::default_init);
  c2h::gen(C2H_SEED(2), values_d);
  thrust::sort(c2h::device_policy, values_d.begin(), values_d.end(), compare_op);

  using Result = std::ptrdiff_t;
  c2h::device_vector<Result> offsets_d(values_num_items, thrust::default_init);

  variant(thrust::raw_pointer_cast(range_d.data()),
          range_num_items,
          thrust::raw_pointer_cast(values_d.data()),
          values_num_items,
          wrap_output(thrust::raw_pointer_cast(offsets_d.data())),
          compare_op);

  c2h::host_vector<Value> range_h          = range_d;
  c2h::host_vector<Value> values_h         = values_d;
  const c2h::host_vector<Result> offsets_h = offsets_d;

  c2h::host_vector<Result> offsets_ref(values_num_items);
  auto offsets_ref_out = wrap_output(offsets_ref.data());
  for (CountT i = 0; i < values_num_items; ++i)
  {
    offsets_ref_out[i] =
      host_variant(range_h.data(), range_h.data() + range_num_items, values_h[i], compare_op) - range_h.data();
  }

  CHECK(offsets_ref == offsets_h);
}

CUB_TEST("DeviceFind::LowerBoundSortedValues works", "[find][device][binary-search]", CUB_SMALL, types)
{
  using value_type = c2h::get<0, TestType>;
  test_sorted<value_type>(lower_bound_sorted_values, std_lower_bound);
}

CUB_TEST("DeviceFind::UpperBoundSortedValues works", "[find][device][binary-search]", CUB_SMALL, types)
{
  using value_type = c2h::get<0, TestType>;
  test_sorted<value_type>(upper_bound_sorted_values, std_upper_bound);
}

using count_types = c2h::type_list<std::int32_t, std::uint32_t, std::int64_t, std::uint64_t>;

CUB_TEST("DeviceFind sorted bounds work with a transform output iterator",
         "[find][device][binary-search]",
         CUB_SMALL,
         count_types)
{
  using value_type = int;
  using count_type = c2h::get<0, TestType>;

  // A transform_output_iterator has a void value_type, which exercises the non_void_value_t fallback.
  const auto negate_output = [](auto it) {
    return cuda::make_transform_output_iterator(it, cuda::std::negate{});
  };
  test_sorted<value_type, count_type>(
    lower_bound_sorted_values, std_lower_bound, 7492, 749, cuda::std::less<value_type>{}, negate_output);
  test_sorted<value_type, count_type>(
    upper_bound_sorted_values, std_upper_bound, 7492, 749, cuda::std::less<value_type>{}, negate_output);
}

CUB_TEST(
  "DeviceFind::LowerBoundSortedValues works with descending order", "[find][device][binary-search]", CUB_SMALL, types)
{
  using value_type = c2h::get<0, TestType>;
  test_sorted<value_type>(lower_bound_sorted_values, std_lower_bound, 7492, 749, cuda::std::greater<value_type>{});
}

CUB_TEST(
  "DeviceFind::UpperBoundSortedValues works with descending order", "[find][device][binary-search]", CUB_SMALL, types)
{
  using value_type = c2h::get<0, TestType>;
  test_sorted<value_type>(upper_bound_sorted_values, std_upper_bound, 7492, 749, cuda::std::greater<value_type>{});
}

CUB_TEST("DeviceFind::LowerBoundSortedValues input sizes", "[find][device][binary-search]", CUB_SMALL)
{
  using value_type                   = int;
  const std::size_t range_num_items  = GENERATE(0, 1, 23, 123, 3234);
  const std::size_t values_num_items = GENERATE(0, 1, 52, 556, 5676);
  CAPTURE(range_num_items, values_num_items);
  test_sorted<value_type>(lower_bound_sorted_values, std_lower_bound, range_num_items, values_num_items);
}

CUB_TEST("DeviceFind::UpperBoundSortedValues input sizes", "[find][device][binary-search]", CUB_SMALL)
{
  using value_type                   = int;
  const std::size_t range_num_items  = GENERATE(0, 1, 23, 123, 3234);
  const std::size_t values_num_items = GENERATE(0, 1, 52, 556, 5676);
  CAPTURE(range_num_items, values_num_items);
  test_sorted<value_type>(upper_bound_sorted_values, std_upper_bound, range_num_items, values_num_items);
}

CUB_TEST("DeviceFind::LowerBoundSortedValues almost tile-sized input sizes", "[find][device][binary-search]", CUB_SMALL)
{
  using value_type = int;
  cuda::compute_capability cc{};
  REQUIRE(cub::detail::ptx_compute_cap(cc) == cudaSuccess);
  const auto policy = cub::detail::find_bound_sorted_values::policy_selector_from_types<value_type, value_type>{}(cc);
  const auto tile_size = std::size_t{static_cast<std::size_t>(policy.threads_per_block) * policy.items_per_thread};
  test_sorted<value_type>(lower_bound_sorted_values, std_lower_bound, tile_size - 1, 1);
  test_sorted<value_type>(lower_bound_sorted_values, std_lower_bound, tile_size, 1);
  test_sorted<value_type>(lower_bound_sorted_values, std_lower_bound, 1, tile_size - 1);
  test_sorted<value_type>(lower_bound_sorted_values, std_lower_bound, 1, tile_size);
}

CUB_TEST("DeviceFind::UpperBoundSortedValues almost tile-sized input sizes", "[find][device][binary-search]", CUB_SMALL)
{
  using value_type = int;
  cuda::compute_capability cc{};
  REQUIRE(cub::detail::ptx_compute_cap(cc) == cudaSuccess);
  const auto policy = cub::detail::find_bound_sorted_values::policy_selector_from_types<value_type, value_type>{}(cc);
  const auto tile_size = std::size_t{static_cast<std::size_t>(policy.threads_per_block) * policy.items_per_thread};
  test_sorted<value_type>(upper_bound_sorted_values, std_upper_bound, tile_size - 1, 1);
  test_sorted<value_type>(upper_bound_sorted_values, std_upper_bound, tile_size, 1);
  test_sorted<value_type>(upper_bound_sorted_values, std_upper_bound, 1, tile_size - 1);
  test_sorted<value_type>(upper_bound_sorted_values, std_upper_bound, 1, tile_size);
}

template <typename Variant, typename HostVariant>
void test_sorted_uint32_limit(Variant variant, HostVariant host_variant)
{
  using value_type         = std::uint32_t;
  constexpr auto max_count = std::numeric_limits<value_type>::max();
  cuda::compute_capability cc{};
  REQUIRE(cub::detail::ptx_compute_cap(cc) == cudaSuccess);
  const auto policy = cub::detail::find_bound_sorted_values::policy_selector_from_types<value_type, value_type>{}(cc);
  const auto tile_size      = static_cast<value_type>(policy.threads_per_block * policy.items_per_thread);
  const auto last_full_tile = cuda::round_down(max_count, tile_size);
  const auto total_items    = GENERATE_COPY(last_full_tile - 1, last_full_tile, last_full_tile + 1, max_count);
  constexpr value_type values_num_items = 2000;
  const value_type range_num_items      = total_items - values_num_items;
  CAPTURE(range_num_items, values_num_items, tile_size);

  const auto range  = cuda::counting_iterator<value_type>{0};
  const auto values = cuda::counting_iterator<value_type>{range_num_items - values_num_items / 2};

  c2h::device_vector<value_type> offsets_d(values_num_items, max_count);
  variant(range,
          range_num_items,
          values,
          values_num_items,
          thrust::raw_pointer_cast(offsets_d.data()),
          cuda::std::less<value_type>{});

  c2h::host_vector<value_type> offsets_ref(values_num_items);
  for (value_type i = 0; i < values_num_items; ++i)
  {
    offsets_ref[i] = static_cast<value_type>(
      host_variant(range, range + range_num_items, values[i], cuda::std::less<value_type>{}) - range);
  }
  REQUIRE(offsets_ref == offsets_d);
}

CUB_TEST("DeviceFind::LowerBoundSortedValues near the uint32 limit", "[find][device][binary-search]", CUB_SMALL)
{
  test_sorted_uint32_limit(lower_bound_sorted_values, std_lower_bound);
}

CUB_TEST("DeviceFind::UpperBoundSortedValues near the uint32 limit", "[find][device][binary-search]", CUB_SMALL)
{
  test_sorted_uint32_limit(upper_bound_sorted_values, std_upper_bound);
}

// this test exceeds 4GiB of memory and the range of 32-bit integers
CUB_TEST("DeviceFind::LowerBoundSortedValues really large input",
         "[find][device][binary-search][skip-cs-racecheck][skip-cs-initcheck][skip-cs-synccheck]",
         CUB_LARGE)
{
  try
  {
    using value_type            = char;
    const auto range_num_items  = size_t{1} << GENERATE(30, 31, 32, 33);
    const auto values_num_items = range_num_items / 100;
    test_sorted<value_type>(lower_bound_sorted_values, std_lower_bound, range_num_items, values_num_items);
  }
  catch (...)
  {
    c2h::skip_if_allocation_failure();
  }
}

// this test exceeds 4GiB of memory and the range of 32-bit integers
CUB_TEST("DeviceFind::UpperBoundSortedValues really large input",
         "[find][device][binary-search][skip-cs-racecheck][skip-cs-initcheck][skip-cs-synccheck]",
         CUB_LARGE)
{
  try
  {
    using value_type            = char;
    const auto range_num_items  = size_t{1} << GENERATE(30, 31, 32, 33);
    const auto values_num_items = range_num_items / 100;
    test_sorted<value_type>(upper_bound_sorted_values, std_upper_bound, range_num_items, values_num_items);
  }
  catch (...)
  {
    c2h::skip_if_allocation_failure();
  }
}
