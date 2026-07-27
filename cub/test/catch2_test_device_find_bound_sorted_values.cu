// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_find.cuh>
#include <cub/device/dispatch/tuning/tuning_find_bound_sorted_values.cuh>

#include <thrust/sort.h>

#include <cuda/iterator>

#include <algorithm>

#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.h>

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

template <typename Value, typename Variant, typename HostVariant, typename CompareOp = cuda::std::less<Value>>
void test_sorted(Variant variant,
                 HostVariant host_variant,
                 std::size_t range_num_items  = 7492,
                 std::size_t values_num_items = 749,
                 CompareOp compare_op         = {})
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
          thrust::raw_pointer_cast(offsets_d.data()),
          compare_op);

  c2h::host_vector<Value> range_h    = range_d;
  c2h::host_vector<Value> values_h   = values_d;
  c2h::host_vector<Result> offsets_h = offsets_d;

  c2h::host_vector<Result> offsets_ref(values_num_items);
  for (auto i = 0u; i < values_num_items; ++i)
  {
    offsets_ref[i] =
      host_variant(range_h.data(), range_h.data() + range_num_items, values_h[i], compare_op) - range_h.data();
  }

  CHECK(offsets_ref == offsets_h);
}

C2H_TEST("DeviceFind::LowerBoundSortedValues works", "[find][device][binary-search]", types)
{
  using value_type = c2h::get<0, TestType>;
  test_sorted<value_type>(lower_bound_sorted_values, std_lower_bound);
}

C2H_TEST("DeviceFind::UpperBoundSortedValues works", "[find][device][binary-search]", types)
{
  using value_type = c2h::get<0, TestType>;
  test_sorted<value_type>(upper_bound_sorted_values, std_upper_bound);
}

C2H_TEST("DeviceFind::LowerBoundSortedValues works with a transform output iterator", "[find][device][binary-search]")
{
  using value_type = int;
  using Result     = std::ptrdiff_t;

  const std::size_t range_num_items  = 7492;
  const std::size_t values_num_items = 749;

  c2h::device_vector<value_type> range_d(range_num_items, thrust::default_init);
  c2h::gen(C2H_SEED(1), range_d);
  thrust::sort(c2h::device_policy, range_d.begin(), range_d.end());

  c2h::device_vector<value_type> values_d(values_num_items, thrust::default_init);
  c2h::gen(C2H_SEED(2), values_d);
  thrust::sort(c2h::device_policy, values_d.begin(), values_d.end());

  // A transform_output_iterator has a void value_type, which exercises the non_void_value_t fallback.
  c2h::device_vector<Result> offsets_d(values_num_items, thrust::default_init);
  auto offsets_out = cuda::make_transform_output_iterator(offsets_d.begin(), cuda::std::negate{});

  lower_bound_sorted_values(
    thrust::raw_pointer_cast(range_d.data()),
    range_num_items,
    thrust::raw_pointer_cast(values_d.data()),
    values_num_items,
    offsets_out,
    cuda::std::less<value_type>{});

  c2h::host_vector<value_type> range_h  = range_d;
  c2h::host_vector<value_type> values_h = values_d;
  c2h::host_vector<Result> offsets_h    = offsets_d;

  c2h::host_vector<Result> offsets_ref(values_num_items);
  for (auto i = 0u; i < values_num_items; ++i)
  {
    offsets_ref[i] =
      -(std::lower_bound(range_h.data(), range_h.data() + range_num_items, values_h[i]) - range_h.data());
  }

  CHECK(offsets_ref == offsets_h);
}

C2H_TEST("DeviceFind::LowerBoundSortedValues works with descending order", "[find][device][binary-search]", types)
{
  using value_type = c2h::get<0, TestType>;
  test_sorted<value_type>(lower_bound_sorted_values, std_lower_bound, 7492, 749, cuda::std::greater<value_type>{});
}

C2H_TEST("DeviceFind::UpperBoundSortedValues works with descending order", "[find][device][binary-search]", types)
{
  using value_type = c2h::get<0, TestType>;
  test_sorted<value_type>(upper_bound_sorted_values, std_upper_bound, 7492, 749, cuda::std::greater<value_type>{});
}

C2H_TEST("DeviceFind::LowerBoundSortedValues input sizes", "[find][device][binary-search]")
{
  using value_type                   = int;
  const std::size_t range_num_items  = GENERATE(0, 1, 23, 123, 3234);
  const std::size_t values_num_items = GENERATE(0, 1, 52, 556, 5676);
  CAPTURE(range_num_items, values_num_items);
  test_sorted<value_type>(lower_bound_sorted_values, std_lower_bound, range_num_items, values_num_items);
}

C2H_TEST("DeviceFind::UpperBoundSortedValues input sizes", "[find][device][binary-search]")
{
  using value_type                   = int;
  const std::size_t range_num_items  = GENERATE(0, 1, 23, 123, 3234);
  const std::size_t values_num_items = GENERATE(0, 1, 52, 556, 5676);
  CAPTURE(range_num_items, values_num_items);
  test_sorted<value_type>(upper_bound_sorted_values, std_upper_bound, range_num_items, values_num_items);
}

C2H_TEST("DeviceFind::LowerBoundSortedValues almost tile-sized input sizes", "[find][device][binary-search]")
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

C2H_TEST("DeviceFind::UpperBoundSortedValues almost tile-sized input sizes", "[find][device][binary-search]")
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

// this test exceeds 4GiB of memory and the range of 32-bit integers
C2H_TEST("DeviceFind::LowerBoundSortedValues really large input",
         "[find][device][binary-search][skip-cs-rangecheck][skip-cs-initcheck][skip-cs-synccheck][large-mem]")
{
  try
  {
    using value_type            = char;
    const auto range_num_items  = size_t{1} << GENERATE(30, 31, 32, 33);
    const auto values_num_items = range_num_items / 100;
    test_sorted<value_type>(lower_bound_sorted_values, std_lower_bound, range_num_items, values_num_items);
  }
  catch (const std::bad_alloc&)
  {
    // allocation failure is not a test failure, so we can run tests on smaller GPUs
    SUCCEED("allocation failure is not a test failure");
  }
}

// this test exceeds 4GiB of memory and the range of 32-bit integers
C2H_TEST("DeviceFind::UpperBoundSortedValues really large input",
         "[find][device][binary-search][skip-cs-rangecheck][skip-cs-initcheck][skip-cs-synccheck][large-mem]")
{
  try
  {
    using value_type            = char;
    const auto range_num_items  = size_t{1} << GENERATE(30, 31, 32, 33);
    const auto values_num_items = range_num_items / 100;
    test_sorted<value_type>(upper_bound_sorted_values, std_upper_bound, range_num_items, values_num_items);
  }
  catch (const std::bad_alloc&)
  {
    // allocation failure is not a test failure, so we can run tests on smaller GPUs
    SUCCEED("allocation failure is not a test failure");
  }
}
