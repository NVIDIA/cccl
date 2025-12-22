// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_find.cuh>

#include <thrust/sort.h>

#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.h>

// %PARAM% TEST_LAUNCH lid 0:1:2

DECLARE_LAUNCH_WRAPPER(cub::DeviceFind::LowerBound, lower_bound);
DECLARE_LAUNCH_WRAPPER(cub::DeviceFind::UpperBound, upper_bound);

using types = c2h::type_list<std::uint8_t, std::int16_t, std::uint32_t, double>;

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

template <typename Value, typename Variant, typename HostVariant, typename CompareOp = cuda::std::less<Value>>
void test_vectorized(Variant variant, HostVariant host_variant, std::size_t num_items = 7492, CompareOp compare_op = {})
{
  c2h::device_vector<Value> target_values_d(num_items / 100, thrust::default_init);
  c2h::gen(C2H_SEED(1), target_values_d);

  c2h::device_vector<Value> values_d(num_items + target_values_d.size(), thrust::default_init);
  c2h::gen(C2H_SEED(1), values_d);

  thrust::copy(c2h::device_policy, target_values_d.begin(), target_values_d.end(), values_d.begin());
  thrust::sort(c2h::device_policy, values_d.begin(), values_d.end(), compare_op);

  using Result = Value*;
  c2h::device_vector<Result> result_d(target_values_d.size(), thrust::default_init);
  variant(thrust::raw_pointer_cast(values_d.data()),
          thrust::raw_pointer_cast(values_d.data() + num_items),
          thrust::raw_pointer_cast(target_values_d.data()),
          thrust::raw_pointer_cast(target_values_d.data() + target_values_d.size()),
          thrust::raw_pointer_cast(result_d.data()),
          compare_op);

  c2h::host_vector<Value> target_values_h = target_values_d;
  c2h::host_vector<Value> values_h        = values_d;

  c2h::host_vector<Result> result_h = result_d;

  c2h::host_vector<std::ptrdiff_t> offsets_ref(result_h.size(), thrust::default_init);
  c2h::host_vector<std::ptrdiff_t> offsets_h(result_h.size(), thrust::default_init);

  for (auto i = 0u; i < target_values_h.size(); ++i)
  {
    offsets_ref[i] =
      host_variant(values_h.data(), values_h.data() + num_items, target_values_h[i], compare_op) - values_h.data();
    offsets_h[i] = result_h[i] - thrust::raw_pointer_cast(values_d.data());
  }

  CHECK(offsets_ref == offsets_h);
}

C2H_TEST("DeviceFind::LowerBound works", "[find][device][binary-search]", types)
{
  using value_type = c2h::get<0, TestType>;
  test_vectorized<value_type>(lower_bound, std_lower_bound);
}

C2H_TEST("DeviceFind::UpperBound works", "[find][device][binary-search]", types)
{
  using value_type = c2h::get<0, TestType>;
  test_vectorized<value_type>(upper_bound, std_upper_bound);
}

// this test exceeds 4GiB of memory and the range of 32-bit integers
C2H_TEST("DeviceFind::LowerBound really large input",
         "[find][device][binary-search][skip-cs-rangecheck][skip-cs-initcheck][skip-cs-synccheck]")
{
  try
  {
    using value_type = char;
    const auto size  = std::int64_t{1} << GENERATE(30, 31, 32, 33);
    test_vectorized<value_type>(lower_bound, std_lower_bound, size);
  }
  catch (const std::bad_alloc&)
  {
    // allocation failure is not a test failure, so we can run tests on smaller GPUs
  }
}

// this test exceeds 4GiB of memory and the range of 32-bit integers
C2H_TEST("DeviceFind::UpperBound really large input",
         "[find][device][binary-search][skip-cs-rangecheck][skip-cs-initcheck][skip-cs-synccheck]")
{
  try
  {
    using value_type = char;
    const auto size  = std::int64_t{1} << GENERATE(30, 31, 32, 33);
    test_vectorized<value_type>(upper_bound, std_upper_bound, size);
  }
  catch (const std::bad_alloc&)
  {
    // allocation failure is not a test failure, so we can run tests on smaller GPUs
  }
}
