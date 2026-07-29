// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <thrust/copy.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/sort.h>

#include <cuda/std/functional>

#include <algorithm>
#include <cstddef>

#include <c2h/catch2_test_helper.h>

// Shared helpers for DeviceFind binary-search tests, used by both
// catch2_test_device_find.cu and catch2_test_device_find_large.cu.

struct std_lower_bound_t
{
  template <typename RangeIteratorT, typename T, typename CompareOpT>
  RangeIteratorT operator()(RangeIteratorT first, RangeIteratorT last, const T& value, CompareOpT comp) const
  {
    return std::lower_bound(first, last, value, comp);
  }
};
inline std_lower_bound_t std_lower_bound{};

struct std_upper_bound_t
{
  template <typename RangeIteratorT, typename T, typename CompareOpT>
  RangeIteratorT operator()(RangeIteratorT first, RangeIteratorT last, const T& value, CompareOpT comp) const
  {
    return std::upper_bound(first, last, value, comp);
  }
};
inline std_upper_bound_t std_upper_bound{};

template <typename Value, typename Variant, typename HostVariant, typename CompareOp = cuda::std::less<Value>>
void test_vectorized(Variant variant, HostVariant host_variant, std::size_t num_items = 7492, CompareOp compare_op = {})
{
  c2h::device_vector<Value> target_values_d(num_items / 100, thrust::default_init);
  c2h::gen(C2H_SEED(1), target_values_d);

  c2h::device_vector<Value> values_d(num_items + target_values_d.size(), thrust::default_init);
  c2h::gen(C2H_SEED(1), values_d);

  thrust::copy(c2h::device_policy, target_values_d.begin(), target_values_d.end(), values_d.begin());
  thrust::sort(c2h::device_policy, values_d.begin(), values_d.end(), compare_op);

  using Result = std::ptrdiff_t;
  c2h::device_vector<Result> offsets_d(target_values_d.size(), thrust::default_init);
  variant(thrust::raw_pointer_cast(values_d.data()),
          num_items,
          thrust::raw_pointer_cast(target_values_d.data()),
          target_values_d.size(),
          thrust::raw_pointer_cast(offsets_d.data()),
          compare_op);

  c2h::host_vector<Value> target_values_h = target_values_d;
  c2h::host_vector<Value> values_h        = values_d;

  c2h::host_vector<Result> offsets_h = offsets_d;

  c2h::host_vector<std::ptrdiff_t> offsets_ref(offsets_h.size(), thrust::default_init);

  for (auto i = 0u; i < target_values_h.size(); ++i)
  {
    offsets_ref[i] =
      host_variant(values_h.data(), values_h.data() + num_items, target_values_h[i], compare_op) - values_h.data();
  }

  CHECK(offsets_ref == offsets_h);
}
