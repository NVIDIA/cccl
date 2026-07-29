// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/device/device_merge.cuh>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <cuda/std/functional>
#include <cuda/std/tuple>

#include <algorithm>

#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.h>

// Shared launch wrappers and helpers for the DeviceMerge tests, used by both
// catch2_test_device_merge.cu and catch2_test_device_merge_large.cu. The helpers
// reference the launch wrappers via default arguments, so the wrappers are declared
// here as well (they expand to `inline constexpr` objects, so this is header-safe).

DECLARE_LAUNCH_WRAPPER(cub::DeviceMerge::MergePairs, merge_pairs);
DECLARE_LAUNCH_WRAPPER(cub::DeviceMerge::MergeKeys, merge_keys);

template <typename Key,
          typename Offset,
          typename CompareOp = cuda::std::less<Key>,
          typename MergeKeys = decltype(::merge_keys)>
void test_keys(Offset size1 = 3623, Offset size2 = 6346, CompareOp compare_op = {}, MergeKeys merge_keys = ::merge_keys)
{
  CAPTURE(c2h::type_name<Key>(), c2h::type_name<Offset>(), size1, size2);

  c2h::device_vector<Key> keys1_d(size1, thrust::default_init);
  c2h::device_vector<Key> keys2_d(size2, thrust::default_init);

  c2h::gen(C2H_SEED(1), keys1_d);
  c2h::gen(C2H_SEED(1), keys2_d);

  thrust::sort(c2h::device_policy, keys1_d.begin(), keys1_d.end(), compare_op);
  thrust::sort(c2h::device_policy, keys2_d.begin(), keys2_d.end(), compare_op);
  // CAPTURE(keys1_d, keys2_d);

  c2h::device_vector<Key> result_d(size1 + size2, thrust::default_init);
  merge_keys(thrust::raw_pointer_cast(keys1_d.data()),
             static_cast<Offset>(keys1_d.size()),
             thrust::raw_pointer_cast(keys2_d.data()),
             static_cast<Offset>(keys2_d.size()),
             thrust::raw_pointer_cast(result_d.data()),
             compare_op);

  c2h::host_vector<Key> keys1_h = keys1_d;
  c2h::host_vector<Key> keys2_h = keys2_d;
  c2h::host_vector<Key> reference_h(size1 + size2, thrust::default_init);
  std::merge(keys1_h.begin(), keys1_h.end(), keys2_h.begin(), keys2_h.end(), reference_h.begin(), compare_op);

  // comparing std::vectors instead compiles in 1m19s, thrust::host_vector 1m23s, thrust::device_vector 1m38
  // let's pick the host_vector, so we don't stress device memory with another (potentially big) allocation
  c2h::host_vector<Key> result_h(result_d); // perform copy outside CHECK() to propagate a potential bad_alloc
  CHECK(reference_h == result_h);
}

// must use thrust::make_zip_iterator for now
// see https://github.com/NVIDIA/cccl/issues/6400
template <typename... Its>
auto zip(Its... its) -> decltype(thrust::make_zip_iterator(its...))
{
  return thrust::make_zip_iterator(its...);
}

template <typename Value>
struct key_to_value
{
  template <typename Key>
  __host__ __device__ auto operator()(const Key& k) const -> Value
  {
    Value v{};
    convert(k, v, 0);
    return v;
  }

  template <typename Key>
  __host__ __device__ static void convert(const Key& k, Value& v, ...)
  {
    v = static_cast<Value>(k);
  }

  template <template <typename> class... Policies>
  __host__ __device__ static void convert(const c2h::custom_type_t<Policies...>& k, Value& v, int)
  {
    v = static_cast<Value>(k.val);
  }

  template <typename Key, template <typename> class... Policies>
  __host__ __device__ static void convert(const Key& k, c2h::custom_type_t<Policies...>& v, int)
  {
    v     = {};
    v.val = static_cast<decltype(v.val)>(k);
  }
};

template <typename Key,
          typename Value,
          typename Offset,
          typename CompareOp  = cuda::std::less<Key>,
          typename MergePairs = decltype(::merge_pairs)>
void test_pairs(
  Offset size1 = 200, Offset size2 = 625, CompareOp compare_op = {}, MergePairs merge_pairs = ::merge_pairs)
{
  CAPTURE(c2h::type_name<Key>(), c2h::type_name<Value>(), c2h::type_name<Offset>(), size1, size2);

  // we start with random but sorted keys
  c2h::device_vector<Key> keys1_d(size1, thrust::no_init);
  c2h::device_vector<Key> keys2_d(size2, thrust::no_init);
  c2h::gen(C2H_SEED(1), keys1_d);
  c2h::gen(C2H_SEED(1), keys2_d);
  thrust::sort(c2h::device_policy, keys1_d.begin(), keys1_d.end(), compare_op);
  thrust::sort(c2h::device_policy, keys2_d.begin(), keys2_d.end(), compare_op);

  // the values must be functionally dependent on the keys (equal key => equal value), since merge is unstable
  c2h::device_vector<Value> values1_d(size1, thrust::no_init);
  c2h::device_vector<Value> values2_d(size2, thrust::no_init);
  thrust::transform(c2h::device_policy, keys1_d.begin(), keys1_d.end(), values1_d.begin(), key_to_value<Value>{});
  thrust::transform(c2h::device_policy, keys2_d.begin(), keys2_d.end(), values2_d.begin(), key_to_value<Value>{});
  //  CAPTURE(keys1_d, keys2_d, values1_d, values2_d);

  // compute CUB result
  c2h::device_vector<Key> result_keys_d(size1 + size2, thrust::no_init);
  c2h::device_vector<Value> result_values_d(size1 + size2, thrust::no_init);
  merge_pairs(
    thrust::raw_pointer_cast(keys1_d.data()),
    thrust::raw_pointer_cast(values1_d.data()),
    static_cast<Offset>(keys1_d.size()),
    thrust::raw_pointer_cast(keys2_d.data()),
    thrust::raw_pointer_cast(values2_d.data()),
    static_cast<Offset>(keys2_d.size()),
    thrust::raw_pointer_cast(result_keys_d.data()),
    thrust::raw_pointer_cast(result_values_d.data()),
    compare_op);

  // compute reference result
  c2h::host_vector<Key> reference_keys_h(size1 + size2, thrust::no_init);
  c2h::host_vector<Value> reference_values_h(size1 + size2, thrust::no_init);
  {
    c2h::host_vector<Key> keys1_h     = keys1_d;
    c2h::host_vector<Value> values1_h = values1_d;
    c2h::host_vector<Key> keys2_h     = keys2_d;
    c2h::host_vector<Value> values2_h = values2_d;
    using value_t                     = typename decltype(zip(keys1_h.begin(), values1_h.begin()))::value_type;
    std::merge(zip(keys1_h.begin(), values1_h.begin()),
               zip(keys1_h.end(), values1_h.end()),
               zip(keys2_h.begin(), values2_h.begin()),
               zip(keys2_h.end(), values2_h.end()),
               zip(reference_keys_h.begin(), reference_values_h.begin()),
               [&](const value_t& a, const value_t& b) {
                 return compare_op(cuda::std::get<0>(a), cuda::std::get<0>(b));
               });
  }

  // FIXME(bgruber): comparing std::vectors (slower than thrust vectors) but compiles a lot faster
  CHECK((detail::to_vec(reference_keys_h) == detail::to_vec(c2h::host_vector<Key>(result_keys_d))));
  CHECK((detail::to_vec(reference_values_h) == detail::to_vec(c2h::host_vector<Value>(result_values_d))));
}
