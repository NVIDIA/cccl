// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_merge.cuh>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>

#include <algorithm>

#include <test_util.h>

#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.h>

// %PARAM% TEST_LAUNCH lid 0:1:2

DECLARE_LAUNCH_WRAPPER(cub::DeviceMerge::MergePairs, merge_pairs);
DECLARE_LAUNCH_WRAPPER(cub::DeviceMerge::MergeKeys, merge_keys);

using types = c2h::type_list<std::uint8_t, std::int16_t, std::uint32_t, double>;

template <typename Key,
          typename Offset,
          typename CompareOp = cuda::std::less<Key>,
          typename MergeKeys = decltype(::merge_keys)>
void test_keys(Offset size1 = 3623, Offset size2 = 6346, CompareOp compare_op = {}, MergeKeys merge_keys = ::merge_keys)
{
  CAPTURE(c2h::type_name<Key>(), c2h::type_name<Offset>(), size1, size2);

  c2h::device_vector<Key> keys1_d(size1);
  c2h::device_vector<Key> keys2_d(size2);

  c2h::gen(C2H_SEED(1), keys1_d);
  c2h::gen(C2H_SEED(1), keys2_d);

  thrust::sort(c2h::device_policy, keys1_d.begin(), keys1_d.end(), compare_op);
  thrust::sort(c2h::device_policy, keys2_d.begin(), keys2_d.end(), compare_op);
  // CAPTURE(keys1_d, keys2_d);

  c2h::device_vector<Key> result_d(size1 + size2);
  merge_keys(thrust::raw_pointer_cast(keys1_d.data()),
             static_cast<Offset>(keys1_d.size()),
             thrust::raw_pointer_cast(keys2_d.data()),
             static_cast<Offset>(keys2_d.size()),
             thrust::raw_pointer_cast(result_d.data()),
             compare_op);

  c2h::host_vector<Key> keys1_h = keys1_d;
  c2h::host_vector<Key> keys2_h = keys2_d;
  c2h::host_vector<Key> reference_h(size1 + size2);
  std::merge(keys1_h.begin(), keys1_h.end(), keys2_h.begin(), keys2_h.end(), reference_h.begin(), compare_op);

  // FIXME(bgruber): comparing std::vectors (slower than thrust vectors) but compiles a lot faster
  CHECK((detail::to_vec(reference_h) == detail::to_vec(c2h::host_vector<Key>(result_d))));
}

C2H_TEST("DeviceMerge::MergeKeys key types", "[merge][device]", types)
{
  using key_t    = c2h::get<0, TestType>;
  using offset_t = int;
  test_keys<key_t, offset_t>();
}

C2H_TEST("DeviceMerge::MergeKeys works for large number of items",
         "[merge][device][skip-cs-racecheck][skip-cs-initcheck][skip-cs-synccheck]")
try
{
  using key_t    = char;
  using offset_t = int64_t;

  // Clamp 64-bit offset type problem sizes to just slightly larger than 2^32 items
  const auto num_items_int_max = static_cast<offset_t>(cuda::std::numeric_limits<std::int32_t>::max());

  // Generate the input sizes to test for
  const offset_t num_items_lhs =
    GENERATE_COPY(values({num_items_int_max + offset_t{1000000}, num_items_int_max - 1, offset_t{3}}));
  const offset_t num_items_rhs =
    GENERATE_COPY(values({num_items_int_max + offset_t{1000000}, num_items_int_max, offset_t{3}}));

  test_keys<key_t, offset_t>(num_items_lhs, num_items_rhs, cuda::std::less<>{});
}
catch (const std::bad_alloc&)
{
  // allocation failure is not a test failure, so we can run tests on smaller GPUs
}

C2H_TEST("DeviceMerge::MergeKeys input sizes", "[merge][device]")
{
  using key_t    = int;
  using offset_t = int;
  // TODO(bgruber): maybe less combinations
  const auto size1 = offset_t{GENERATE(0, 1, 23, 123, 3234)};
  const auto size2 = offset_t{GENERATE(0, 1, 52, 556, 56767)};
  test_keys<key_t>(size1, size2);
}

// cannot put those in an anon namespace, or nvcc complains that the kernels have internal linkage
using unordered_t = c2h::custom_type_t<c2h::equal_comparable_t>;
struct order
{
  __host__ __device__ auto operator()(const unordered_t& a, const unordered_t& b) const -> bool
  {
    return a.key < b.key;
  }
};

C2H_TEST("DeviceMerge::MergeKeys no operator<", "[merge][device]")
{
  using key_t    = unordered_t;
  using offset_t = int;
  test_keys<key_t, offset_t, order>();
}

namespace
{
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
} // namespace

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
  c2h::device_vector<Key> keys1_d(size1);
  c2h::device_vector<Key> keys2_d(size2);
  c2h::gen(C2H_SEED(1), keys1_d);
  c2h::gen(C2H_SEED(1), keys2_d);
  thrust::sort(c2h::device_policy, keys1_d.begin(), keys1_d.end(), compare_op);
  thrust::sort(c2h::device_policy, keys2_d.begin(), keys2_d.end(), compare_op);

  // the values must be functionally dependent on the keys (equal key => equal value), since merge is unstable
  c2h::device_vector<Value> values1_d(size1);
  c2h::device_vector<Value> values2_d(size2);
  thrust::transform(c2h::device_policy, keys1_d.begin(), keys1_d.end(), values1_d.begin(), key_to_value<Value>{});
  thrust::transform(c2h::device_policy, keys2_d.begin(), keys2_d.end(), values2_d.begin(), key_to_value<Value>{});
  //  CAPTURE(keys1_d, keys2_d, values1_d, values2_d);

  // compute CUB result
  c2h::device_vector<Key> result_keys_d(size1 + size2);
  c2h::device_vector<Value> result_values_d(size1 + size2);
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
  c2h::host_vector<Key> reference_keys_h(size1 + size2);
  c2h::host_vector<Value> reference_values_h(size1 + size2);
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
                 return compare_op(thrust::get<0>(a), thrust::get<0>(b));
               });
  }

  // FIXME(bgruber): comparing std::vectors (slower than thrust vectors) but compiles a lot faster
  CHECK((detail::to_vec(reference_keys_h) == detail::to_vec(c2h::host_vector<Key>(result_keys_d))));
  CHECK((detail::to_vec(reference_values_h) == detail::to_vec(c2h::host_vector<Value>(result_values_d))));
}

C2H_TEST("DeviceMerge::MergePairs key types", "[merge][device]", types)
{
  using key_t    = c2h::get<0, TestType>;
  using value_t  = int;
  using offset_t = int;
  test_pairs<key_t, value_t, offset_t>();
}

// TODO(bgruber): fine tune the type sizes again to hit the fallback and the vsmem policies
// C2H_TEST("DeviceMerge::MergePairs large key types", "[merge][device]", large_types)
// {
//   using key_t    = c2h::get<0, TestType>;
//   using value_t  = int;
//   using offset_t = int;
//   test_pairs<key_t, value_t, offset_t>();
// }

C2H_TEST("DeviceMerge::MergePairs value types", "[merge][device]", types)
{
  using key_t    = int;
  using value_t  = c2h::get<0, TestType>;
  using offset_t = int;
  test_pairs<key_t, value_t, offset_t>();
}

C2H_TEST("DeviceMerge::MergePairs input sizes", "[merge][device]")
{
  using key_t      = int;
  using value_t    = int;
  using offset_t   = int;
  const auto size1 = offset_t{GENERATE(0, 1, 23, 123, 3234234)};
  const auto size2 = offset_t{GENERATE(0, 1, 52, 556, 56767)};
  test_pairs<key_t, value_t>(size1, size2);
}

// this test exceeds 4GiB of memory and the range of 32-bit integers
C2H_TEST("DeviceMerge::MergePairs really large input",
         "[merge][device][skip-cs-racecheck][skip-cs-initcheck][skip-cs-synccheck]")
try
{
  using key_t     = char;
  using value_t   = char;
  const auto size = std::int64_t{1} << GENERATE(30, 31, 32, 33);
  test_pairs<key_t, value_t>(size, size, cuda::std::less<>{});
}
catch (const std::bad_alloc&)
{
  // allocation failure is not a test failure, so we can run tests on smaller GPUs
}

C2H_TEST("DeviceMerge::MergePairs iterators", "[merge][device]")
{
  using key_t             = int;
  using value_t           = int;
  using offset_t          = int;
  const offset_t size1    = 363;
  const offset_t size2    = 634;
  const auto values_start = 123456789;

  auto key_it   = thrust::counting_iterator<key_t>{};
  auto value_it = thrust::counting_iterator<key_t>{values_start};

  // compute CUB result
  c2h::device_vector<key_t> result_keys_d(size1 + size2);
  c2h::device_vector<value_t> result_values_d(size1 + size2);
  merge_pairs(
    key_it,
    value_it,
    size1,
    key_it,
    value_it,
    size2,
    result_keys_d.begin(),
    result_values_d.begin(),
    cuda::std::less<key_t>{});

  // check result
  c2h::host_vector<key_t> result_keys_h     = result_keys_d;
  c2h::host_vector<value_t> result_values_h = result_values_d;
  const auto smaller_size                   = std::min(size1, size2);
  for (offset_t i = 0; i < static_cast<offset_t>(result_keys_h.size()); i++)
  {
    if (i < 2 * smaller_size)
    {
      CHECK(result_keys_h[i + 0] == i / 2);
      CHECK(result_values_h[i + 0] == values_start + i / 2);
    }
    else
    {
      CHECK(result_keys_h[i] == i - smaller_size);
      CHECK(result_values_h[i] == values_start + i - smaller_size);
    }
  }
}
