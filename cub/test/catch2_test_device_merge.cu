// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_merge.cuh>

#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>

#include <cuda/iterator>

#include <algorithm>

#include <test_util.h>

#include "catch2_test_device_merge_common.cuh"
#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.h>

// %PARAM% TEST_LAUNCH lid 0:1:2

using types = c2h::type_list<std::uint8_t, std::int16_t, std::uint32_t, double>;

C2H_TEST("DeviceMerge::MergeKeys key types", "[merge][device]", types)
{
  using key_t    = c2h::get<0, TestType>;
  using offset_t = int;
  test_keys<key_t, offset_t>();
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

C2H_TEST("DeviceMerge::MergeKeys almost tile-sized input sizes", "[merge][device]")
{
  using key_t    = int;
  using offset_t = int;

  cuda::compute_capability cc{};
  REQUIRE(cub::detail::ptx_compute_cap(cc) == cudaSuccess);
  const offset_t items_per_tile =
    cub::detail::merge::policy_selector_from_types<key_t*, cub::NullType*, key_t*, cub::NullType*, offset_t>{}(cc)
      .items_per_thread;

  test_keys<key_t>(items_per_tile - 1, 1);
  test_keys<key_t>(items_per_tile, 1);
  test_keys<key_t>(1, items_per_tile - 1);
  test_keys<key_t>(1, items_per_tile);
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

C2H_TEST("DeviceMerge::MergePairs iterators", "[merge][device]")
{
  using key_t             = int;
  using value_t           = int;
  using offset_t          = int;
  const offset_t size1    = 363;
  const offset_t size2    = 634;
  const auto values_start = 123456789;

  const auto larger_size  = std::max(size1, size2);
  const auto smaller_size = std::min(size1, size2);

  auto test = [&](auto key1_it, auto value1_it, auto key2_it, auto value2_it) {
    // compute CUB result
    c2h::device_vector<key_t> result_keys_d(size1 + size2);
    c2h::device_vector<value_t> result_values_d(size1 + size2);
    merge_pairs(
      key1_it,
      value1_it,
      size1,
      key2_it,
      value2_it,
      size2,
      result_keys_d.begin(),
      result_values_d.begin(),
      cuda::std::less<key_t>{});

    // check result
    c2h::host_vector<key_t> result_keys_h     = result_keys_d;
    c2h::host_vector<value_t> result_values_h = result_values_d;

    for (offset_t i = 0; i < static_cast<offset_t>(result_keys_h.size()); i++)
    {
      CAPTURE(i);
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
  };

  auto key_it   = cuda::counting_iterator<key_t>{};
  auto value_it = cuda::counting_iterator<key_t>{values_start};

  c2h::device_vector<key_t> keys_vec(larger_size);
  thrust::sequence(keys_vec.begin(), keys_vec.end());
  c2h::device_vector<key_t> values_vec(larger_size);
  thrust::sequence(values_vec.begin(), values_vec.end(), values_start);

  SECTION("cit/cit/cit/cit")
  {
    test(key_it, value_it, key_it, value_it);
  }
  // key arrays have mixed types
  SECTION("vec/cit/cit/cit")
  {
    test(keys_vec.begin(), value_it, key_it, value_it);
  }
  // value arrays have mixed types
  SECTION("cit/vec/cit/cit")
  {
    test(key_it, values_vec.begin(), key_it, value_it);
  }
  // key and value arrays have mixed types
  SECTION("cit/vec/vec/cit")
  {
    test(key_it, values_vec.begin(), keys_vec.begin(), value_it);
  }
  // values have different iterator and keys
  SECTION("cit/vec/cit/vec")
  {
    test(key_it, values_vec.begin(), key_it, values_vec.begin());
  }
}
