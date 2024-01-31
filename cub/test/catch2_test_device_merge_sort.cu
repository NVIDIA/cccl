/******************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

#include <cub/device/device_merge_sort.cuh>

#include <thrust/copy.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>
#include <thrust/equal.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>

#include <algorithm>

#include "catch2_test_device_merge_sort_common.cuh"
#include "catch2_test_helper.h"
#include "catch2_test_launch_helper.h"

// %PARAM% TEST_LAUNCH lid 0:1:2

DECLARE_LAUNCH_WRAPPER(cub::DeviceMergeSort::SortPairs, sort_pairs);
DECLARE_LAUNCH_WRAPPER(cub::DeviceMergeSort::SortPairsCopy, sort_pairs_copy);
DECLARE_LAUNCH_WRAPPER(cub::DeviceMergeSort::StableSortPairs, stable_sort_pairs);

DECLARE_LAUNCH_WRAPPER(cub::DeviceMergeSort::SortKeys, sort_keys);
DECLARE_LAUNCH_WRAPPER(cub::DeviceMergeSort::SortKeysCopy, sort_keys_copy);
DECLARE_LAUNCH_WRAPPER(cub::DeviceMergeSort::StableSortKeys, stable_sort_keys);
DECLARE_LAUNCH_WRAPPER(cub::DeviceMergeSort::StableSortKeysCopy, stable_sort_keys_copy);

using key_types =
  c2h::type_list<std::uint8_t,
                 std::int16_t,
                 std::uint32_t,
                 double,
                 c2h::custom_type_t<c2h::equal_comparable_t, c2h::less_comparable_t>>;
using wide_key_types = c2h::type_list<std::uint32_t, double>;

using value_types =
  c2h::type_list<std::uint8_t, float, c2h::custom_type_t<c2h::equal_comparable_t, c2h::less_comparable_t>>;

/**
 * Function object that maps the targeted sorted rank of an item to a key.

 * E.g., `OffsetT` is `int32_t` and `KeyT` is `float`:
 * [  4,   2,   3,   1,   0] <= targeted key ranks
 * [4.0, 2.0, 3.0, 1.0, 0.0] <= corresponding keys
 */
template <typename OffsetT, typename KeyT>
struct rank_to_key_op_t
{
  __device__ __host__ KeyT operator()(const OffsetT& val)
  {
    return static_cast<KeyT>(val);
  }
};

template <typename OffsetT>
struct rank_to_key_op_t<OffsetT, c2h::custom_type_t<c2h::equal_comparable_t, c2h::less_comparable_t>>
{
  using custom_t = c2h::custom_type_t<c2h::equal_comparable_t, c2h::less_comparable_t>;
  __device__ __host__ custom_t operator()(const OffsetT& val)
  {
    custom_t custom_val{};
    custom_val.key = val;
    custom_val.val = val;
    return custom_val;
  }
};

/**
 * Helps initialize custom_type_t from a zip-iterator combination of sort-key and value
 */
template <typename CustomT>
struct tuple_to_custom_op_t
{
  template <typename KeyT, typename ValueT>
  __device__ __host__ CustomT operator()(const thrust::tuple<KeyT, ValueT>& val)
  {
    CustomT custom_val{};
    custom_val.key = static_cast<std::size_t>(thrust::get<0>(val));
    custom_val.val = static_cast<std::size_t>(thrust::get<1>(val));
    return custom_val;
  }
};

/**
 * Generates a shuffled array of key ranks. E.g., for a vector of size 5: [4, 2, 3, 1, 0]
 */
template <typename OffsetT>
thrust::device_vector<OffsetT> make_shuffled_key_ranks_vector(OffsetT num_items, c2h::seed_t seed)
{
  thrust::device_vector<OffsetT> key_ranks(num_items);
  thrust::sequence(key_ranks.begin(), key_ranks.end());
  thrust::shuffle(
    key_ranks.begin(), key_ranks.end(), thrust::default_random_engine{static_cast<unsigned int>(seed.get())});
  return key_ranks;
}

CUB_TEST("DeviceMergeSort::SortKeysCopy works", "[merge][sort][device]", wide_key_types)
{
  using key_t    = typename c2h::get<0, TestType>;
  using offset_t = std::int32_t;

  // Prepare input
  const offset_t num_items = GENERATE_COPY(take(2, random(1, 1000000)), values({500, 1000000, 2000000}));
  auto key_ranks           = make_shuffled_key_ranks_vector(num_items, CUB_SEED(2));
  thrust::device_vector<key_t> keys_in(num_items);
  thrust::transform(key_ranks.begin(), key_ranks.end(), keys_in.begin(), rank_to_key_op_t<offset_t, key_t>{});

  // Perform sort
  thrust::device_vector<key_t> keys_out(num_items, static_cast<key_t>(42));
  sort_keys_copy(
    thrust::raw_pointer_cast(keys_in.data()), thrust::raw_pointer_cast(keys_out.data()), num_items, custom_less_op_t{});

  // Verify results
  auto key_ranks_it     = thrust::make_counting_iterator(offset_t{});
  auto keys_expected_it = thrust::make_transform_iterator(key_ranks_it, rank_to_key_op_t<offset_t, key_t>{});
  bool results_equal    = thrust::equal(keys_out.cbegin(), keys_out.cend(), keys_expected_it);
  REQUIRE(results_equal == true);
}

CUB_TEST("DeviceMergeSort::SortKeys works", "[merge][sort][device]", wide_key_types)
{
  using key_t    = typename c2h::get<0, TestType>;
  using offset_t = std::int32_t;

  // Prepare input
  const offset_t num_items = GENERATE_COPY(take(2, random(1, 1000000)), values({500, 1000000, 2000000}));
  auto key_ranks           = make_shuffled_key_ranks_vector(num_items, CUB_SEED(2));
  thrust::device_vector<key_t> keys_in_out(num_items);
  thrust::transform(key_ranks.begin(), key_ranks.end(), keys_in_out.begin(), rank_to_key_op_t<offset_t, key_t>{});

  // Perform sort
  sort_keys(thrust::raw_pointer_cast(keys_in_out.data()), num_items, custom_less_op_t{});

  // Verify results
  auto key_ranks_it     = thrust::make_counting_iterator(offset_t{});
  auto keys_expected_it = thrust::make_transform_iterator(key_ranks_it, rank_to_key_op_t<offset_t, key_t>{});
  bool results_equal    = thrust::equal(keys_in_out.cbegin(), keys_in_out.cend(), keys_expected_it);
  REQUIRE(results_equal == true);
}

CUB_TEST("DeviceMergeSort::StableSortKeysCopy works and performs a stable sort when there are a lot sort-keys that "
         "compare equal",
         "[merge][sort][device]")
{
  using key_t    = c2h::custom_type_t<c2h::equal_comparable_t, c2h::less_comparable_t>;
  using offset_t = std::size_t;

  // Prepare input (generate a items that compare equally to check for stability of sort)
  const offset_t num_items = GENERATE_COPY(take(2, random(1, 1000000)), values({500, 1000000, 2000000}));
  thrust::device_vector<offset_t> key_ranks(num_items);
  c2h::gen(CUB_SEED(2), key_ranks, offset_t{}, static_cast<offset_t>(128));
  thrust::device_vector<key_t> keys_in(num_items);
  auto key_value_it = thrust::make_counting_iterator(offset_t{});
  auto key_init_it  = thrust::make_zip_iterator(key_ranks.begin(), key_value_it);
  thrust::transform(key_init_it, key_init_it + num_items, keys_in.begin(), tuple_to_custom_op_t<key_t>{});

  // Perform sort
  thrust::device_vector<key_t> keys_out(num_items, rank_to_key_op_t<offset_t, key_t>{}(42));
  stable_sort_keys_copy(
    thrust::raw_pointer_cast(keys_in.data()), thrust::raw_pointer_cast(keys_out.data()), num_items, custom_less_op_t{});

  // Verify results
  thrust::host_vector<key_t> keys_expected(keys_in);
  std::stable_sort(keys_expected.begin(), keys_expected.end(), custom_less_op_t{});

  REQUIRE(keys_expected == keys_out);
}

CUB_TEST("DeviceMergeSort::StableSortKeys works", "[merge][sort][device]")
{
  using key_t    = c2h::custom_type_t<c2h::equal_comparable_t, c2h::less_comparable_t>;
  using offset_t = std::int32_t;

  // Prepare input
  const offset_t num_items = GENERATE_COPY(take(2, random(1, 1000000)), values({500, 1000000, 2000000}));
  thrust::device_vector<key_t> keys_in_out(num_items);
  c2h::gen(CUB_SEED(2), keys_in_out);

  // Perform sort
  stable_sort_keys(thrust::raw_pointer_cast(keys_in_out.data()), num_items, custom_less_op_t{});

  // Verify results
  thrust::host_vector<key_t> keys_expected(keys_in_out);
  std::stable_sort(keys_expected.begin(), keys_expected.end(), custom_less_op_t{});

  REQUIRE(keys_expected == keys_in_out);
}

CUB_TEST("DeviceMergeSort::SortPairsCopy works", "[merge][sort][device]", wide_key_types)
{
  using key_t    = typename c2h::get<0, TestType>;
  using offset_t = std::int32_t;

  // Prepare input
  const offset_t num_items = GENERATE_COPY(take(2, random(1, 1000000)), values({500, 1000000, 2000000}));
  auto key_ranks           = make_shuffled_key_ranks_vector(num_items, CUB_SEED(2));
  thrust::device_vector<key_t> keys_in(num_items);
  thrust::transform(key_ranks.begin(), key_ranks.end(), keys_in.begin(), rank_to_key_op_t<offset_t, key_t>{});

  // Perform sort
  thrust::device_vector<key_t> keys_out(num_items, static_cast<key_t>(42));
  thrust::device_vector<offset_t> values_out(num_items, static_cast<offset_t>(42));
  sort_pairs_copy(
    thrust::raw_pointer_cast(keys_in.data()),
    thrust::raw_pointer_cast(key_ranks.data()),
    thrust::raw_pointer_cast(keys_out.data()),
    thrust::raw_pointer_cast(values_out.data()),
    num_items,
    custom_less_op_t{});

  // Verify results
  auto key_ranks_it       = thrust::make_counting_iterator(offset_t{});
  auto keys_expected_it   = thrust::make_transform_iterator(key_ranks_it, rank_to_key_op_t<offset_t, key_t>{});
  auto values_expected_it = thrust::make_counting_iterator(offset_t{});
  bool keys_equal         = thrust::equal(keys_out.cbegin(), keys_out.cend(), keys_expected_it);
  bool values_equal       = thrust::equal(values_out.cbegin(), values_out.cend(), values_expected_it);
  REQUIRE(keys_equal == true);
  REQUIRE(values_equal == true);
}

CUB_TEST("DeviceMergeSort::SortPairs works", "[merge][sort][device]", wide_key_types)
{
  using key_t    = typename c2h::get<0, TestType>;
  using offset_t = std::int32_t;

  // Prepare input
  const offset_t num_items = GENERATE_COPY(take(2, random(1, 1000000)), values({500, 1000000, 2000000}));
  auto key_ranks           = make_shuffled_key_ranks_vector(num_items, CUB_SEED(2));
  thrust::device_vector<key_t> keys_in_out(num_items);
  thrust::transform(key_ranks.begin(), key_ranks.end(), keys_in_out.begin(), rank_to_key_op_t<offset_t, key_t>{});

  // Perform sort
  sort_pairs(thrust::raw_pointer_cast(keys_in_out.data()),
             thrust::raw_pointer_cast(key_ranks.data()),
             num_items,
             custom_less_op_t{});

  // Verify results
  auto key_ranks_it       = thrust::make_counting_iterator(offset_t{});
  auto keys_expected_it   = thrust::make_transform_iterator(key_ranks_it, rank_to_key_op_t<offset_t, key_t>{});
  auto values_expected_it = thrust::make_counting_iterator(offset_t{});
  bool keys_equal         = thrust::equal(keys_in_out.cbegin(), keys_in_out.cend(), keys_expected_it);
  bool values_equal       = thrust::equal(key_ranks.cbegin(), key_ranks.cend(), values_expected_it);
  REQUIRE(keys_equal == true);
  REQUIRE(values_equal == true);
}

CUB_TEST(
  "DeviceMergeSort::StableSortPairs works and performs a stable sort", "[merge][sort][device]", key_types, value_types)
{
  using key_t    = typename c2h::get<0, TestType>;
  using data_t   = typename c2h::get<1, TestType>;
  using offset_t = std::int32_t;

  // Prepare input
  const offset_t num_items = GENERATE_COPY(take(2, random(1, 1000000)), values({500, 1000000, 2000000}));
  thrust::device_vector<key_t> keys_in_out(num_items);
  thrust::device_vector<data_t> values_in_out(num_items);
  c2h::gen(CUB_SEED(2), keys_in_out);
  c2h::gen(CUB_SEED(1), values_in_out);

  // Prepare host data for verification
  thrust::host_vector<key_t> keys_expected(keys_in_out);
  thrust::host_vector<data_t> values_expected(values_in_out);
  auto zipped_expected_it = thrust::make_zip_iterator(keys_expected.begin(), values_expected.begin());
  std::stable_sort(zipped_expected_it, zipped_expected_it + num_items, compare_first_lt_op_t{});

  // Perform sort
  stable_sort_pairs(thrust::raw_pointer_cast(keys_in_out.data()),
                    thrust::raw_pointer_cast(values_in_out.data()),
                    num_items,
                    custom_less_op_t{});

  REQUIRE(keys_expected == keys_in_out);
  REQUIRE(values_expected == values_in_out);
}
