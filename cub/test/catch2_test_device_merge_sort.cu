// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_merge_sort.cuh>

#include <thrust/copy.h>
#include <thrust/equal.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/random.h>
#include <thrust/reverse.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>

#include <cuda/buffer>
#include <cuda/iterator>
#include <cuda/std/iterator>
#include <cuda/std/limits>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/stream>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <ostream>

#include "catch2_large_array_sort_helper.cuh"
#include "catch2_test_device_merge_sort_common.cuh"
#include "catch2_test_launch_helper.h"
#include "cub_test_macros.h"
#include <c2h/buffer_generators.cuh>
#include <c2h/buffer_utils.cuh>
#include <c2h/catch2_allocation_failure.h>
#include <c2h/catch2_test_helper.h>
#include <c2h/checked_memory_resource.cuh>
#include <c2h/custom_type.h>
#include <c2h/device_and_stream.h>
#include <c2h/device_policy.h>

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

template <typename OffsetT, typename KeyT = std::uint8_t>
struct type_tuple
{
  using offset_t = OffsetT;
  using key_t    = KeyT;
};
using offset_types =
  c2h::type_list<type_tuple<std::int16_t>,
                 type_tuple<std::int32_t, std::uint32_t>,
                 type_tuple<std::uint32_t>,
                 type_tuple<std::uint64_t>>;

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
  __device__ __host__ CustomT operator()(const cuda::std::tuple<KeyT, ValueT>& val)
  {
    CustomT custom_val{};
    custom_val.key = static_cast<std::size_t>(cuda::std::get<0>(val));
    custom_val.val = static_cast<std::size_t>(cuda::std::get<1>(val));
    return custom_val;
  }
};

/**
 * @brief In combination with a counting iterator, this function object generates a sequence that wraps around after
 * reaching `UnsignedIntegralKeyT`'s maximum value. E.g., for a uint8_t this maps the sequence of indexes [0, ..., 254,
 * 255, 256, 256] -> [0, ..., 254, 255, 0, 1]
 */
template <typename UnsignedIntegralKeyT>
struct index_to_key_value_op
{
  static constexpr std::size_t max_key_value =
    static_cast<std::size_t>(cuda::std::numeric_limits<UnsignedIntegralKeyT>::max());
  static constexpr std::size_t lowest_key_value =
    static_cast<std::size_t>(cuda::std::numeric_limits<UnsignedIntegralKeyT>::lowest());
  static_assert(sizeof(UnsignedIntegralKeyT) < sizeof(std::size_t),
                "Calculation of num_distinct_key_values would overflow");
  static constexpr std::size_t num_distinct_key_values = (max_key_value - lowest_key_value + std::size_t{1ULL});

  __device__ __host__ UnsignedIntegralKeyT operator()(std::size_t index)
  {
    return static_cast<UnsignedIntegralKeyT>(index % num_distinct_key_values);
  }
};

/**
 * @brief In combination with a counting iterator, this function object helps generate the expected sorted order for a
 * sequence generated with `index_to_key_value_op`. It respects how many remainder items there are following the last
 * occurrence of `UnsignedIntegralKeyT`'s max value. E.g., when we use `num_total_items` of `260` with an `uint8_t`, the
 * input sequence was:
 * [0, ..., 254, 255, 256, 257, 258, 259] <= index
 * [0, ..., 254, 255,   0,   1,   2,   3] <= input
 * -----------------
 * [0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 6, ..., 255] <= expected sorted order (note, [0, 3] occur twice)
 */
template <typename UnsignedIntegralKeyT>
class index_to_expected_key_op
{
  static constexpr std::size_t max_key_value =
    static_cast<std::size_t>(cuda::std::numeric_limits<UnsignedIntegralKeyT>::max());
  static constexpr std::size_t lowest_key_value =
    static_cast<std::size_t>(cuda::std::numeric_limits<UnsignedIntegralKeyT>::lowest());
  static_assert(sizeof(UnsignedIntegralKeyT) < sizeof(std::size_t),
                "Calculation of num_distinct_key_values would overflow");
  static constexpr std::size_t num_distinct_key_values = (max_key_value - lowest_key_value + std::size_t{1ULL});

  // item_count / num_distinct_key_values
  std::size_t expected_count_per_item;
  // num remainder items: item_count%num_distinct_key_values
  std::size_t num_remainder_items;
  // remainder item_count: expected_count_per_item+1
  std::size_t remainder_item_count;

public:
  index_to_expected_key_op(std::size_t num_total_items)
      : expected_count_per_item(num_total_items / num_distinct_key_values)
      , num_remainder_items(num_total_items % num_distinct_key_values)
      , remainder_item_count(expected_count_per_item + std::size_t{1ULL})
  {}

  __device__ __host__ UnsignedIntegralKeyT operator()(std::size_t index)
  {
    // The first (num_remainder_items * remainder_item_count) are items that appear once more often than the items that
    // follow remainder_items_offset
    const std::size_t remainder_items_offset = num_remainder_items * remainder_item_count;

    UnsignedIntegralKeyT target_item_index =
      (index <= remainder_items_offset)
        ?
        // This is one of the remainder items
        static_cast<UnsignedIntegralKeyT>(index / remainder_item_count)
        :
        // This is an item that appears exactly expected_count_per_item times
        static_cast<UnsignedIntegralKeyT>(
          num_remainder_items + ((index - remainder_items_offset) / expected_count_per_item));
    return target_item_index;
  }
};

/**
 * Generates shuffled key ranks. E.g., for a buffer of size 5: [4, 2, 3, 1, 0]
 */
template <typename OffsetT>
cuda::device_buffer<OffsetT> make_shuffled_key_ranks(cuda::stream_ref stream, OffsetT num_items, c2h::seed_t seed)
{
  auto key_ranks    = c2h::make_device_buffer<OffsetT>(stream, num_items, cuda::no_init);
  const auto policy = c2h::device_policy.on(stream.get());
  thrust::sequence(policy, key_ranks.data(), key_ranks.data() + key_ranks.size());
  thrust::shuffle(policy,
                  key_ranks.data(),
                  key_ranks.data() + key_ranks.size(),
                  thrust::default_random_engine{static_cast<unsigned int>(seed.get())});
  return key_ranks;
}

CUB_TEST("DeviceMergeSort::SortKeysCopy works",
         "[merge][sort][device][skip-cs-racecheck][skip-cs-memcheck]",
         CUB_SMALL,
         wide_key_types)
{
  const auto stream = c2h::make_current_device_stream();

  using key_t    = typename c2h::get<0, TestType>;
  using offset_t = std::int32_t;

  // Prepare input
  // Use c2h::adjust_seed_count to reduce runtime on sanitizers.
  const offset_t num_items =
    GENERATE_COPY(take(c2h::adjust_seed_count(2), random(1, 1000000)), values({0, 500, 1000000, 2000000}));
  auto key_ranks = make_shuffled_key_ranks(stream, num_items, C2H_SEED(2));
  auto keys_in   = c2h::make_device_buffer<key_t>(stream, num_items, cuda::no_init);
  thrust::transform(
    c2h::device_policy.on(stream.get()),
    key_ranks.data(),
    key_ranks.data() + key_ranks.size(),
    keys_in.data(),
    rank_to_key_op_t<offset_t, key_t>{});

  // Perform sort
  auto keys_out = c2h::make_device_buffer<key_t>(stream, num_items, static_cast<key_t>(42));
  sort_keys_copy(keys_in.data(), keys_out.data(), num_items, custom_less_op_t{}, stream);

  // Verify results
  const auto key_ranks_it     = cuda::counting_iterator(offset_t{});
  const auto keys_expected_it = cuda::transform_iterator(key_ranks_it, rank_to_key_op_t<offset_t, key_t>{});
  const bool results_equal    = thrust::equal(
    c2h::device_policy.on(stream.get()), keys_out.data(), keys_out.data() + keys_out.size(), keys_expected_it);
  REQUIRE(results_equal == true);
}

CUB_TEST("DeviceMergeSort::SortKeys works", "[merge][sort][device]", CUB_SMALL, wide_key_types)
{
  const auto stream = c2h::make_current_device_stream();

  using key_t    = typename c2h::get<0, TestType>;
  using offset_t = std::int32_t;

  // Prepare input
  // Use c2h::adjust_seed_count to reduce runtime on sanitizers.
  const offset_t num_items =
    GENERATE_COPY(take(c2h::adjust_seed_count(2), random(1, 1000000)), values({0, 500, 1000000, 2000000}));
  auto key_ranks   = make_shuffled_key_ranks(stream, num_items, C2H_SEED(2));
  auto keys_in_out = c2h::make_device_buffer<key_t>(stream, num_items, cuda::no_init);
  thrust::transform(
    c2h::device_policy.on(stream.get()),
    key_ranks.data(),
    key_ranks.data() + key_ranks.size(),
    keys_in_out.data(),
    rank_to_key_op_t<offset_t, key_t>{});

  // Perform sort
  sort_keys(keys_in_out.data(), num_items, custom_less_op_t{}, stream);

  // Verify results
  auto key_ranks_it        = cuda::counting_iterator(offset_t{});
  auto keys_expected_it    = cuda::transform_iterator(key_ranks_it, rank_to_key_op_t<offset_t, key_t>{});
  const bool results_equal = thrust::equal(
    c2h::device_policy.on(stream.get()), keys_in_out.data(), keys_in_out.data() + keys_in_out.size(), keys_expected_it);
  REQUIRE(results_equal == true);
}

CUB_TEST("DeviceMergeSort::StableSortKeysCopy works and performs a stable sort when there are a lot sort-keys that "
         "compare equal",
         "[merge][sort][device][skip-cs-racecheck][skip-cs-memcheck]",
         CUB_SMALL)
{
  const auto stream = c2h::make_current_device_stream();

  using key_t    = c2h::custom_type_t<c2h::equal_comparable_t, c2h::less_comparable_t>;
  using offset_t = std::size_t;

  // Prepare input (generate a items that compare equally to check for stability of sort)
  // Use c2h::adjust_seed_count to reduce runtime on sanitizers.
  const offset_t num_items =
    GENERATE_COPY(take(c2h::adjust_seed_count(2), random(1, 1000000)), values({0, 500, 1000000, 2000000}));
  auto key_ranks = c2h::make_device_buffer<offset_t>(stream, num_items, cuda::no_init);
  c2h::gen(C2H_SEED(2), key_ranks, offset_t{}, static_cast<offset_t>(128));
  auto keys_in      = c2h::make_device_buffer<key_t>(stream, num_items, cuda::no_init);
  auto key_value_it = cuda::counting_iterator(offset_t{});
  auto key_init_it  = cuda::make_zip_iterator(key_ranks.data(), key_value_it);
  thrust::transform(
    c2h::device_policy.on(stream.get()),
    key_init_it,
    key_init_it + num_items,
    keys_in.data(),
    tuple_to_custom_op_t<key_t>{});

  // Perform sort
  auto keys_out = c2h::make_device_buffer<key_t>(stream, num_items, rank_to_key_op_t<offset_t, key_t>{}(42));
  stable_sort_keys_copy(keys_in.data(), keys_out.data(), num_items, custom_less_op_t{}, stream);

  // Verify results
  auto keys_expected = c2h::make_host_buffer<key_t>(stream, num_items, cuda::no_init);
  c2h::copy_to_host(stream, keys_in, keys_expected);
  std::stable_sort(keys_expected.data(), keys_expected.data() + keys_expected.size(), custom_less_op_t{});

  REQUIRE_THAT_QUIET(keys_out, c2h::Equals(stream, keys_expected));
}

CUB_TEST("DeviceMergeSort::StableSortKeys works", "[merge][sort][device]", CUB_SMALL)
{
  const auto stream = c2h::make_current_device_stream();

  using key_t    = c2h::custom_type_t<c2h::equal_comparable_t, c2h::less_comparable_t>;
  using offset_t = std::int32_t;

  // Prepare input
  // Use c2h::adjust_seed_count to reduce runtime on sanitizers.
  const offset_t num_items =
    GENERATE_COPY(take(c2h::adjust_seed_count(2), random(1, 1000000)), values({0, 500, 1000000, 2000000}));

  auto key_buffers    = c2h::gen_buffers<key_t>(stream, C2H_SEED(2), num_items);
  auto& keys_in_out   = key_buffers.d_items;
  auto& keys_expected = key_buffers.h_items;
  std::stable_sort(keys_expected.data(), keys_expected.data() + keys_expected.size(), custom_less_op_t{});

  stable_sort_keys(keys_in_out.data(), num_items, custom_less_op_t{}, stream);

  REQUIRE_THAT_QUIET(keys_in_out, c2h::Equals(stream, keys_expected));
}

CUB_TEST("DeviceMergeSort::SortPairsCopy works",
         "[merge][sort][device][skip-cs-racecheck][skip-cs-memcheck]",
         CUB_SMALL,
         wide_key_types)
{
  const auto stream = c2h::make_current_device_stream();

  using key_t    = typename c2h::get<0, TestType>;
  using offset_t = std::int32_t;

  // Prepare input
  // Use c2h::adjust_seed_count to reduce runtime on sanitizers.
  const offset_t num_items =
    GENERATE_COPY(take(c2h::adjust_seed_count(2), random(1, 1000000)), values({0, 500, 1000000, 2000000}));
  auto key_ranks = make_shuffled_key_ranks(stream, num_items, C2H_SEED(2));
  auto keys_in   = c2h::make_device_buffer<key_t>(stream, num_items, cuda::no_init);
  thrust::transform(
    c2h::device_policy.on(stream.get()),
    key_ranks.data(),
    key_ranks.data() + key_ranks.size(),
    keys_in.data(),
    rank_to_key_op_t<offset_t, key_t>{});

  // Perform sort
  auto keys_out   = c2h::make_device_buffer<key_t>(stream, num_items, static_cast<key_t>(42));
  auto values_out = c2h::make_device_buffer<offset_t>(stream, num_items, static_cast<offset_t>(42));
  sort_pairs_copy(
    keys_in.data(), key_ranks.data(), keys_out.data(), values_out.data(), num_items, custom_less_op_t{}, stream);

  // Verify results
  auto key_ranks_it       = cuda::counting_iterator(offset_t{});
  auto keys_expected_it   = cuda::transform_iterator(key_ranks_it, rank_to_key_op_t<offset_t, key_t>{});
  auto values_expected_it = cuda::counting_iterator(offset_t{});
  const bool keys_equal   = thrust::equal(
    c2h::device_policy.on(stream.get()), keys_out.data(), keys_out.data() + keys_out.size(), keys_expected_it);
  const bool values_equal = thrust::equal(
    c2h::device_policy.on(stream.get()), values_out.data(), values_out.data() + values_out.size(), values_expected_it);
  REQUIRE(keys_equal == true);
  REQUIRE(values_equal == true);
}

CUB_TEST("DeviceMergeSort::SortPairs works", "[merge][sort][device]", CUB_SMALL, wide_key_types)
{
  const auto stream = c2h::make_current_device_stream();

  using key_t    = typename c2h::get<0, TestType>;
  using offset_t = std::int32_t;

  // Prepare input
  // Use c2h::adjust_seed_count to reduce runtime on sanitizers.
  const offset_t num_items =
    GENERATE_COPY(take(c2h::adjust_seed_count(2), random(1, 1000000)), values({0, 500, 1000000, 2000000}));
  auto key_ranks   = make_shuffled_key_ranks(stream, num_items, C2H_SEED(2));
  auto keys_in_out = c2h::make_device_buffer<key_t>(stream, num_items, cuda::no_init);
  thrust::transform(
    c2h::device_policy.on(stream.get()),
    key_ranks.data(),
    key_ranks.data() + key_ranks.size(),
    keys_in_out.data(),
    rank_to_key_op_t<offset_t, key_t>{});

  // Perform sort
  sort_pairs(keys_in_out.data(), key_ranks.data(), num_items, custom_less_op_t{}, stream);

  // Verify results
  auto key_ranks_it       = cuda::counting_iterator(offset_t{});
  auto keys_expected_it   = cuda::transform_iterator(key_ranks_it, rank_to_key_op_t<offset_t, key_t>{});
  auto values_expected_it = cuda::counting_iterator(offset_t{});
  const bool keys_equal   = thrust::equal(
    c2h::device_policy.on(stream.get()), keys_in_out.data(), keys_in_out.data() + keys_in_out.size(), keys_expected_it);
  const bool values_equal = thrust::equal(
    c2h::device_policy.on(stream.get()), key_ranks.data(), key_ranks.data() + key_ranks.size(), values_expected_it);
  REQUIRE(keys_equal == true);
  REQUIRE(values_equal == true);
}

CUB_TEST("DeviceMergeSort::StableSortPairs works and performs a stable sort",
         "[merge][sort][device]",
         CUB_SMALL,
         key_types,
         value_types)
{
  const auto stream = c2h::make_current_device_stream();

  using key_t    = typename c2h::get<0, TestType>;
  using data_t   = typename c2h::get<1, TestType>;
  using offset_t = std::int32_t;

  // Prepare input
  // Use c2h::adjust_seed_count to reduce runtime on sanitizers.
  const offset_t num_items =
    GENERATE_COPY(take(c2h::adjust_seed_count(2), random(1, 1000000)), values({0, 500, 1000000, 2000000}));

  auto key_buffers      = c2h::gen_buffers<key_t>(stream, C2H_SEED(2), num_items);
  auto value_buffers    = c2h::gen_buffers<data_t>(stream, C2H_SEED(1), num_items);
  auto& keys_in_out     = key_buffers.d_items;
  auto& keys_expected   = key_buffers.h_items;
  auto& values_in_out   = value_buffers.d_items;
  auto& values_expected = value_buffers.h_items;

  // Use thrust::make_zip_iterator instead of cuda::make_zip_iterator as a work-around
  // See https://github.com/NVIDIA/cccl/issues/6400
  auto zipped_expected_it = thrust::make_zip_iterator(keys_expected.data(), values_expected.data());
  std::stable_sort(zipped_expected_it, zipped_expected_it + num_items, compare_first_lt_op_t{});

  // Perform sort
  stable_sort_pairs(keys_in_out.data(), values_in_out.data(), num_items, custom_less_op_t{}, stream);

  REQUIRE_THAT_QUIET(keys_in_out, c2h::Equals(stream, keys_expected));
  REQUIRE_THAT_QUIET(values_in_out, c2h::Equals(stream, values_expected));
}

CUB_TEST("DeviceMergeSort::StableSortPairs works for large inputs",
         "[merge][sort][device][skip-cs-initcheck][skip-cs-racecheck][skip-cs-synccheck]",
         CUB_LARGE,
         offset_types)
{
  using testing_types_tuple = c2h::get<0, TestType>;
  using key_t               = typename testing_types_tuple::key_t;
  using offset_t            = typename testing_types_tuple::offset_t;

  // Clamp 64-bit offset type problem sizes to just slightly larger than 2^32 items
  auto num_items_ull = std::min(static_cast<std::size_t>(cuda::std::numeric_limits<offset_t>::max()) - 1,
                                cuda::std::numeric_limits<std::uint32_t>::max() + static_cast<std::size_t>(2000000ULL));
  const offset_t num_items = static_cast<offset_t>(num_items_ull);

  SECTION("Random")
  {
    try
    {
      // Initialize random input data
      const auto stream = c2h::make_current_device_stream();
      large_array_sort_helper<key_t> arrays(stream);
      constexpr bool is_descending = false;
      arrays.initialize_for_unstable_key_sort(C2H_SEED(1), num_items, is_descending);

      // Free extra data buffer used during initialization, but not needed for the "in-place" merge sort
      arrays.deallocate_outputs();

      // Perform sort
      stable_sort_keys(arrays.input_keys_data(), num_items, custom_less_op_t{}, stream);

      // Verify results
      arrays.verify_unstable_key_sort(num_items, is_descending);
    }
    catch (...)
    {
      c2h::skip_if_allocation_failure();
    }
  }

  SECTION("Pre-sorted input")
  {
    try
    {
      const auto stream = c2h::make_current_device_stream();
      auto keys_in_out  = c2h::make_device_buffer<key_t>(stream, num_items, cuda::no_init);
      const auto policy = c2h::device_policy.on(stream.get());

      // Pre-populated array with a constant value
      auto counting_it = cuda::counting_iterator(std::size_t{0});
      thrust::copy(policy, counting_it, counting_it + num_items, keys_in_out.data());

      // Perform sort
      stable_sort_keys(keys_in_out.data(), num_items, custom_less_op_t{}, stream);

      // Perform comparison
      auto expected_result_it =
        cuda::transform_iterator(cuda::counting_iterator(std::size_t{}), index_to_expected_key_op<key_t>(num_items));
      const bool is_correct =
        thrust::equal(policy, keys_in_out.data(), keys_in_out.data() + keys_in_out.size(), expected_result_it);
      REQUIRE(is_correct == true);
    }
    catch (...)
    {
      c2h::skip_if_allocation_failure();
    }
  }

  SECTION("Reverse-sorted input")
  {
    try
    {
      const auto stream = c2h::make_current_device_stream();
      auto keys_in_out  = c2h::make_device_buffer<key_t>(stream, num_items, cuda::no_init);
      const auto policy = c2h::device_policy.on(stream.get());

      auto counting_it   = cuda::counting_iterator(std::size_t{0});
      auto key_value_it  = cuda::transform_iterator(counting_it, index_to_key_value_op<key_t>{});
      auto rev_sorted_it = cuda::std::make_reverse_iterator(key_value_it + num_items);
      thrust::copy(policy, rev_sorted_it, rev_sorted_it + num_items, keys_in_out.data());

      // Perform sort
      stable_sort_keys(keys_in_out.data(), num_items, custom_less_op_t{}, stream);

      // Perform comparison
      auto expected_result_it =
        cuda::transform_iterator(cuda::counting_iterator(std::size_t{}), index_to_expected_key_op<key_t>(num_items));
      const bool is_correct =
        thrust::equal(policy, keys_in_out.data(), keys_in_out.data() + keys_in_out.size(), expected_result_it);
      REQUIRE(is_correct == true);
    }
    catch (...)
    {
      c2h::skip_if_allocation_failure();
    }
  }
}

// A trivially copyable type without a default constructor
struct not_default_constructible_t
{
  int value;

  __host__ __device__ explicit not_default_constructible_t(int v)
      : value(v)
  {}

  __host__ __device__ friend bool
  operator==(const not_default_constructible_t& lhs, const not_default_constructible_t& rhs)
  {
    return lhs.value == rhs.value;
  }

  __host__ __device__ friend bool
  operator!=(const not_default_constructible_t& lhs, const not_default_constructible_t& rhs)
  {
    return !(lhs == rhs);
  }

  __host__ __device__ friend bool
  operator<(const not_default_constructible_t& lhs, const not_default_constructible_t& rhs)
  {
    return lhs.value < rhs.value;
  }

  // for printing mismatches
  friend std::ostream& operator<<(std::ostream& os, const not_default_constructible_t& v)
  {
    return os << v.value;
  }
};

static_assert(!cuda::std::is_default_constructible_v<not_default_constructible_t>);
static_assert(cuda::std::is_trivially_copyable_v<not_default_constructible_t>);

struct make_not_default_constructible_op_t
{
  __host__ __device__ not_default_constructible_t operator()(int i) const
  {
    return not_default_constructible_t{i};
  }
};

// Device storage for not_default_constructible_t, which thrust::device_vector cannot hold. Elements are constructed on
// the device. Holds 0, 1, ..., num_items - 1, or the same values in reverse order.
inline auto make_ndc_buffer(cuda::stream_ref stream, int num_items, bool reversed = false)
{
  auto buffer       = c2h::make_device_buffer<not_default_constructible_t>(stream, num_items, cuda::no_init);
  const auto policy = c2h::device_policy.on(stream.get());
  thrust::tabulate(policy, buffer.data(), buffer.data() + num_items, make_not_default_constructible_op_t{});
  if (reversed)
  {
    thrust::reverse(policy, buffer.data(), buffer.data() + num_items);
  }
  stream.sync();
  return buffer;
}

CUB_TEST("DeviceMergeSort::SortKeysCopy works with non-default-constructible keys", "[merge][sort][device]", CUB_SMALL)
{
  const cuda::stream stream = c2h::make_current_device_stream();
  const int num_items       = GENERATE(1, 500, 1000000);
  auto keys_in              = make_ndc_buffer(stream, num_items, true);
  auto keys_out             = c2h::make_device_buffer<not_default_constructible_t>(stream, num_items, cuda::no_init);

  sort_keys_copy(keys_in.data(), keys_out.data(), num_items, custom_less_op_t{}, stream);

  REQUIRE_THAT(keys_out, Equals(make_ndc_buffer(stream, num_items)));
}

CUB_TEST("DeviceMergeSort::SortKeys works with non-default-constructible keys", "[merge][sort][device]", CUB_SMALL)
{
  const cuda::stream stream = c2h::make_current_device_stream();
  const int num_items       = GENERATE(1, 500, 1000000);
  auto keys_in_out          = make_ndc_buffer(stream, num_items, true);

  sort_keys(keys_in_out.data(), num_items, custom_less_op_t{}, stream);

  REQUIRE_THAT(keys_in_out, Equals(make_ndc_buffer(stream, num_items)));
}

CUB_TEST("DeviceMergeSort::SortPairsCopy works with non-default-constructible keys and values",
         "[merge][sort][device]",
         CUB_SMALL)
{
  const cuda::stream stream = c2h::make_current_device_stream();
  const int num_items       = GENERATE(1, 500, 1000000);
  auto keys_in              = make_ndc_buffer(stream, num_items, true);
  auto values_in            = make_ndc_buffer(stream, num_items);
  auto keys_out             = c2h::make_device_buffer<not_default_constructible_t>(stream, num_items, cuda::no_init);
  auto values_out           = c2h::make_device_buffer<not_default_constructible_t>(stream, num_items, cuda::no_init);

  sort_pairs_copy(
    keys_in.data(), values_in.data(), keys_out.data(), values_out.data(), num_items, custom_less_op_t{}, stream);

  REQUIRE_THAT(keys_out, Equals(make_ndc_buffer(stream, num_items)));
  // values were ascending while keys were descending, so values end up descending
  REQUIRE_THAT(values_out, Equals(make_ndc_buffer(stream, num_items, true)));
}

CUB_TEST("DeviceMergeSort::SortPairs works with non-default-constructible keys and values",
         "[merge][sort][device]",
         CUB_SMALL)
{
  const cuda::stream stream = c2h::make_current_device_stream();
  const int num_items       = GENERATE(1, 500, 1000000);
  auto keys_in_out          = make_ndc_buffer(stream, num_items, true);
  auto values_in_out        = make_ndc_buffer(stream, num_items);

  sort_pairs(keys_in_out.data(), values_in_out.data(), num_items, custom_less_op_t{}, stream);

  REQUIRE_THAT(keys_in_out, Equals(make_ndc_buffer(stream, num_items)));
  REQUIRE_THAT(values_in_out, Equals(make_ndc_buffer(stream, num_items, true)));
}
