// SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_run_length_encode.cuh>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>

#include <cuda/iterator>

#include <algorithm>
#include <limits>
#include <numeric>
#include <random>

#include "catch2_large_problem_helper.cuh"
#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.h>

DECLARE_LAUNCH_WRAPPER(cub::DeviceRunLengthEncode::Encode, run_length_encode);

// %PARAM% TEST_LAUNCH lid 0:1:2

using all_types =
  c2h::type_list<std::uint8_t,
                 std::uint64_t,
                 std::int8_t,
                 std::int64_t,
                 ulonglong2,
                 c2h::custom_type_t<c2h::equal_comparable_t>>;

using types = c2h::type_list<std::uint32_t, std::int8_t>;

// List of offset types to be used for testing large number of items
using offset_types = c2h::type_list<std::int64_t, std::int32_t>;

C2H_TEST("DeviceRunLengthEncode::Encode can handle empty input", "[device][run_length_encode]")
{
  constexpr int num_items = 0;
  c2h::device_vector<int> in(num_items);
  c2h::device_vector<int> out_num_runs(1, 42);

  // Note intentionally no discard_iterator as we want to ensure nothing is written to the output arrays
  run_length_encode(
    in.begin(),
    static_cast<int*>(nullptr),
    static_cast<int*>(nullptr),
    thrust::raw_pointer_cast(out_num_runs.data()),
    num_items);

  REQUIRE(out_num_runs.front() == num_items);
}

template <typename OffsetT>
struct repeat_item_gen_op
{
  OffsetT num_small_runs;

  __host__ __device__ OffsetT operator()(OffsetT index) const
  {
    return (index < num_small_runs) ? index : (num_small_runs + (index - num_small_runs) / 2);
  }
};

template <typename OffsetT, typename RunLengthT>
struct run_to_run_length_op
{
  OffsetT num_small_runs;

  __host__ __device__ RunLengthT operator()(OffsetT index) const
  {
    return (index < num_small_runs) ? RunLengthT{1} : RunLengthT{2};
  }
};

class non_default_constructible_iterator_op
{
  const std::size_t k;

public:
  // Not default constructible
  non_default_constructible_iterator_op() = delete;
  __host__ __device__ non_default_constructible_iterator_op(std::size_t k)
      : k{k}
  {}

  __host__ __device__ std::size_t operator()(std::size_t index) const
  {
    return index + k;
  }
};

constexpr long long segment_grid_tile_size(std::size_t key_size)
{
  return (key_size >= 16) ? 2048 : (key_size == 8) ? 4096 : 8192;
}

template <class KeyT>
KeyT make_segment_key(int value)
{
  return KeyT(value);
}

template <>
ulonglong2 make_segment_key<ulonglong2>(int value)
{
  return {static_cast<unsigned long long>(value), static_cast<unsigned long long>(value)};
}

// max_seg > 0: run lengths uniform in [1, max_seg]; max_seg == 0: one constant key for the whole
// input; max_seg < 0: every run exactly -max_seg long (fixed run head positions)
template <class KeyT>
c2h::host_vector<KeyT> generate_segmented_keys(long long num_items, int max_seg, unsigned seed)
{
  c2h::host_vector<KeyT> keys(static_cast<std::size_t>(num_items));
  if (max_seg == 0)
  {
    thrust::fill(keys.begin(), keys.end(), make_segment_key<KeyT>(7));
    return keys;
  }
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> segment_length(1, std::max(1, max_seg));
  std::uniform_int_distribution<int> key_value(0, 1000000);
  long long i = 0;
  KeyT prev   = make_segment_key<KeyT>(-1);
  while (i < num_items)
  {
    const int run_length = (max_seg < 0) ? -max_seg : segment_length(rng);
    KeyT value           = make_segment_key<KeyT>(key_value(rng));
    // narrow key types can wrap onto the previous run's key; retry a few times to keep runs
    // distinct (a residual collision only merges two runs; the reference uses the final keys)
    for (int tries = 0; value == prev && tries < 8; ++tries)
    {
      value = make_segment_key<KeyT>(key_value(rng));
    }
    prev                = value;
    const long long end = std::min(i + run_length, num_items);
    for (; i < end; ++i)
    {
      keys[static_cast<std::size_t>(i)] = value;
    }
  }
  return keys;
}

template <class KeyT, class OffsetT>
void test_segmented_encode(long long num_items, int max_seg, int elem_offset, unsigned seed)
{
  CAPTURE(c2h::type_name<KeyT>(), c2h::type_name<OffsetT>(), num_items, max_seg, elem_offset, seed);

  const auto h_keys = generate_segmented_keys<KeyT>(num_items, max_seg, seed);

  c2h::host_vector<KeyT> ref_unique;
  c2h::host_vector<int> ref_counts;
  for (long long i = 0; i < num_items;)
  {
    long long j = i + 1;
    while (j < num_items && h_keys[static_cast<std::size_t>(j)] == h_keys[static_cast<std::size_t>(i)])
    {
      ++j;
    }
    ref_unique.push_back(h_keys[static_cast<std::size_t>(i)]);
    ref_counts.push_back(static_cast<int>(j - i));
    i = j;
  }
  const auto ref_num_runs = static_cast<OffsetT>(ref_unique.size());

  // exact-size input allocation: reading past num_items would trip the sanitizers. The
  // elem_offset elements in front of the input are set EQUAL to the first key: reading before
  // the input would extend the first run and fail the count comparison.
  c2h::device_vector<KeyT> d_keys_alloc(static_cast<std::size_t>(num_items) + elem_offset);
  thrust::copy(c2h::device_policy, h_keys.begin(), h_keys.end(), d_keys_alloc.begin() + elem_offset);
  thrust::fill(c2h::device_policy, d_keys_alloc.begin(), d_keys_alloc.begin() + elem_offset, h_keys.front());

  // exact-size outputs; counts are sentinel-filled with 0, which no run can produce
  c2h::device_vector<KeyT> d_unique(ref_unique.size(), make_segment_key<KeyT>(42424242));
  c2h::device_vector<int> d_counts(ref_counts.size(), 0);
  c2h::device_vector<OffsetT> d_num_runs(1, OffsetT{-1});

  run_length_encode(
    thrust::raw_pointer_cast(d_keys_alloc.data()) + elem_offset,
    thrust::raw_pointer_cast(d_unique.data()),
    thrust::raw_pointer_cast(d_counts.data()),
    thrust::raw_pointer_cast(d_num_runs.data()),
    static_cast<OffsetT>(num_items));

  REQUIRE(d_num_runs.front() == ref_num_runs);
  REQUIRE(c2h::host_vector<KeyT>(d_unique) == ref_unique);
  REQUIRE(c2h::host_vector<int>(d_counts) == ref_counts);
}

// the five key size classes: 1, 2, 4, 8 and 16 bytes
using segment_key_types = c2h::type_list<std::int8_t, std::int16_t, std::uint32_t, std::int64_t, ulonglong2>;

C2H_TEST("DeviceRunLengthEncode::Encode can handle a single element", "[device][run_length_encode]")
{
  constexpr int num_items = 1;
  c2h::device_vector<int> in(num_items, 42);
  c2h::device_vector<int> out_unique(num_items);
  c2h::device_vector<int> out_counts(num_items);
  c2h::device_vector<int> out_num_runs(num_items, -1);

  run_length_encode(in.begin(), out_unique.begin(), out_counts.begin(), out_num_runs.begin(), num_items);

  REQUIRE(out_unique.front() == 42);
  REQUIRE(out_counts.front() == 1);
  REQUIRE(out_num_runs.front() == num_items);
}

C2H_TEST("DeviceRunLengthEncode::Encode can handle different counting types", "[device][run_length_encode]")
{
  constexpr int num_items = 1;
  c2h::device_vector<int> in(num_items, 42);
  c2h::device_vector<int> out_unique(num_items);
  c2h::device_vector<cuda::std::size_t> out_counts(num_items);
  c2h::device_vector<std::int16_t> out_num_runs(num_items);

  run_length_encode(in.begin(), out_unique.begin(), out_counts.begin(), out_num_runs.begin(), num_items);

  REQUIRE(out_unique.front() == 42);
  REQUIRE(out_counts.front() == 1);
  REQUIRE(out_num_runs.front() == static_cast<std::int16_t>(num_items));
}

C2H_TEST("DeviceRunLengthEncode::Encode can handle all unique", "[device][run_length_encode]", types)
{
  using type = typename c2h::get<0, TestType>;

  constexpr int num_items = 10;
  c2h::device_vector<type> out_unique(num_items);
  c2h::device_vector<int> out_counts(num_items);
  c2h::device_vector<int> out_num_runs(1);

  run_length_encode(
    cuda::counting_iterator(type{}), out_unique.begin(), out_counts.begin(), out_num_runs.begin(), num_items);

  c2h::device_vector<type> reference_unique(num_items);
  thrust::sequence(c2h::device_policy, reference_unique.begin(), reference_unique.end(), type{}); // [0, 1, 2, ...,
                                                                                                  // num_items -1]
  c2h::device_vector<int> reference_counts(num_items, 1); // [1, 1, ..., 1]
  c2h::device_vector<int> reference_num_runs(1, num_items); // [num_items]

  REQUIRE(out_unique == reference_unique);
  REQUIRE(out_counts == reference_counts);
  REQUIRE(out_num_runs == reference_num_runs);
}

C2H_TEST("DeviceRunLengthEncode::Encode can handle all equal", "[device][run_length_encode]", types)
{
  using type = typename c2h::get<0, TestType>;

  constexpr int num_items = 10;
  c2h::device_vector<type> in(num_items, type{1});
  c2h::device_vector<type> out_unique(1);
  c2h::device_vector<int> out_counts(1);
  c2h::device_vector<int> out_num_runs(1);

  run_length_encode(in.begin(), out_unique.begin(), out_counts.begin(), out_num_runs.begin(), num_items);

  c2h::device_vector<type> reference_unique(1, type{1}); // [1]
  c2h::device_vector<int> reference_counts(1, num_items); // [num_items]
  c2h::device_vector<int> reference_num_runs(1, 1); // [1]

  REQUIRE(out_unique == reference_unique);
  REQUIRE(out_counts == reference_counts);
  REQUIRE(out_num_runs == reference_num_runs);
}

C2H_TEST("DeviceRunLengthEncode::Encode can handle iterators", "[device][run_length_encode]", all_types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<type> in(num_items);
  c2h::device_vector<type> out_unique(num_items);
  c2h::device_vector<int> out_counts(num_items);
  c2h::device_vector<int> out_num_runs(num_items);
  c2h::gen(C2H_SEED(2), in);

  run_length_encode(in.begin(), out_unique.begin(), out_counts.begin(), out_num_runs.begin(), num_items);

  // trim output
  out_unique.resize(out_num_runs.front());
  out_counts.resize(out_num_runs.front());

  c2h::host_vector<type> reference_out = in;
  reference_out.erase(std::unique(reference_out.begin(), reference_out.end()), reference_out.end());
  REQUIRE(out_unique == reference_out);
}

C2H_TEST("DeviceRunLengthEncode::Encode can handle pointers", "[device][run_length_encode]", types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<type> in(num_items);
  c2h::device_vector<type> out_unique(num_items);
  c2h::device_vector<int> out_counts(num_items);
  c2h::device_vector<int> out_num_runs(num_items);
  c2h::gen(C2H_SEED(2), in);

  run_length_encode(
    thrust::raw_pointer_cast(in.data()),
    thrust::raw_pointer_cast(out_unique.data()),
    thrust::raw_pointer_cast(out_counts.data()),
    thrust::raw_pointer_cast(out_num_runs.data()),
    num_items);

  // trim output
  out_unique.resize(out_num_runs.front());
  out_counts.resize(out_num_runs.front());

  c2h::host_vector<type> reference_out = in;
  reference_out.erase(std::unique(reference_out.begin(), reference_out.end()), reference_out.end());
  REQUIRE(out_unique == reference_out);
}

#if 0 // https://github.com/NVIDIA/cccl/issues/400
template<class T>
struct convertible_from_T {
  T val_;

  convertible_from_T() = default;
  __host__ __device__ convertible_from_T(const T& val) noexcept : val_(val) {}
  __host__ __device__ convertible_from_T& operator=(const T& val) noexcept {
    val_ = val;
  }
  // Converting back to T helps satisfy all the machinery that T supports
  __host__ __device__ operator T() const noexcept { return val_; }
};

C2H_TEST("DeviceRunLengthEncode::Encode works with a different output type", "[device][run_length_encode]")
{
  using type = c2h::custom_type_t<c2h::equal_comparable_t>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<type> in(num_items);
  c2h::device_vector<convertible_from_T<type>> out_unique(num_items);
  c2h::device_vector<int>  out_counts(num_items);
  c2h::device_vector<int>  out_num_runs(num_items);
  c2h::gen(C2H_SEED(2), in);

  run_length_encode(in.begin(),
                    out_unique.begin(),
                    out_counts.begin(),
                    out_num_runs.begin(),
                    num_items);

  // trim output
  out_unique.resize(out_num_runs.front());

  c2h::host_vector<convertible_from_T<type>> reference_out = in;
  reference_out.erase(std::unique(reference_out.begin(), reference_out.end()), reference_out.end());
  REQUIRE(out_unique == reference_out);
}
#endif // https://github.com/NVIDIA/cccl/issues/400

C2H_TEST("DeviceRunLengthEncode::Encode can handle leading NaN", "[device][run_length_encode]")
{
  using type = double;

  constexpr int num_items = 10;
  c2h::device_vector<type> in(num_items);
  thrust::sequence(c2h::device_policy, in.begin(), in.end(), 0.0);
  c2h::device_vector<type> out_unique(num_items);
  c2h::device_vector<int> out_counts(num_items);
  c2h::device_vector<int> out_num_runs(1);

  c2h::device_vector<type> reference_unique = in;
  in.front()                                = cuda::std::numeric_limits<type>::quiet_NaN();

  run_length_encode(in.begin(), out_unique.begin(), out_counts.begin(), out_num_runs.begin(), num_items);

  c2h::device_vector<int> reference_counts(num_items, 1); // [1, 1, ..., 1]
  c2h::device_vector<int> reference_num_runs(1, num_items); // [num_items]

  // turn the NaN into something else to make it comparable
  out_unique.front()       = 42.0;
  reference_unique.front() = 42.0;

  REQUIRE(out_unique == reference_unique);
  REQUIRE(out_counts == reference_counts);
  REQUIRE(out_num_runs == reference_num_runs);
}

C2H_TEST("DeviceRunLengthEncode::Encode works with non-default constructible iterators",
         "[device][run_length_encode]",
         offset_types)
{
  // This is a smoke test to ensure that the algorithm works with iterators that are not default-constructible, as was
  // the case before introducing the streaming context (see https://github.com/NVIDIA/cccl/issues/6419).
  using type        = int;
  using offset_type = typename c2h::get<0, TestType>;

  constexpr int64_t num_items = 1000;
  auto counting_it            = thrust::make_counting_iterator(0);
  auto custom_it              = thrust::make_transform_iterator(counting_it, non_default_constructible_iterator_op{42});

  c2h::device_vector<type> out_unique(num_items);
  c2h::device_vector<int> out_counts(num_items);
  c2h::device_vector<int> out_num_runs(1);

  run_length_encode(custom_it, out_unique.begin(), out_counts.begin(), out_num_runs.begin(), num_items);

  c2h::device_vector<type> reference_unique{custom_it, custom_it + num_items};
  c2h::device_vector<int> reference_counts(num_items, 1); // [1, 1, ..., 1]
  c2h::device_vector<int> reference_num_runs(1, num_items); // [num_items]

  REQUIRE(out_unique == reference_unique);
  REQUIRE(out_counts == reference_counts);
  REQUIRE(out_num_runs == reference_num_runs);
}

C2H_TEST("DeviceRunLengthEncode::Encode works for a large number of items",
         "[device][run_length_encode][skip-cs-initcheck][skip-cs-racecheck][skip-cs-synccheck]",
         offset_types)
try
{
  using offset_type     = typename c2h::get<0, TestType>;
  using run_length_type = offset_type;

  constexpr std::size_t uint32_max = cuda::std::numeric_limits<std::uint32_t>::max();

  std::size_t random_range = GENERATE_COPY(take(1, random((1 << 20), (1 << 22))));
  random_range += (random_range % 2);
  const std::size_t num_items =
    (sizeof(offset_type) == 8) ? uint32_max + random_range : cuda::std::numeric_limits<offset_type>::max();

  auto counting_it = cuda::counting_iterator(offset_type{0});

  // We repeat each number once for the first <num_small_runs> number of items and all subsequent numbers twice
  const std::size_t num_small_runs = cuda::std::min(uint32_max, num_items) - 4;
  const auto num_uniques           = static_cast<offset_type>(num_small_runs + (num_items - num_small_runs + 1) / 2);
  auto input_item_it =
    cuda::transform_iterator(counting_it, repeat_item_gen_op<offset_type>{static_cast<offset_type>(num_small_runs)});

  // Prepare helper to check the unique items being written: we expect the i-th item corresponding to value i
  auto check_unique_out_helper = detail::large_problem_test_helper(num_uniques);
  auto check_unique_out_it     = check_unique_out_helper.get_flagging_output_iterator(counting_it);

  // Prepare helper to check the run-lengths being written: i-th item corresponding to value i
  // We repeat each number once for the first num_small_runs number of items and all subsequent numbers twice
  auto check_run_length_out_helper = detail::large_problem_test_helper(num_uniques);
  auto expected_run_lengths_it     = cuda::transform_iterator(
    counting_it, run_to_run_length_op<offset_type, run_length_type>{static_cast<offset_type>(num_small_runs)});
  auto check_run_length_out_it = check_run_length_out_helper.get_flagging_output_iterator(expected_run_lengths_it);

  // This is a requirement on the test to simplify test logic bit: if the last run is truncated and, hence, is not a
  // "full" _long_ run of length 2, we'd otherwise need to account for the run-length of the very last run
  REQUIRE((num_items - num_small_runs) % 2 == 0);

  // Allocate memory for the number of expected unique items
  c2h::device_vector<offset_type> out_num_runs(1);

  // Run algorithm under test
  run_length_encode(
    input_item_it,
    check_unique_out_it,
    check_run_length_out_it,
    out_num_runs.begin(),
    static_cast<offset_type>(num_items));

  // Verify result
  REQUIRE(out_num_runs[0] == num_uniques);
  check_unique_out_helper.check_all_results_correct();
  check_run_length_out_helper.check_all_results_correct();
}
catch (std::bad_alloc& e)
{
  std::cerr << "Caught bad_alloc: " << e.what() << '\n';
}

C2H_TEST("DeviceRunLengthEncode::Encode works for large runs of equal items",
         "[device][run_length_encode][skip-cs-initcheck][skip-cs-racecheck][skip-cs-synccheck]",
         offset_types)
try
{
  using offset_type     = typename c2h::get<0, TestType>;
  using run_length_type = offset_type;
  using item_t          = char;

  CAPTURE(c2h::type_name<offset_type>(), c2h::type_name<run_length_type>(), c2h::type_name<item_t>());

  constexpr std::size_t uint32_max = cuda::std::numeric_limits<std::uint32_t>::max();

  constexpr std::size_t num_items =
    (sizeof(offset_type) == 8) ? uint32_max + (1 << 22) : cuda::std::numeric_limits<offset_type>::max();

  constexpr auto num_uniques                = offset_type{2};
  constexpr run_length_type first_run_size  = 200;
  constexpr run_length_type second_run_size = num_items - first_run_size;

  // First run is a small run of equal items
  auto small_segment_it = cuda::constant_iterator(item_t{3});
  // Second run is a very large run of equal items
  auto large_segment_it = cuda::constant_iterator(item_t{42});
  auto input_item_it    = detail::make_concat_iterators_op(small_segment_it, large_segment_it, first_run_size);

  // Allocate some memory for the results
  c2h::device_vector<item_t> uniques_out(num_uniques);
  c2h::device_vector<run_length_type> run_lengths_out(num_uniques);

  // Allocate memory for the number of expected unique items
  c2h::device_vector<offset_type> out_num_runs(1);

  // Run algorithm under test
  run_length_encode(
    input_item_it,
    uniques_out.begin(),
    run_lengths_out.begin(),
    out_num_runs.begin(),
    static_cast<offset_type>(num_items));

  // Expected results
  c2h::device_vector<item_t> expected_uniques{item_t{3}, item_t{42}};
  c2h::device_vector<run_length_type> expected_run_lengths{first_run_size, second_run_size};

  // Verify result
  REQUIRE(out_num_runs[0] == num_uniques);
  REQUIRE(expected_uniques == uniques_out);
  REQUIRE(expected_run_lengths == run_lengths_out);
}
catch (std::bad_alloc& e)
{
  std::cerr << "Caught bad_alloc: " << e.what() << '\n';
}

C2H_TEST("DeviceRunLengthEncode::Encode is exact over a segment-length grid",
         "[device][run_length_encode]",
         segment_key_types,
         offset_types)
{
  using key_t    = typename c2h::get<0, TestType>;
  using offset_t = typename c2h::get<1, TestType>;

  constexpr long long tile      = segment_grid_tile_size(sizeof(key_t));
  constexpr long long warp_tile = tile / 8;

  struct segment_grid_case
  {
    long long num_items;
    int max_seg;
  };
  const segment_grid_case cases[] = {
    {200000, 2}, // mid-size, dense
    {150000, 3}, // mid-size, different tile alignment
    {tile, 1}, // single tile, all runs of length 1
    {tile, 1000000}, // single tile, one run
    {3 * tile + 7, 1}, // partial tail tile, dense
    {3 * tile + 1, 1000000}, // run crossing into a one-element tail tile
    {(1 << 20) + 12345, 2}, // partial tail, mid density
    {64 * tile + 7, 7}, // run count per warp-tile straddles the warp-tile capacity boundary
    {64 * tile + 7, 100}, // a couple of runs per warp of input
    {64 * tile, -40}, // fixed-length runs: several run heads per 32-element window
    {64 * tile, -31}, // fixed-length runs: run count per warp-tile just above one full warp
    {64 * tile, -4}, // fixed-length runs: run count per warp-tile exactly at a power of two
    {64 * tile, -3}, // fixed-length runs: run count per warp-tile just above a power of two
    {64 * tile, -static_cast<int>(warp_tile + 1)}, // run head drifts through every in-warp-tile offset
    {64 * tile, 0}, // one constant run over the whole input
    {64 * tile, -static_cast<int>(tile)}, // run length == tile: a run head at element 0 of every tile
  };

  for (const segment_grid_case& grid_case : cases)
  {
    // fixed run lengths (max_seg <= 0) only vary in key values; one seed suffices
    const int num_seeds = (grid_case.max_seg > 0) ? 2 : 1;
    for (int seed = 0; seed < num_seeds; ++seed)
    {
      test_segmented_encode<key_t, offset_t>(grid_case.num_items, grid_case.max_seg, 0, seed == 0 ? 1u : 42u);
    }
  }
}

C2H_TEST("DeviceRunLengthEncode::Encode is exact at pipeline-scale sizes",
         "[device][run_length_encode]",
         c2h::type_list<std::uint32_t, std::int8_t>)
{
  using key_t = typename c2h::get<0, TestType>;

  constexpr long long tile = segment_grid_tile_size(sizeof(key_t));

  test_segmented_encode<key_t, int>(1030 * tile + 7, 1, 1, 1u); // misaligned input, dense, partial tail
  test_segmented_encode<key_t, int>(1024 * tile, -static_cast<int>(tile), 0, 1u); // a head at every tile start
  // one-element run whose head is the only element of a one-element final tile
  test_segmented_encode<key_t, int>(1024 * tile + 1, -static_cast<int>(1024 * tile), 0, 1u);
  test_segmented_encode<key_t, int>(1024 * tile, 0, 0, 1u); // one constant run over the whole input
  // run length == tile + 1: the run head position drifts through every in-tile offset
  test_segmented_encode<key_t, int>(1200 * tile + 3, -static_cast<int>(tile + 1), 0, 1u);
}

C2H_TEST("DeviceRunLengthEncode::Encode handles every input misalignment",
         "[device][run_length_encode]",
         segment_key_types)
{
  using key_t = typename c2h::get<0, TestType>;

  constexpr long long tile    = segment_grid_tile_size(sizeof(key_t));
  constexpr int offsets_swept = static_cast<int>(32 / sizeof(key_t));

  // sweeps every sub-16B misalignment of the input pointer plus the two 16B-aligned shifted bases
  for (int elem_offset = 1; elem_offset <= offsets_swept; ++elem_offset)
  {
    test_segmented_encode<key_t, int>(3 * tile + 7, 32, elem_offset, 1u);
    if ((elem_offset * sizeof(key_t)) % 16 == 0)
    {
      // aligned shifted base with a run boundary at every tile boundary
      test_segmented_encode<key_t, int>(3 * tile + 7, -static_cast<int>(tile), elem_offset, 1u);
    }
  }
}
