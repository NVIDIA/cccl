/******************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_run_length_encode.cuh>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/sequence.h>

#include <algorithm>
#include <limits>
#include <numeric>

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
    thrust::make_counting_iterator(type{}), out_unique.begin(), out_counts.begin(), out_num_runs.begin(), num_items);

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

  auto counting_it = thrust::make_counting_iterator(offset_type{0});

  // We repeat each number once for the first <num_small_runs> number of items and all subsequent numbers twice
  const std::size_t num_small_runs = cuda::std::min(uint32_max, num_items) - 4;
  const auto num_uniques           = static_cast<offset_type>(num_small_runs + (num_items - num_small_runs + 1) / 2);
  auto input_item_it               = thrust::make_transform_iterator(
    counting_it, repeat_item_gen_op<offset_type>{static_cast<offset_type>(num_small_runs)});

  // Prepare helper to check the unique items being written: we expect the i-th item corresponding to value i
  auto check_unique_out_helper = detail::large_problem_test_helper(num_uniques);
  auto check_unique_out_it     = check_unique_out_helper.get_flagging_output_iterator(counting_it);

  // Prepare helper to check the run-lengths being written: i-th item corresponding to value i
  // We repeat each number once for the first num_small_runs number of items and all subsequent numbers twice
  auto check_run_length_out_helper = detail::large_problem_test_helper(num_uniques);
  auto expected_run_lengths_it     = thrust::make_transform_iterator(
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
  std::cerr << "Caught bad_alloc: " << e.what() << std::endl;
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
  auto small_segment_it = thrust::make_constant_iterator(item_t{3});
  // Second run is a very large run of equal items
  auto large_segment_it = thrust::make_constant_iterator(item_t{42});
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
  std::cerr << "Caught bad_alloc: " << e.what() << std::endl;
}
