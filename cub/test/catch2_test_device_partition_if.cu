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

#include <cub/device/device_partition.cuh>

#include <thrust/distance.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/tabulate_output_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/partition.h>
#include <thrust/reverse.h>

#include <cuda/cmath>
#include <cuda/std/iterator>

#include <algorithm>

#include "catch2_large_problem_helper.cuh"
#include "catch2_test_device_select_common.cuh"
#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.h>

DECLARE_LAUNCH_WRAPPER(cub::DevicePartition::If, partition_if);

// %PARAM% TEST_LAUNCH lid 0:1:2

struct always_false_t
{
  template <typename T>
  __device__ bool operator()(const T&) const
  {
    return false;
  }
};

struct always_true_t
{
  template <typename T>
  __device__ bool operator()(const T&) const
  {
    return true;
  }
};

using all_types =
  c2h::type_list<std::uint8_t,
                 std::uint16_t,
                 std::uint32_t,
                 std::uint64_t,
                 ulonglong2,
// WAR bug in vec type handling in NVCC 12.0 + GCC 11.4 + C++20
#if !(_CCCL_CUDA_COMPILER(NVCC, ==, 12, 0) && _CCCL_COMPILER(GCC, ==, 11, 4) && _CCCL_STD_VER == 2020)
#  if _CCCL_CTK_AT_LEAST(13, 0)
                 ulonglong4_16a,
#  else // _CCCL_CTK_AT_LEAST(13, 0)
                 ulonglong4,
#  endif // _CCCL_CTK_AT_LEAST(13, 0)
#endif // !(NVCC 12.0 and GCC 11.4 and C++20)
                 int,
                 long2,
                 c2h::custom_type_t<c2h::less_comparable_t, c2h::equal_comparable_t>>;

using types =
  c2h::type_list<std::uint8_t,
                 std::uint32_t,
// WAR bug in vec type handling in NVCC 12.0 + GCC 11.4 + C++20
#if !(_CCCL_CUDA_COMPILER(NVCC, ==, 12, 0) && _CCCL_COMPILER(GCC, ==, 11, 4) && _CCCL_STD_VER == 2020)
#  if _CCCL_CTK_AT_LEAST(13, 0)
                 ulonglong4_16a,
#  else // _CCCL_CTK_AT_LEAST(13, 0)
                 ulonglong4,
#  endif // _CCCL_CTK_AT_LEAST(13, 0)
#endif // !(NVCC 12.0 and GCC 11.4 and C++20)
                 c2h::custom_type_t<c2h::less_comparable_t, c2h::equal_comparable_t>>;

// List of offset types to be used for testing large number of items
using offset_types = c2h::type_list<std::int32_t, std::uint32_t, std::uint64_t>;

C2H_TEST("DevicePartition::If can run with empty input", "[device][partition_if]", types)
{
  using type = typename c2h::get<0, TestType>;

  constexpr int num_items = 0;
  c2h::device_vector<type> in(num_items);
  c2h::device_vector<type> out(num_items);

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 42);
  int* d_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  partition_if(in.begin(), out.begin(), d_num_selected_out, num_items, always_true_t{});

  REQUIRE(num_selected_out[0] == 0);
}

C2H_TEST("DevicePartition::If handles all matched", "[device][partition_if]", types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<type> in(num_items);
  c2h::device_vector<type> out(num_items);
  c2h::gen(C2H_SEED(2), in);

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 0);
  int* d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  partition_if(in.begin(), out.begin(), d_first_num_selected_out, num_items, always_true_t{});

  REQUIRE(num_selected_out[0] == num_items);
  REQUIRE(out == in);
}

C2H_TEST("DevicePartition::If handles no matched", "[device][partition_if]", types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<type> in(num_items);
  c2h::device_vector<type> out(num_items);
  c2h::gen(C2H_SEED(2), in);

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 0);
  int* d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  partition_if(in.begin(), out.begin(), d_first_num_selected_out, num_items, always_false_t{});

  // The false partition is in reverse order
  thrust::reverse(c2h::device_policy, out.begin(), out.end());

  REQUIRE(num_selected_out[0] == 0);
  REQUIRE(out == in);
}

C2H_TEST("DevicePartition::If does not change input", "[device][partition_if]", types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<type> in(num_items);
  c2h::device_vector<type> out(num_items);
  c2h::gen(C2H_SEED(2), in);

  // just pick one of the input elements as boundary
  less_than_t<type> le{in[num_items / 2]};

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 0);
  int* d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  // copy input first
  c2h::device_vector<type> reference = in;

  partition_if(in.begin(), out.begin(), d_first_num_selected_out, num_items, le);

  REQUIRE(reference == in);
}

C2H_TEST("DevicePartition::If is stable", "[device][partition_if]")
{
  using type = c2h::custom_type_t<c2h::less_comparable_t, c2h::equal_comparable_t>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<type> in(num_items);
  c2h::device_vector<type> out(num_items);
  c2h::gen(C2H_SEED(2), in);

  // just pick one of the input elements as boundary
  less_than_t<type> le{in[num_items / 2]};

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 0);
  int* d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  // Ensure that we create the same output as std
  c2h::host_vector<type> reference = in;
  // The main difference between stable_partition and DevicePartition::If is that the false partition is in reverse
  // order
  const auto boundary = std::stable_partition(reference.begin(), reference.end(), le);
  std::reverse(boundary, reference.end());

  partition_if(in.begin(), out.begin(), d_first_num_selected_out, num_items, le);

  REQUIRE(num_selected_out[0] == cuda::std::distance(reference.begin(), boundary));
  REQUIRE(reference == out);
}

C2H_TEST("DevicePartition::If works with iterators", "[device][partition_if]", all_types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<type> in(num_items);
  c2h::device_vector<type> out(num_items);
  c2h::gen(C2H_SEED(2), in);

  // just pick one of the input elements as boundary
  less_than_t<type> le{in[num_items / 2]};

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 0);
  int* d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  // Ensure that we create the same output as std
  c2h::host_vector<type> reference = in;
  // The main difference between stable_partition and DevicePartition::If is that the false partition is in reverse
  // order
  const auto boundary = std::stable_partition(reference.begin(), reference.end(), le);
  std::reverse(boundary, reference.end());

  partition_if(in.begin(), out.begin(), d_first_num_selected_out, num_items, le);

  REQUIRE(num_selected_out[0] == cuda::std::distance(reference.begin(), boundary));
  REQUIRE(reference == out);
}

C2H_TEST("DevicePartition::If works with pointers", "[device][partition_if]", types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<type> in(num_items);
  c2h::device_vector<type> out(num_items);
  c2h::gen(C2H_SEED(2), in);

  // just pick one of the input elements as boundary
  less_than_t<type> le{in[num_items / 2]};

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 0);
  int* d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  // Ensure that we create the same output as std
  c2h::host_vector<type> reference = in;
  // The main difference between stable_partition and DevicePartition::If is that the false partition is in reverse
  // order
  const auto boundary = std::stable_partition(reference.begin(), reference.end(), le);
  std::reverse(boundary, reference.end());

  partition_if(
    thrust::raw_pointer_cast(in.data()), thrust::raw_pointer_cast(out.data()), d_first_num_selected_out, num_items, le);

  REQUIRE(num_selected_out[0] == cuda::std::distance(reference.begin(), boundary));
  REQUIRE(reference == out);
}

template <class T>
struct convertible_from_T
{
  T val_;

  convertible_from_T() = default;
  __host__ __device__ convertible_from_T(const T& val) noexcept
      : val_(val)
  {}
  __host__ __device__ convertible_from_T& operator=(const T& val) noexcept
  {
    val_ = val;
  }
  // Converting back to T helps satisfy all the machinery that T supports
  __host__ __device__ operator T() const noexcept
  {
    return val_;
  }
};

C2H_TEST("DevicePartition::If works with a different output type", "[device][partition_if]")
{
  using type = c2h::custom_type_t<c2h::less_comparable_t, c2h::equal_comparable_t>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<type> in(num_items);
  c2h::device_vector<convertible_from_T<type>> out(num_items);
  c2h::gen(C2H_SEED(2), in);

  // just pick one of the input elements as boundary
  less_than_t<type> le{in[num_items / 2]};

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 0);
  int* d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  // Ensure that we create the same output as std
  c2h::host_vector<type> reference = in;
  // The main difference between stable_partition and DevicePartition::If is that the false partition is in reverse
  // order
  const auto boundary = std::stable_partition(reference.begin(), reference.end(), le);
  std::reverse(boundary, reference.end());

  partition_if(in.begin(), out.begin(), d_first_num_selected_out, num_items, le);

  REQUIRE(num_selected_out[0] == cuda::std::distance(reference.begin(), boundary));
  REQUIRE(reference == out);
}

C2H_TEST("DevicePartition::If works for very large number of items",
         "[device][partition_if][skip-cs-initcheck][skip-cs-racecheck][skip-cs-synccheck]",
         offset_types)
try
{
  using type     = std::int64_t;
  using offset_t = typename c2h::get<0, TestType>;

  const offset_t num_items_max = detail::make_large_offset<offset_t>();
  const offset_t num_items_min = num_items_max > 10000 ? num_items_max - 10000ULL : offset_t{0};
  const offset_t num_items     = GENERATE_COPY(
    values(
      {num_items_max, static_cast<offset_t>(num_items_max - 1), static_cast<offset_t>(1), static_cast<offset_t>(3)}),
    take(2, random(num_items_min, num_items_max)));

  auto in = thrust::make_counting_iterator(offset_t{0});

  // We select the first <cut_off_index> items and reject the rest
  const offset_t cut_off_index = num_items / 4;

  // Prepare expected data
  auto expected_selected_it = thrust::make_counting_iterator(offset_t{0});
  auto expected_rejected_it = cuda::std::make_reverse_iterator(
    thrust::make_counting_iterator(offset_t{cut_off_index}) + (num_items - cut_off_index));
  auto expected_result_op =
    make_index_to_expected_partition_op(expected_selected_it, expected_rejected_it, cut_off_index);
  auto expected_result_it =
    thrust::make_transform_iterator(thrust::make_counting_iterator(offset_t{0}), expected_result_op);

  // Prepare helper to check results
  auto check_result_helper = detail::large_problem_test_helper(num_items);
  auto check_result_it     = check_result_helper.get_flagging_output_iterator(expected_result_it);

  // Needs to be device accessible
  c2h::device_vector<offset_t> num_selected_out(1, 0);
  offset_t* d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  // Run test
  partition_if(
    in, check_result_it, d_first_num_selected_out, num_items, less_than_t<type>{static_cast<type>(cut_off_index)});

  // Ensure that we created the correct output
  REQUIRE(num_selected_out[0] == cut_off_index);
  check_result_helper.check_all_results_correct();
}
catch (std::bad_alloc&)
{
  // Exceeding memory is not a failure.
}
