// SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_select.cuh>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>

#include <cuda/cmath>

#include <algorithm>

#include "catch2_large_problem_helper.cuh"
#include "catch2_test_device_select_common.cuh"
#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.h>

template <class T>
inline T to_bound(const unsigned long long bound)
{
  return static_cast<T>(bound);
}

template <>
inline ulonglong2 to_bound(const unsigned long long bound)
{
  return {bound, bound};
}

_CCCL_SUPPRESS_DEPRECATED_PUSH
template <>
inline ulonglong4 to_bound(const unsigned long long bound)
{
  return {bound, bound, bound, bound};
}
_CCCL_SUPPRESS_DEPRECATED_POP

#if _CCCL_CTK_AT_LEAST(13, 0)
template <>
inline ulonglong4_16a to_bound(const unsigned long long bound)
{
  return {bound, bound, bound, bound};
}
#endif // _CCCL_CTK_AT_LEAST(13, 0)

template <>
inline long2 to_bound(const unsigned long long bound)
{
  return {static_cast<long>(bound), static_cast<long>(bound)};
}

template <>
inline c2h::custom_type_t<c2h::equal_comparable_t> to_bound(const unsigned long long bound)
{
  c2h::custom_type_t<c2h::equal_comparable_t> val;
  val.key = bound;
  val.val = bound;
  return val;
}

DECLARE_LAUNCH_WRAPPER(cub::DeviceSelect::Unique, select_unique);

// %PARAM% TEST_LAUNCH lid 0:1:2

struct equal_to_default_t
{
  template <typename T>
  __host__ __device__ bool operator()(const T& a) const
  {
    return a == T{};
  }
};

using all_types =
  c2h::type_list<std::uint8_t,
                 std::uint16_t,
                 std::uint32_t,
                 std::uint64_t,
                 ulonglong2,
#if _CCCL_CTK_AT_LEAST(13, 0)
                 ulonglong4_16a,
#else // _CCCL_CTK_AT_LEAST(13, 0)
                 ulonglong4,
#endif // _CCCL_CTK_AT_LEAST(13, 0)
                 int,
                 long2,
                 c2h::custom_type_t<c2h::equal_comparable_t>>;

using types = c2h::type_list<std::uint8_t, std::uint32_t>;

C2H_TEST("DeviceSelect::Unique can run with empty input", "[device][select_unique]", types)
{
  using type = typename c2h::get<0, TestType>;

  constexpr int num_items = 0;
  c2h::device_vector<type> in(num_items);
  c2h::device_vector<type> out(num_items);

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 42);
  int* d_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  select_unique(in.begin(), out.begin(), d_num_selected_out, num_items);

  REQUIRE(num_selected_out[0] == 0);
}

C2H_TEST("DeviceSelect::Unique handles none equal", "[device][select_unique]", types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 0);
  int* d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  select_unique(thrust::counting_iterator<type>(0), thrust::discard_iterator<>(), d_first_num_selected_out, num_items);

  REQUIRE(num_selected_out[0] == num_items);
}

C2H_TEST("DeviceSelect::Unique handles all equal", "[device][select_unique]", types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<type> in(num_items, static_cast<type>(1));
  c2h::device_vector<type> out(1);

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 0);
  int* d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  select_unique(in.begin(), out.begin(), d_first_num_selected_out, num_items);

  // At least one item is selected
  REQUIRE(num_selected_out[0] == 1);
  REQUIRE(out[0] == in[0]);
}

C2H_TEST("DeviceSelect::Unique does not change input", "[device][select_unique]", types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<type> in(num_items);
  c2h::device_vector<type> out(num_items);
  c2h::gen(C2H_SEED(2), in, to_bound<type>(0), to_bound<type>(42));

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 0);
  int* d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  // copy input first
  c2h::device_vector<type> reference = in;

  select_unique(in.begin(), out.begin(), d_first_num_selected_out, num_items);

  REQUIRE(reference == in);
}

C2H_TEST("DeviceSelect::Unique works with iterators", "[device][select_unique]", all_types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<type> in(num_items);
  c2h::device_vector<type> out(num_items);
  c2h::gen(C2H_SEED(2), in, to_bound<type>(0), to_bound<type>(42));

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 0);
  int* d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  select_unique(in.begin(), out.begin(), d_first_num_selected_out, num_items);

  // Ensure that we create the same output as std
  c2h::host_vector<type> reference = in;
  const auto boundary              = std::unique(reference.begin(), reference.end());
  REQUIRE((boundary - reference.begin()) == num_selected_out[0]);

  out.resize(num_selected_out[0]);
  reference.resize(num_selected_out[0]);
  REQUIRE(reference == out);
}

C2H_TEST("DeviceSelect::Unique works with pointers", "[device][select_unique]", types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<type> in(num_items);
  c2h::device_vector<type> out(num_items);
  c2h::gen(C2H_SEED(2), in, to_bound<type>(0), to_bound<type>(42));

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 0);
  int* d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  select_unique(
    thrust::raw_pointer_cast(in.data()), thrust::raw_pointer_cast(out.data()), d_first_num_selected_out, num_items);

  // Ensure that we create the same output as std
  c2h::host_vector<type> reference = in;
  const auto boundary              = std::unique(reference.begin(), reference.end());
  REQUIRE((boundary - reference.begin()) == num_selected_out[0]);

  out.resize(num_selected_out[0]);
  reference.resize(num_selected_out[0]);
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

C2H_TEST("DeviceSelect::Unique works with a different output type", "[device][select_unique]", types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<type> in(num_items);
  c2h::device_vector<convertible_from_T<type>> out(num_items);
  c2h::gen(C2H_SEED(2), in, to_bound<type>(0), to_bound<type>(42));

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 0);
  int* d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  select_unique(in.begin(), out.begin(), d_first_num_selected_out, num_items);

  // Ensure that we create the same output as std
  c2h::host_vector<type> reference = in;
  const auto boundary              = std::unique(reference.begin(), reference.end());
  REQUIRE((boundary - reference.begin()) == num_selected_out[0]);

  out.resize(num_selected_out[0]);
  reference.resize(num_selected_out[0]);
  REQUIRE(reference == out);
}

C2H_TEST("DeviceSelect::Unique works for very large number of items",
         "[device][select_unique][skip-cs-initcheck][skip-cs-racecheck][skip-cs-synccheck]")
try
{
  using type     = std::int64_t;
  using offset_t = std::int64_t;

  // The partition size (the maximum number of items processed by a single kernel invocation) is an important boundary
  constexpr auto max_partition_size = static_cast<offset_t>(cuda::std::numeric_limits<std::int32_t>::max());

  offset_t num_items = GENERATE_COPY(
    values({
      offset_t{2} * max_partition_size + offset_t{20000000}, // 3 partitions
      offset_t{2} * max_partition_size, // 2 partitions
      max_partition_size + offset_t{1}, // 2 partitions
      max_partition_size, // 1 partitions
      max_partition_size - offset_t{1} // 1 partitions
    }),
    take(2, random(max_partition_size - offset_t{1000000}, max_partition_size + offset_t{1000000})));

  // All unique
  SECTION("AllUnique")
  {
    auto in = thrust::make_counting_iterator(offset_t{0});

    // Prepare expected data
    auto expected_result_it = in;

    // Prepare helper to check results
    auto check_result_helper = detail::large_problem_test_helper(num_items);
    auto check_result_it     = check_result_helper.get_flagging_output_iterator(expected_result_it);

    // Needs to be device accessible
    c2h::device_vector<offset_t> num_selected_out(1, 0);
    offset_t* d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

    // Run test
    select_unique(in, check_result_it, d_first_num_selected_out, num_items);

    // Ensure that we created the correct output
    REQUIRE(num_selected_out[0] == num_items);
    check_result_helper.check_all_results_correct();
  }

  // All the same -> single unique
  SECTION("AllSame")
  {
    auto in = thrust::make_constant_iterator(offset_t{0});
    constexpr offset_t expected_num_unique{1};

    // Prepare expected data
    auto expected_result_it = in;

    // Prepare helper to check results
    auto check_result_helper = detail::large_problem_test_helper(expected_num_unique);
    auto check_result_it     = check_result_helper.get_flagging_output_iterator(expected_result_it);

    // Needs to be device accessible
    c2h::device_vector<offset_t> num_selected_out(1, 0);
    offset_t* d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

    // Run test
    select_unique(in, check_result_it, d_first_num_selected_out, num_items);

    // Ensure that we created the correct output
    REQUIRE(num_selected_out[0] == expected_num_unique);
    check_result_helper.check_all_results_correct();
  }
}
catch (std::bad_alloc&)
{
  // Exceeding memory is not a failure.
}
