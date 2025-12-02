// SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_select.cuh>
#include <cub/device/dispatch/dispatch_select_if.cuh>

#include <thrust/distance.h>
#include <thrust/functional.h>
#include <thrust/iterator/offset_iterator.h>
#include <thrust/logical.h>
#include <thrust/partition.h>
#include <thrust/reverse.h>

#include <cuda/iterator>
#include <cuda/std/limits>

#include <algorithm>

#include "catch2_test_device_select_common.cuh"
#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.h>

DECLARE_LAUNCH_WRAPPER(cub::DeviceSelect::If, select_if);

// %PARAM% TEST_LAUNCH lid 0:1:2

struct equal_to_default_t
{
  template <typename T>
  __host__ __device__ bool operator()(const T& a) const
  {
    return a == T{};
  }
};

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

C2H_TEST("DeviceSelect::If can run with empty input", "[device][select_if]", types)
{
  using type = typename c2h::get<0, TestType>;

  constexpr int num_items = 0;
  c2h::device_vector<type> in(num_items);
  c2h::device_vector<type> out(num_items);

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 42);
  int* d_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  select_if(in.begin(), out.begin(), d_num_selected_out, num_items, always_true_t{});

  REQUIRE(num_selected_out[0] == 0);
}

C2H_TEST("DeviceSelect::If handles all matched", "[device][select_if]", types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<type> in(num_items);
  c2h::device_vector<type> out(num_items);
  c2h::gen(C2H_SEED(2), in);

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 0);
  int* d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  select_if(in.begin(), out.begin(), d_first_num_selected_out, num_items, always_true_t{});

  REQUIRE(num_selected_out[0] == num_items);
  REQUIRE(out == in);
}

C2H_TEST("DeviceSelect::If handles no matched", "[device][select_if]", types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<type> in(num_items);
  c2h::device_vector<type> out(0);
  c2h::gen(C2H_SEED(2), in);

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 0);
  int* d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  select_if(in.begin(), out.begin(), d_first_num_selected_out, num_items, always_false_t{});

  REQUIRE(num_selected_out[0] == 0);
}

C2H_TEST("DeviceSelect::If does not change input", "[device][select_if]", types)
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

  select_if(in.begin(), out.begin(), d_first_num_selected_out, num_items, le);

  REQUIRE(reference == in);
}

C2H_TEST("DeviceSelect::If is stable", "[device][select_if]")
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

  select_if(in.begin(), out.begin(), d_first_num_selected_out, num_items, le);

  // Ensure that we create the same output as std
  c2h::host_vector<type> reference = in;
  std::stable_partition(reference.begin(), reference.end(), le);

  // Ensure that we did not overwrite other elements
  const auto boundary = out.begin() + num_selected_out[0];
  REQUIRE(thrust::all_of(c2h::device_policy, boundary, out.end(), equal_to_default_t{}));

  out.resize(num_selected_out[0]);
  reference.resize(num_selected_out[0]);
  REQUIRE(reference == out);
}

C2H_TEST("DeviceSelect::If works with iterators", "[device][select_if]", all_types)
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

  select_if(in.begin(), out.begin(), d_first_num_selected_out, num_items, le);

  const auto boundary = out.begin() + num_selected_out[0];
  REQUIRE(thrust::all_of(c2h::device_policy, out.begin(), boundary, le));
  REQUIRE(thrust::all_of(c2h::device_policy, boundary, out.end(), equal_to_default_t{}));
}

C2H_TEST("DeviceSelect::If works with pointers", "[device][select_if]", types)
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

  select_if(
    thrust::raw_pointer_cast(in.data()), thrust::raw_pointer_cast(out.data()), d_first_num_selected_out, num_items, le);

  const auto boundary = out.begin() + num_selected_out[0];
  REQUIRE(thrust::all_of(c2h::device_policy, out.begin(), boundary, le));
  REQUIRE(thrust::all_of(c2h::device_policy, boundary, out.end(), equal_to_default_t{}));
}

C2H_TEST("DeviceSelect::If works in place", "[device][select_if]", types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<type> in(num_items);
  c2h::gen(C2H_SEED(2), in);

  // just pick one of the input elements as boundary
  less_than_t<type> le{in[num_items / 2]};

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 0);
  int* d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  // Ensure that we create the same output as std
  c2h::host_vector<type> reference = in;
  std::stable_partition(reference.begin(), reference.end(), le);

  select_if(in.begin(), d_first_num_selected_out, num_items, le);

  in.resize(num_selected_out[0]);
  reference.resize(num_selected_out[0]);
  REQUIRE(reference == in);
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

C2H_TEST("DeviceSelect::If works with a different output type", "[device][select_if]")
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

  select_if(in.begin(), out.begin(), d_first_num_selected_out, num_items, le);

  const auto boundary = out.begin() + num_selected_out[0];
  REQUIRE(thrust::all_of(c2h::device_policy, out.begin(), boundary, le));
  REQUIRE(thrust::all_of(c2h::device_policy, boundary, out.end(), equal_to_default_t{}));
}

C2H_TEST("DeviceSelect::If works for very large number of items",
         "[device][select_if][skip-cs-initcheck][skip-cs-racecheck][skip-cs-synccheck]")
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

  // Input
  auto in = cuda::counting_iterator(static_cast<type>(0));

  // Needs to be device accessible
  c2h::device_vector<offset_t> num_selected_out(1, 0);
  offset_t* d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  // Run test
  constexpr offset_t match_every_nth = 1000000;
  offset_t expected_num_copied       = (num_items + match_every_nth - offset_t{1}) / match_every_nth;
  c2h::device_vector<type> out(expected_num_copied);
  select_if(
    in, out.begin(), d_first_num_selected_out, num_items, mod_n<offset_t>{static_cast<offset_t>(match_every_nth)});

  // Ensure that we created the correct output
  REQUIRE(num_selected_out[0] == expected_num_copied);
  auto expected_out_it     = cuda::transform_iterator(in, multiply_n<offset_t>{static_cast<offset_t>(match_every_nth)});
  bool all_results_correct = thrust::equal(out.cbegin(), out.cend(), expected_out_it);
  REQUIRE(all_results_correct == true);
}
catch (std::bad_alloc&)
{
  // Exceeding memory is not a failure.
}

C2H_TEST("DeviceSelect::If works for very large number of output items",
         "[device][select_if][skip-cs-initcheck][skip-cs-racecheck][skip-cs-synccheck]")
try
{
  using type     = std::uint8_t;
  using offset_t = std::int64_t;

  // The partition size (the maximum number of items processed by a single kernel invocation) is an important boundary
  constexpr auto max_partition_size = static_cast<offset_t>(cuda::std::numeric_limits<std::int32_t>::max());

  offset_t num_items = GENERATE_COPY(
    values({
      offset_t{2} * max_partition_size + offset_t{20000000}, // 3 partitions
      offset_t{2} * max_partition_size, // 2 partitions
      max_partition_size + offset_t{1}, // 2 partitions
      max_partition_size, // 1 or 2 partitions
      max_partition_size - offset_t{745}, // 1 or 2 partitions
      max_partition_size - offset_t{10745} // 1 partition
    }),
    take(2, random(max_partition_size - offset_t{100000}, max_partition_size + offset_t{100000})));

  CAPTURE(num_items);

  // Prepare input iterator: it[i] = (i%mod)+(i/div)
  static constexpr offset_t mod = 200;
  static constexpr offset_t div = 1000000000;
  auto in = cuda::transform_iterator(cuda::counting_iterator(offset_t{0}), modx_and_add_divy<offset_t, type>{mod, div});

  // Prepare output
  c2h::device_vector<type> out(num_items);

  // Needs to be device accessible
  c2h::device_vector<offset_t> num_selected_out(1, 0);
  offset_t* d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  // Run test
  select_if(in, out.begin(), d_first_num_selected_out, num_items, always_true_t{});

  // Ensure that we created the correct output
  REQUIRE(num_selected_out[0] == num_items);
  bool all_results_correct = thrust::equal(out.cbegin(), out.cend(), in);
  REQUIRE(all_results_correct == true);
}
catch (std::bad_alloc&)
{
  // Exceeding memory is not a failure.
}

C2H_TEST("DeviceSelect::If works with iterators", "[device][select_if]")
{
  using type = int;

  const int num_items = 10'000;
  c2h::device_vector<type> in(num_items);
  thrust::sequence(in.begin(), in.end());
  c2h::device_vector<type> out(num_items);
  using thrust::placeholders::_1;

  // select twice, appending the second selection to the first one without bringing the first selection's count to the
  // host
  c2h::device_vector<int> num_selected_out(2);
  select_if(in.begin(), out.begin(), num_selected_out.begin(), num_items, _1 < 1000); // [0;999]
  auto output_end = thrust::offset_iterator{out.begin(), num_selected_out.begin()};
  select_if(in.begin(), output_end, num_selected_out.begin() + 1, num_items, _1 >= 9000); // [9000;9999]

  c2h::device_vector<type> expected(2000);
  thrust::sequence(expected.begin(), expected.begin() + 1000);
  thrust::sequence(expected.begin() + 1000, expected.end(), 9000);

  out.resize(2000);
  REQUIRE(num_selected_out == c2h::device_vector<int>{1000, 1000});
  REQUIRE(out == expected);
}
