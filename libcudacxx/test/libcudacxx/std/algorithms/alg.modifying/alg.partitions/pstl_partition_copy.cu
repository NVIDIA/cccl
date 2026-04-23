//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template <class Policy, class InputIterator, class OutputIterator1, class OutputIterator2, class UnaryPredicate>
// void partition_copy_copy(const Policy&   policy,
//                     InputIterator   first,
//                     InputIterator   last,
//                     OutputIterator1 result_true,
//                     OutputIterator2 result_false,
//                     UnaryPredicate  pred);

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include <cuda/cmath>
#include <cuda/iterator>
#include <cuda/memory_pool>
#include <cuda/std/execution>
#include <cuda/std/utility>
#include <cuda/stream>

#include <testing.cuh>
#include <utility.cuh>

#include "test_iterators.h"
#include "test_macros.h"
#include "test_pstl.h"

inline constexpr int size = 1000;

template <class T>
struct is_even
{
  [[nodiscard]] TEST_DEVICE_FUNC constexpr bool operator()(T value) const noexcept
  {
    return value % 2 == 0;
  }
};

template <class Policy, class T>
void test_partition_copy(const Policy& policy,
                         const c2h::device_vector<T>& input,
                         c2h::device_vector<T>& output_true,
                         c2h::device_vector<T>& output_false)
{
  { // Empty does not access anything
    auto res = cuda::std::partition_copy(
      policy,
      static_cast<T*>(nullptr),
      static_cast<T*>(nullptr),
      static_cast<T*>(nullptr),
      static_cast<T*>(nullptr),
      is_even<T>{});
    CHECK(res.first == nullptr);
    CHECK(res.second == nullptr);
  }

  auto expected_true  = cuda::transform_iterator{cuda::strided_iterator{cuda::counting_iterator{0}, 2}, cast_to<T>{}};
  auto expected_false = cuda::transform_iterator{cuda::strided_iterator{cuda::counting_iterator{1}, 2}, cast_to<T>{}};

  cuda::std::fill(policy, output_true.begin(), output_true.end(), static_cast<T>(-1));
  cuda::std::fill(policy, output_false.begin(), output_false.end(), static_cast<T>(-1));
  { // contiguous iterators
    auto res = cuda::std::partition_copy(
      policy, input.begin(), input.end(), output_true.begin(), output_false.begin(), is_even<T>{});
    CHECK(res == cuda::std::pair{output_true.end(), output_false.end()});
    CHECK(cuda::std::equal(policy, output_true.begin(), output_true.end(), expected_true));
    CHECK(cuda::std::equal(policy, output_false.begin(), output_false.end(), expected_false));
  }

  cuda::std::fill(policy, output_true.begin(), output_true.end(), static_cast<T>(-1));
  cuda::std::fill(policy, output_false.begin(), output_false.end(), static_cast<T>(-1));
  const T* raw_in = thrust::raw_pointer_cast(input.data());
  { // random access input
    auto res = cuda::std::partition_copy(
      policy,
      random_access_iterator{raw_in},
      random_access_iterator{raw_in + size},
      output_true.begin(),
      output_false.begin(),
      is_even<T>{});
    CHECK(res == cuda::std::pair{output_true.end(), output_false.end()});
    CHECK(cuda::std::equal(policy, output_true.begin(), output_true.end(), expected_true));
    CHECK(cuda::std::equal(policy, output_false.begin(), output_false.end(), expected_false));
  }

  cuda::std::fill(policy, output_true.begin(), output_true.end(), static_cast<T>(-1));
  cuda::std::fill(policy, output_false.begin(), output_false.end(), static_cast<T>(-1));
  T* raw_true  = thrust::raw_pointer_cast(output_true.data());
  T* raw_false = thrust::raw_pointer_cast(output_false.data());
  { // random access outputs
    auto res = cuda::std::partition_copy(
      policy,
      input.begin(),
      input.end(),
      random_access_iterator{raw_true},
      random_access_iterator{raw_false},
      is_even<T>{});
    CHECK(
      res
      == cuda::std::pair{random_access_iterator{raw_true + size / 2}, random_access_iterator{raw_false + size / 2}});
    CHECK(cuda::std::equal(policy, output_true.begin(), output_true.end(), expected_true));
    CHECK(cuda::std::equal(policy, output_false.begin(), output_false.end(), expected_false));
  }

  cuda::std::fill(policy, output_true.begin(), output_true.end(), static_cast<T>(-1));
  cuda::std::fill(policy, output_false.begin(), output_false.end(), static_cast<T>(-1));
  { // counting_iterator input
    auto res = cuda::std::partition_copy(
      policy,
      cuda::counting_iterator<T>{static_cast<T>(0)},
      cuda::counting_iterator<T>{static_cast<T>(size)},
      output_true.begin(),
      output_false.begin(),
      is_even<T>{});
    CHECK(res == cuda::std::pair{output_true.end(), output_false.end()});
    CHECK(cuda::std::equal(policy, output_true.begin(), output_true.end(), expected_true));
    CHECK(cuda::std::equal(policy, output_false.begin(), output_false.end(), expected_false));
  }

  cuda::std::fill(policy, output_true.begin(), output_true.end(), static_cast<T>(-1));
  cuda::std::fill(policy, output_false.begin(), output_false.end(), static_cast<T>(-1));
  { // converting input iterators
    auto res = cuda::std::partition_copy(
      policy,
      cuda::counting_iterator<short>{0},
      cuda::counting_iterator<short>{static_cast<short>(size)},
      output_true.begin(),
      output_false.begin(),
      is_even<short>{});
    CHECK(res == cuda::std::pair{output_true.end(), output_false.end()});
    CHECK(cuda::std::equal(policy, output_true.begin(), output_true.end(), expected_true));
    CHECK(cuda::std::equal(policy, output_false.begin(), output_false.end(), expected_false));
  }

  cuda::std::fill(policy, output_true.begin(), output_true.end(), static_cast<T>(-1));
  cuda::std::fill(policy, output_false.begin(), output_false.end(), static_cast<T>(-1));
  { // contiguous input, converting predicate
    auto res = cuda::std::partition_copy(
      policy, input.begin(), input.end(), output_true.begin(), output_false.begin(), is_even<long>{});
    CHECK(res == cuda::std::pair{output_true.end(), output_false.end()});
    CHECK(cuda::std::equal(policy, output_true.begin(), output_true.end(), expected_true));
    CHECK(cuda::std::equal(policy, output_false.begin(), output_false.end(), expected_false));
  }

  cuda::std::fill(policy, output_true.begin(), output_true.end(), static_cast<T>(-1));
  cuda::std::fill(policy, output_false.begin(), output_false.end(), static_cast<T>(-1));
  { // short iterators, converting predicate
    auto res = cuda::std::partition_copy(
      policy,
      cuda::counting_iterator<short>{0},
      cuda::counting_iterator<short>{static_cast<short>(size)},
      output_true.begin(),
      output_false.begin(),
      is_even<long>{});
    CHECK(res == cuda::std::pair{output_true.end(), output_false.end()});
    CHECK(cuda::std::equal(policy, output_true.begin(), output_true.end(), expected_true));
    CHECK(cuda::std::equal(policy, output_false.begin(), output_false.end(), expected_false));
  }
}

C2H_TEST("cuda::std::partition_copy", "[parallel algorithm]", integral_types)
{
  using T = typename c2h::get<0, TestType>;
  c2h::device_vector<T> input(size, thrust::no_init);
  c2h::device_vector<T> output_true(size / 2, thrust::no_init);
  c2h::device_vector<T> output_false(size / 2, thrust::no_init);
  thrust::sequence(input.begin(), input.end(), static_cast<T>(0));

  SECTION("with default stream")
  {
    const auto policy = cuda::execution::gpu;

    test_partition_copy(policy, input, output_true, output_false);
  }

  SECTION("with provided stream")
  {
    cuda::stream stream{cuda::device_ref{0}};
    const auto policy = cuda::execution::gpu.with(cuda::get_stream, stream);

    test_partition_copy(policy, input, output_true, output_false);
  }

  SECTION("with provided memory_resource")
  {
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(cuda::device_ref{0});
    const auto policy = cuda::execution::gpu.with(cuda::mr::get_memory_resource, device_resource);

    test_partition_copy(policy, input, output_true, output_false);
  }

  SECTION("with provided stream and memory_resource")
  {
    cuda::stream stream{cuda::device_ref{0}};
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(stream.device());
    const auto policy =
      cuda::execution::gpu.with(cuda::mr::get_memory_resource, device_resource).with(cuda::get_stream, stream);

    test_partition_copy(policy, input, output_true, output_false);
  }
}
