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

inline constexpr int size = 1000;

template <class T>
struct is_even
{
  [[nodiscard]] __device__ constexpr bool operator()(T value) const noexcept
  {
    return value % 2 == 0;
  }
};

template <class Policy>
void test_partition_copy(const Policy& policy,
                         const thrust::device_vector<int>& input,
                         thrust::device_vector<int>& output_true,
                         thrust::device_vector<int>& output_false)
{
  { // Empty does not access anything
    auto res = cuda::std::partition_copy(
      policy,
      static_cast<int*>(nullptr),
      static_cast<int*>(nullptr),
      static_cast<int*>(nullptr),
      static_cast<int*>(nullptr),
      is_even<int>{});
    CHECK(res.first == nullptr);
    CHECK(res.second == nullptr);
  }

  cuda::std::fill(policy, output_true.begin(), output_true.end(), -1);
  cuda::std::fill(policy, output_false.begin(), output_false.end(), -1);
  { // With contiguous input
    auto res = cuda::std::partition_copy(
      policy, input.begin(), input.end(), output_true.begin(), output_false.begin(), is_even<int>{});
    CHECK(res == cuda::std::pair{output_true.end(), output_false.end()});
    CHECK(cuda::std::equal(
      policy, output_true.begin(), output_true.end(), cuda::strided_iterator{cuda::counting_iterator{0}, 2}));
    CHECK(cuda::std::equal(
      policy, output_false.begin(), output_false.end(), cuda::strided_iterator{cuda::counting_iterator{1}, 2}));
  }

  cuda::std::fill(policy, output_true.begin(), output_true.end(), -1);
  cuda::std::fill(policy, output_false.begin(), output_false.end(), -1);
  { // With random access input
    auto res = cuda::std::partition_copy(
      policy,
      cuda::counting_iterator<int>{0},
      cuda::counting_iterator<int>{size},
      output_true.begin(),
      output_false.begin(),
      is_even<int>{});
    CHECK(res == cuda::std::pair{output_true.end(), output_false.end()});
    CHECK(cuda::std::equal(
      policy, output_true.begin(), output_true.end(), cuda::strided_iterator{cuda::counting_iterator{0}, 2}));
    CHECK(cuda::std::equal(
      policy, output_false.begin(), output_false.end(), cuda::strided_iterator{cuda::counting_iterator{1}, 2}));
  }

  cuda::std::fill(policy, output_true.begin(), output_true.end(), -1);
  cuda::std::fill(policy, output_false.begin(), output_false.end(), -1);
  { // With different input type
    auto res = cuda::std::partition_copy(
      policy,
      cuda::counting_iterator<short>{0},
      cuda::counting_iterator<short>{size},
      output_true.begin(),
      output_false.begin(),
      is_even<short>{});
    CHECK(res == cuda::std::pair{output_true.end(), output_false.end()});
    CHECK(cuda::std::equal(
      policy, output_true.begin(), output_true.end(), cuda::strided_iterator{cuda::counting_iterator{0}, 2}));
    CHECK(cuda::std::equal(
      policy, output_false.begin(), output_false.end(), cuda::strided_iterator{cuda::counting_iterator{1}, 2}));
  }

  cuda::std::fill(policy, output_true.begin(), output_true.end(), -1);
  cuda::std::fill(policy, output_false.begin(), output_false.end(), -1);
  { // With contiguous input, converting predicate
    auto res = cuda::std::partition_copy(
      policy, input.begin(), input.end(), output_true.begin(), output_false.begin(), is_even<long>{});
    CHECK(res == cuda::std::pair{output_true.end(), output_false.end()});
    CHECK(cuda::std::equal(
      policy, output_true.begin(), output_true.end(), cuda::strided_iterator{cuda::counting_iterator{0}, 2}));
    CHECK(cuda::std::equal(
      policy, output_false.begin(), output_false.end(), cuda::strided_iterator{cuda::counting_iterator{1}, 2}));
  }

  cuda::std::fill(policy, output_true.begin(), output_true.end(), -1);
  cuda::std::fill(policy, output_false.begin(), output_false.end(), -1);
  { // With different input type, converting predicate
    auto res = cuda::std::partition_copy(
      policy,
      cuda::counting_iterator<short>{0},
      cuda::counting_iterator<short>{size},
      output_true.begin(),
      output_false.begin(),
      is_even<long>{});
    CHECK(res == cuda::std::pair{output_true.end(), output_false.end()});
    CHECK(cuda::std::equal(
      policy, output_true.begin(), output_true.end(), cuda::strided_iterator{cuda::counting_iterator{0}, 2}));
    CHECK(cuda::std::equal(
      policy, output_false.begin(), output_false.end(), cuda::strided_iterator{cuda::counting_iterator{1}, 2}));
  }
}

C2H_TEST("cuda::std::partition_copy", "[parallel algorithm]")
{
  thrust::device_vector<int> input(size, thrust::no_init);
  thrust::device_vector<int> output_true(size / 2, thrust::no_init);
  thrust::device_vector<int> output_false(size / 2, thrust::no_init);
  thrust::sequence(input.begin(), input.end(), 0);

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
