//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template <class ExecutionPolicy, class InputIterator, class OutputIterator, class BinaryPredicate>
// OutputIterator unique_copy(ExecutionPolicy&& policy,
//                            InputIterator first,
//                            InputIterator last,
//                            OutputIterator result,
//                            BinaryPredicate pred);

#include <thrust/device_vector.h>
#include <thrust/equal.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <cuda/iterator>
#include <cuda/memory_pool>
#include <cuda/std/__pstl_algorithm>
#include <cuda/std/execution>
#include <cuda/std/iterator>
#include <cuda/stream>

#include <testing.cuh>
#include <utility.cuh>

inline constexpr int size = 1000;

template <class Policy>
void test_unique_copy(const Policy& policy,
                      [[maybe_unused]] thrust::device_vector<int>& input,
                      [[maybe_unused]] thrust::device_vector<int>& output)
{
  { // empty should not access anything
    const auto res = cuda::std::unique_copy(
      policy,
      static_cast<int*>(nullptr),
      static_cast<int*>(nullptr),
      static_cast<int*>(nullptr),
      cuda::std::equal_to<>{});
    CHECK(res == nullptr);
  }

  cuda::std::fill(output.begin(), output.end(), -1);
  { // no duplicates
    const auto res =
      cuda::std::unique_copy(policy, input.begin(), input.end(), output.begin(), cuda::std::equal_to<>{});
    CHECK(input == output);
    CHECK(res == output.end());
    CHECK(cuda::std::equal(policy, output.begin(), output.end(), input.begin()));
  }

  cuda::std::fill(output.begin(), output.end(), -1);
  { // one match,
    input[43] = input[42];
    const auto res =
      cuda::std::unique_copy(policy, input.begin(), input.end(), output.begin(), cuda::std::equal_to<>{});
    CHECK(res == cuda::std::prev(output.end()));

    auto mid_in  = input.begin() + 43;
    auto mid_out = output.begin() + 42;
    CHECK(cuda::std::equal(policy, output.begin(), mid_out, input.begin()));
    CHECK(cuda::std::equal(policy, mid_in, input.end(), mid_out));
  }

  cuda::std::fill(output.begin(), output.end(), -1);
  { // all equal, not_equal to ensure we are actually using the predicate
    cuda::std::fill(input.begin(), input.end(), 42);
    const auto res =
      cuda::std::unique_copy(policy, input.begin(), input.end(), output.begin(), cuda::std::not_equal_to<>{});
    CHECK(res == output.end());
    CHECK(cuda::std::equal(policy, output.begin(), output.end(), cuda::constant_iterator{42}));
  }

  cuda::std::fill(output.begin(), output.end(), -1);
  { // all equal, continuous iterator
    const auto res =
      cuda::std::unique_copy(policy, input.begin(), input.end(), output.begin(), cuda::std::equal_to<>{});
    CHECK(res == cuda::std::next(output.begin()));
    CHECK(output[0] == 42);
  }
}

C2H_TEST("cuda::std::unique_copy", "[parallel algorithm]")
{
  thrust::device_vector<int> input(size, thrust::no_init);
  thrust::device_vector<int> output(size, thrust::no_init);
  thrust::sequence(input.begin(), input.end(), 0);

  SECTION("with default stream")
  {
    const auto policy = cuda::execution::__cub_par_unseq;
    test_unique_copy(policy, input, output);
  }

  SECTION("with provided stream")
  {
    cuda::stream stream{cuda::device_ref{0}};
    const auto policy = cuda::execution::__cub_par_unseq.with(cuda::get_stream, stream);
    test_unique_copy(policy, input, output);
  }

  SECTION("with provided memory_resource")
  {
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(cuda::device_ref{0});
    const auto policy = cuda::execution::__cub_par_unseq.with(cuda::mr::get_memory_resource, device_resource);
    test_unique_copy(policy, input, output);
  }

  SECTION("with provided stream and memory_resource")
  {
    cuda::stream stream{cuda::device_ref{0}};
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(stream.device());
    const auto policy = cuda::execution::__cub_par_unseq.with(cuda::mr::get_memory_resource, device_resource)
                          .with(cuda::get_stream, stream);
    test_unique_copy(policy, input, output);
  }
}
