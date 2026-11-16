//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template <class Policy, class InputIterator, class UnaryPredicate>
// void partition(const Policy&  policy,
//                InputIterator  first,
//                InputIterator  last,
//                UnaryPredicate pred);

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include <cuda/cmath>
#include <cuda/execution>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/memory_pool>
#include <cuda/std/algorithm>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/execution>
#include <cuda/std/limits>
#include <cuda/stream>

#include <testing.cuh>
#include <utility.cuh>

#include "test_iterators.h"
#include "test_macros.h"
#include "test_pstl.h"
#include <c2h/catch2_allocation_failure.h>

inline constexpr int size = 1000;

template <class Policy, class T>
void test_partition(const Policy& policy, c2h::device_vector<T>& input)
{
  T* raw         = thrust::raw_pointer_cast(input.data());
  const auto mid = size / 2;

  { // Empty does not access anything
    auto res = cuda::std::partition(policy, static_cast<T*>(nullptr), static_cast<T*>(nullptr), cuda::__is_even{});
    CHECK(res == nullptr);
  }

  thrust::sequence(input.begin(), input.end(), static_cast<T>(0));
  { // contiguous
    auto res = cuda::std::partition(policy, input.begin(), input.end(), cuda::__is_even{});
    CHECK(res == cuda::std::next(input.begin(), mid));
    CHECK(cuda::std::equal(
      policy, input.begin(), res, cuda::strided_iterator{cuda::counting_iterator{static_cast<T>(0)}, 2}));

    CHECK(cuda::std::equal(
      policy, res, input.end(), cuda::strided_iterator{cuda::counting_iterator{static_cast<T>(size - 1)}, -2}));
  }

  thrust::sequence(input.begin(), input.end(), static_cast<T>(0));
  { // random access
    auto res =
      cuda::std::partition(policy, random_access_iterator{raw}, random_access_iterator{raw + size}, cuda::__is_even{});
    CHECK(res == random_access_iterator{raw + mid});
    CHECK(cuda::std::equal(policy,
                           input.begin(),
                           input.begin() + mid,
                           cuda::strided_iterator{cuda::counting_iterator{static_cast<T>(0)}, 2}));

    CHECK(cuda::std::equal(policy,
                           input.begin() + mid,
                           input.end(),
                           cuda::strided_iterator{cuda::counting_iterator{static_cast<T>(size - 1)}, -2}));
  }

  thrust::sequence(input.begin(), input.end(), static_cast<T>(0));
  { // converting predicate
    auto res = cuda::std::partition(policy, input.begin(), input.end(), cuda::__is_even{});
    CHECK(res == cuda::std::next(input.begin(), mid));
    CHECK(cuda::std::equal(
      policy, input.begin(), res, cuda::strided_iterator{cuda::counting_iterator{static_cast<T>(0)}, 2}));

    CHECK(cuda::std::equal(
      policy, res, input.end(), cuda::strided_iterator{cuda::counting_iterator{static_cast<T>(size - 1)}, -2}));
  }
}

C2H_TEST("cuda::std::partition", "[parallel algorithm]", integral_types)
{
  using T = typename c2h::get<0, TestType>;
  c2h::device_vector<T> input(size, thrust::no_init);

  SECTION("with default stream")
  {
    const auto policy = cuda::execution::gpu;

    test_partition(policy, input);
  }

  SECTION("with provided stream")
  {
    const cuda::stream stream{cuda::device_ref{0}};
    const auto policy = cuda::execution::gpu.with(cuda::get_stream, stream);

    test_partition(policy, input);
  }

  SECTION("with a maximum input size guarantee and provided stream")
  {
    const cuda::stream stream{cuda::device_ref{0}};
    const auto policy = cuda::execution::gpu.with(cuda::execution::guarantee(cuda::execution::max_num_items<size>{}))
                          .with(cuda::get_stream, stream);

    test_partition(policy, input);
  }

  SECTION("with provided memory_resource")
  {
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(cuda::device_ref{0});
    const auto policy = cuda::execution::gpu.with(cuda::mr::get_memory_resource, device_resource);

    test_partition(policy, input);
  }

  SECTION("with provided stream and memory_resource")
  {
    const cuda::stream stream{cuda::device_ref{0}};
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(stream.device());
    const auto policy =
      cuda::execution::gpu.with(cuda::mr::get_memory_resource, device_resource).with(cuda::get_stream, stream);

    test_partition(policy, input);
  }
}

using partition_boundary_types =
  c2h::type_list<cuda::std::int8_t,
                 cuda::std::uint8_t,
                 cuda::std::int16_t,
                 cuda::std::int32_t,
                 cuda::std::int64_t,
                 cuda::std::uint64_t,
                 double>;

template <class T, cuda::std::size_t Count>
void test_partition_count_boundary(int rejected)
{
  CAPTURE(Count, rejected, sizeof(T));

  c2h::device_vector<T> input(Count, T{1});
  const auto policy = cuda::execution::gpu;
  cuda::std::fill(policy, input.begin(), input.begin() + rejected, T{0});

  const auto bounded_policy = policy.with(cuda::execution::guarantee(cuda::execution::max_num_items<Count>{}));
  const auto result = cuda::std::partition(bounded_policy, input.begin(), input.end(), cuda::equal_to_value{T{1}});
  REQUIRE(result - input.begin() == static_cast<cuda::std::ptrdiff_t>(Count) - rejected);

  CHECK(cuda::std::all_of(policy, input.begin(), result, cuda::equal_to_value{T{1}}));
  CHECK(cuda::std::all_of(policy, result, input.end(), cuda::equal_to_value{T{0}}));
}

C2H_TEST("cuda::std::partition count boundary", "[parallel algorithm][large]", partition_boundary_types)
try
{
  using T                            = typename c2h::get<0, TestType>;
  constexpr cuda::std::size_t cutoff = cuda::std::numeric_limits<int>::max() / 2;
  const int rejected                 = GENERATE(0, 3000);

  SECTION("below the narrow offset cutoff")
  {
    test_partition_count_boundary<T, cutoff - 1>(rejected);
  }

  SECTION("at the narrow offset cutoff")
  {
    test_partition_count_boundary<T, cutoff>(rejected);
  }

  SECTION("above the narrow offset cutoff")
  {
    test_partition_count_boundary<T, cutoff + 1>(rejected);
  }
}
catch (...)
{
  c2h::skip_if_allocation_failure("Skipping partition count boundary: insufficient device memory");
}
