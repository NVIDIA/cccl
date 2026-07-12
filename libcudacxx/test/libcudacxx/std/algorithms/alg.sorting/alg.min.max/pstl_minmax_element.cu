//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template<class ExecutionPolicy, class ForwardIterator>
//   pair<ForwardIterator, ForwardIterator>
//   minmax_element(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last);

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include <cuda/iterator>
#include <cuda/memory_pool>
#include <cuda/std/algorithm>
#include <cuda/std/execution>
#include <cuda/std/functional>
#include <cuda/stream>

#include <testing.cuh>
#include <utility.cuh>

#include "test_iterators.h"
#include "test_macros.h"
#include "test_pstl.h"

inline constexpr int size = 1000;

template <class Policy, class T>
void test_minmax_element(const Policy& policy, c2h::device_vector<T>& input)
{
  { // empty range should not access anything and return {first, first}
    auto res = cuda::std::minmax_element(policy, static_cast<T*>(nullptr), static_cast<T*>(nullptr));
    static_assert(cuda::std::is_same_v<decltype(res), cuda::std::pair<T*, T*>>);
    CHECK(res.first == nullptr);
    CHECK(res.second == nullptr);
  }

  const T* raw_pointer = thrust::raw_pointer_cast(input.data());

  // Verify the parallel result against the serial cuda::std::minmax_element reference on the
  // same data. This is robust to ties (e.g. low-precision floating point types where several
  // adjacent values round to the same representation): the standard returns the *first* minimum
  // and the *last* maximum, and the parallel overload must match that positionally for any data.
  auto check_matches_serial = [&] {
    c2h::host_vector<T> host = input;
    const T* hp              = thrust::raw_pointer_cast(host.data());
    const auto expected      = cuda::std::minmax_element(hp, hp + size);
    const auto expected_min  = expected.first - hp;
    const auto expected_max  = expected.second - hp;

    { // contiguous iterator
      auto res = cuda::std::minmax_element(policy, input.begin(), input.end());
      CHECK((res.first - input.begin()) == expected_min);
      CHECK((res.second - input.begin()) == expected_max);
    }

    { // random access iterator
      auto res = cuda::std::minmax_element(
        policy, random_access_iterator{raw_pointer}, random_access_iterator{raw_pointer + size});
      CHECK((res.first - random_access_iterator{raw_pointer}) == expected_min);
      CHECK((res.second - random_access_iterator{raw_pointer}) == expected_max);
    }
  };

  thrust::sequence(input.begin(), input.end(), 1); // strictly ascending: min first, max last
  check_matches_serial();

  thrust::sequence(input.begin(), input.end(), size, -1); // strictly descending: min last, max first
  check_matches_serial();

  cuda::std::fill(policy, input.begin(), input.end(), T{42}); // all equal: first min, *last* max
  check_matches_serial();

  { // single element -> {first, first}
    c2h::device_vector<T> one(1, T{7});
    auto res = cuda::std::minmax_element(policy, one.begin(), one.end());
    CHECK(res.first == one.begin());
    CHECK(res.second == one.begin());
  }
}

C2H_TEST("cuda::std::minmax_element(Iter, Iter)", "[parallel algorithm]", all_types)
{
  using T = typename c2h::get<0, TestType>;
  c2h::device_vector<T> input(size, thrust::no_init);

  SECTION("with default stream")
  {
    const auto policy = cuda::execution::gpu;
    test_minmax_element(policy, input);
  }

  SECTION("with provided stream")
  {
    cuda::stream stream{cuda::device_ref{0}};
    const auto policy = cuda::execution::gpu.with(cuda::get_stream, stream);
    test_minmax_element(policy, input);
  }

  SECTION("with provided memory_resource")
  {
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(cuda::device_ref{0});
    const auto policy = cuda::execution::gpu.with(cuda::mr::get_memory_resource, device_resource);
    test_minmax_element(policy, input);
  }

  SECTION("with provided stream and memory_resource")
  {
    cuda::stream stream{cuda::device_ref{0}};
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(stream.device());
    const auto policy =
      cuda::execution::gpu.with(cuda::get_stream, stream).with(cuda::mr::get_memory_resource, device_resource);
    test_minmax_element(policy, input);
  }
}
