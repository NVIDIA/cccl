//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class Compare>
// bool lexicographical_compare(ExecutionPolicy&& exec,
//                              ForwardIterator1 first1, ForwardIterator1 last1,
//                              ForwardIterator2 first2, ForwardIterator2 last2,
//                              Compare comp);

#include <thrust/device_vector.h>
#include <thrust/sequence.h>

#include <cuda/iterator>
#include <cuda/memory_pool>
#include <cuda/std/execution>
#include <cuda/std/functional>
#include <cuda/std/iterator>
#include <cuda/stream>

#include <testing.cuh>
#include <utility.cuh>

#include "test_iterators.h"
#include "test_macros.h"
#include "test_pstl.h"

inline constexpr int size = 1000;

// Custom strict weak order so we exercise the explicit-Compare overload and
// also verify the duality lex_cmp(s1, s2, greater) == lex_cmp(s2, s1, less).
template <class Policy, class T>
void test_lexicographical_compare_comp(
  const Policy& policy, c2h::device_vector<T>& input1, c2h::device_vector<T>& input2)
{
  { // empty vs empty under any strict weak order
    const auto res = cuda::std::lexicographical_compare(
      policy,
      static_cast<T*>(nullptr),
      static_cast<T*>(nullptr),
      static_cast<T*>(nullptr),
      static_cast<T*>(nullptr),
      cuda::std::greater<>{});
    CHECK(res == false);
  }

  // Identical strictly-increasing sequences: equivalent under any strict weak order.
  thrust::sequence(input1.begin(), input1.end(), T(0));
  thrust::sequence(input2.begin(), input2.end(), T(0));
  { // equal ranges with greater<>: not less under reversed order either
    const auto res = cuda::std::lexicographical_compare(
      policy, input1.begin(), input1.end(), input2.begin(), input2.end(), cuda::std::greater<>{});
    CHECK(res == false);
  }

  // Make s1[42] < s2[42] under default order. Under greater<>, s2[42] becomes
  // "less than" s1[42], so the ordering flips: lex_cmp(s1, s2, greater) == false.
  input1[42] = static_cast<T>(0);
  input2[42] = static_cast<T>(size + 1);
  { // s1 "less" under default order, "greater" under greater<>
    const auto res = cuda::std::lexicographical_compare(
      policy, input1.begin(), input1.end(), input2.begin(), input2.end(), cuda::std::greater<>{});
    CHECK(res == false);
  }

  { // and the duality: lex_cmp(s2, s1, greater) == lex_cmp(s1, s2, less) == true
    const auto res = cuda::std::lexicographical_compare(
      policy, input2.begin(), input2.end(), input1.begin(), input1.end(), cuda::std::greater<>{});
    CHECK(res == true);
  }

  // Random access iterator wrapper: same divergence
  T* raw1 = thrust::raw_pointer_cast(input1.data());
  T* raw2 = thrust::raw_pointer_cast(input2.data());
  { // wrapped iterators, greater<>
    const auto res = cuda::std::lexicographical_compare(
      policy,
      random_access_iterator{raw2},
      random_access_iterator{raw2 + size},
      random_access_iterator{raw1},
      random_access_iterator{raw1 + size},
      cuda::std::greater<>{});
    CHECK(res == true);
  }

  // Prefix case under greater<>: identical sequences with n1 < n2 -> still true
  // (the shorter range is "less" under any strict weak order).
  input1[42] = static_cast<T>(42);
  input2[42] = static_cast<T>(42);
  { // s1 prefix of s2 under greater<>
    const auto res = cuda::std::lexicographical_compare(
      policy,
      input1.begin(),
      cuda::std::next(input1.begin(), size / 2),
      input2.begin(),
      input2.end(),
      cuda::std::greater<>{});
    CHECK(res == true);
  }
}

C2H_TEST("cuda::std::lexicographical_compare(first1, last1, first2, last2, comp)", "[parallel algorithm]", all_types)
{
  using T = typename c2h::get<0, TestType>;
  c2h::device_vector<T> input1(size, thrust::no_init);
  c2h::device_vector<T> input2(size, thrust::no_init);

  SECTION("with default stream")
  {
    const auto policy = cuda::execution::gpu;
    test_lexicographical_compare_comp(policy, input1, input2);
  }

  SECTION("with provided stream")
  {
    cuda::stream stream{cuda::device_ref{0}};
    const auto policy = cuda::execution::gpu.with(cuda::get_stream, stream);
    test_lexicographical_compare_comp(policy, input1, input2);
  }

  SECTION("with provided memory_resource")
  {
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(cuda::device_ref{0});
    const auto policy = cuda::execution::gpu.with(cuda::mr::get_memory_resource, device_resource);
    test_lexicographical_compare_comp(policy, input1, input2);
  }

  SECTION("with provided stream and memory_resource")
  {
    cuda::stream stream{cuda::device_ref{0}};
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(stream.device());
    const auto policy =
      cuda::execution::gpu.with(cuda::mr::get_memory_resource, device_resource).with(cuda::get_stream, stream);
    test_lexicographical_compare_comp(policy, input1, input2);
  }
}
