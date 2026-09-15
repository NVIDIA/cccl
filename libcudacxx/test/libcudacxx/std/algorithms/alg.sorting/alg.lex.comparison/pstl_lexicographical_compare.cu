//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2>
// bool lexicographical_compare(ExecutionPolicy&& exec,
//                              ForwardIterator1 first1, ForwardIterator1 last1,
//                              ForwardIterator2 first2, ForwardIterator2 last2);

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

template <class Policy, class T>
void test_lexicographical_compare(const Policy& policy, c2h::device_vector<T>& input1, c2h::device_vector<T>& input2)
{
  { // empty vs empty: equal => not less
    const auto res = cuda::std::lexicographical_compare(
      policy, static_cast<T*>(nullptr), static_cast<T*>(nullptr), static_cast<T*>(nullptr), static_cast<T*>(nullptr));
    CHECK(res == false);
  }

  // Build identical sequences in both buffers.
  thrust::sequence(input1.begin(), input1.end(), T(0));
  thrust::sequence(input2.begin(), input2.end(), T(0));

  { // empty s1 vs non-empty s2: empty < non-empty => true (no element access on s1 side)
    const auto res =
      cuda::std::lexicographical_compare(policy, input1.begin(), input1.begin(), input2.begin(), input2.end());
    CHECK(res == true);
  }

  { // non-empty s1 vs empty s2: false (no element access on s2 side)
    const auto res =
      cuda::std::lexicographical_compare(policy, input1.begin(), input1.end(), input2.begin(), input2.begin());
    CHECK(res == false);
  }

  { // single-element ranges, equal: false (covers the n=1 path through both passes)
    const auto res = cuda::std::lexicographical_compare(
      policy, input1.begin(), cuda::std::next(input1.begin(), 1), input2.begin(), cuda::std::next(input2.begin(), 1));
    CHECK(res == false);
  }

  // Single-element with s1[0] < s2[0] => true. Restored after this block.
  input1[0] = static_cast<T>(0);
  input2[0] = static_cast<T>(1);
  { // single-element ranges, s1 < s2
    const auto res = cuda::std::lexicographical_compare(
      policy, input1.begin(), cuda::std::next(input1.begin(), 1), input2.begin(), cuda::std::next(input2.begin(), 1));
    CHECK(res == true);
  }
  { // single-element ranges, s1 > s2 (swap argument order)
    const auto res = cuda::std::lexicographical_compare(
      policy, input2.begin(), cuda::std::next(input2.begin(), 1), input1.begin(), cuda::std::next(input1.begin(), 1));
    CHECK(res == false);
  }
  input1[0] = static_cast<T>(0);
  input2[0] = static_cast<T>(0);

  { // equal ranges of same length: not less
    const auto res =
      cuda::std::lexicographical_compare(policy, input1.begin(), input1.end(), input2.begin(), input2.end());
    CHECK(res == false);
  }

  { // s1 is a strict prefix of s2 (n1 < n2, equiv up to n1) => true
    const auto res = cuda::std::lexicographical_compare(
      policy, input1.begin(), cuda::std::next(input1.begin(), size / 2), input2.begin(), input2.end());
    CHECK(res == true);
  }

  { // s2 is a strict prefix of s1 (n2 < n1, equiv up to n2) => false
    const auto res = cuda::std::lexicographical_compare(
      policy, input1.begin(), input1.end(), input2.begin(), cuda::std::next(input2.begin(), size / 2));
    CHECK(res == false);
  }

  T* raw1 = thrust::raw_pointer_cast(input1.data());
  T* raw2 = thrust::raw_pointer_cast(input2.data());

  { // equal ranges via random_access_iterator wrapper
    const auto res = cuda::std::lexicographical_compare(
      policy,
      random_access_iterator{raw1},
      random_access_iterator{raw1 + size},
      random_access_iterator{raw2},
      random_access_iterator{raw2 + size});
    CHECK(res == false);
  }

  // Make s1[42] < s2[42] (early divergence, "a < b" wins) => true.
  input1[42] = static_cast<T>(0);
  input2[42] = static_cast<T>(size + 1);
  { // s1 < s2 at index 42, contiguous range
    const auto res =
      cuda::std::lexicographical_compare(policy, input1.begin(), input1.end(), input2.begin(), input2.end());
    CHECK(res == true);
  }

  { // and the reverse: s2 < s1 at index 42 => false
    const auto res =
      cuda::std::lexicographical_compare(policy, input2.begin(), input2.end(), input1.begin(), input1.end());
    CHECK(res == false);
  }

  { // same divergence under random_access_iterator wrapper
    const auto res = cuda::std::lexicographical_compare(
      policy,
      random_access_iterator{raw1},
      random_access_iterator{raw1 + size},
      random_access_iterator{raw2},
      random_access_iterator{raw2 + size});
    CHECK(res == true);
  }

  // Restore equality, then put divergence at the very last index.
  input1[42]       = static_cast<T>(42);
  input2[42]       = static_cast<T>(42);
  input1[size - 1] = static_cast<T>(0);
  input2[size - 1] = static_cast<T>(size + 1);
  { // late divergence still detected
    const auto res =
      cuda::std::lexicographical_compare(policy, input1.begin(), input1.end(), input2.begin(), input2.end());
    CHECK(res == true);
  }

  // Restore the last element to its sequence value so both ranges are identical
  // again, then verify the equal-ranges case still returns false (sanity check
  // that the late-divergence test above did not leave residual state).
  input1[size - 1] = static_cast<T>(size - 1);
  input2[size - 1] = static_cast<T>(size - 1);
  { // ranges restored to equal
    const auto res =
      cuda::std::lexicographical_compare(policy, input1.begin(), input1.end(), input2.begin(), input2.end());
    CHECK(res == false);
  }
}

C2H_TEST("cuda::std::lexicographical_compare(first1, last1, first2, last2)", "[parallel algorithm]", all_types)
{
  using T = typename c2h::get<0, TestType>;
  c2h::device_vector<T> input1(size, thrust::no_init);
  c2h::device_vector<T> input2(size, thrust::no_init);

  SECTION("with default stream")
  {
    const auto policy = cuda::execution::gpu;
    test_lexicographical_compare(policy, input1, input2);
  }

  SECTION("with provided stream")
  {
    cuda::stream stream{cuda::device_ref{0}};
    const auto policy = cuda::execution::gpu.with(cuda::get_stream, stream);
    test_lexicographical_compare(policy, input1, input2);
  }

  SECTION("with provided memory_resource")
  {
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(cuda::device_ref{0});
    const auto policy = cuda::execution::gpu.with(cuda::mr::get_memory_resource, device_resource);
    test_lexicographical_compare(policy, input1, input2);
  }

  SECTION("with provided stream and memory_resource")
  {
    cuda::stream stream{cuda::device_ref{0}};
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(stream.device());
    const auto policy =
      cuda::execution::gpu.with(cuda::mr::get_memory_resource, device_resource).with(cuda::get_stream, stream);
    test_lexicographical_compare(policy, input1, input2);
  }
}
