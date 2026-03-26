//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template <class ExecutionPolicy, class ForwardIt1, class ForwardIt2, class ForwardIt3>
// ForwardIt3 merge(ExecutionPolicy&& policy,
//                  ForwardIt1 first1, ForwardIt1 last1,
//                  ForwardIt2 first2, ForwardIt2 last2,
//                  ForwardIt3 d_first);
//
// template <class ExecutionPolicy, class ForwardIt1, class ForwardIt2,
//           class ForwardIt3, class Compare>
// ForwardIt3 merge(ExecutionPolicy&& policy,
//                  ForwardIt1 first1, ForwardIt1 last1,
//                  ForwardIt2 first2, ForwardIt2 last2,
//                  ForwardIt3 d_first, Compare comp);

#include <thrust/device_vector.h>
#include <thrust/equal.h>

#include <cuda/iterator>
#include <cuda/memory_pool>
#include <cuda/std/__pstl_algorithm>
#include <cuda/std/execution>
#include <cuda/std/functional>
#include <cuda/std/iterator>
#include <cuda/stream>

#include <testing.cuh>
#include <utility.cuh>

inline constexpr int size1 = 1000;
inline constexpr int size2 = 500;

template <class Policy>
void test_merge(const Policy& policy,
                const thrust::device_vector<int>& in1,
                const thrust::device_vector<int>& in2,
                thrust::device_vector<int>& out)
{
  { // both ranges empty
    const auto res = cuda::std::merge(
      policy,
      static_cast<int*>(nullptr),
      static_cast<int*>(nullptr),
      static_cast<int*>(nullptr),
      static_cast<int*>(nullptr),
      out.begin());
    CHECK(res == out.begin());
  }

  cuda::std::fill(policy, out.begin(), out.end(), -1);
  { // empty first range
    const auto res = cuda::std::merge(
      policy, static_cast<int*>(nullptr), static_cast<int*>(nullptr), in2.begin(), in2.end(), out.begin());
    CHECK(res == out.begin() + size2);
    CHECK(thrust::equal(out.begin(), res, in2.begin()));
  }

  cuda::std::fill(policy, out.begin(), out.end(), -1);
  { // empty second range
    const auto res = cuda::std::merge(
      policy, in1.begin(), in1.end(), static_cast<int*>(nullptr), static_cast<int*>(nullptr), out.begin());
    CHECK(res == out.begin() + size1);
    CHECK(thrust::equal(out.begin(), res, in1.begin()));
  }

  cuda::std::fill(policy, out.begin(), out.end(), -1);
  { // two sorted ranges:
    const auto res = cuda::std::merge(policy, in1.begin(), in1.end(), in2.begin(), in2.end(), out.begin());
    CHECK(res == out.end());

    // in1 = [0,2,4,..., 2 * size1), in2 = [1,3,5,..., 2 * size2)
    // First subrange is equal [0, 1, ..., 2 * size2)
    // The remaining elements are equal to [2 * size2 - 1, 2 * size2 + 1, ..., 2 * size2)
    const auto mid = out.begin() + 2 * size2;
    CHECK(thrust::equal(out.begin(), mid, cuda::counting_iterator{0}));
    CHECK(thrust::equal(mid, out.end(), cuda::strided_iterator{cuda::counting_iterator{2 * size2}, 2}));
  }

  cuda::std::fill(policy, out.begin(), out.end(), -1);
  { // first range non-contiguous
    const auto in  = cuda::strided_iterator{cuda::counting_iterator{0}, 2};
    const auto res = cuda::std::merge(policy, in, in + size1, in2.begin(), in2.end(), out.begin());
    CHECK(res == out.end());

    // in1 = [0,2,4,..., 2 * size1), in2 = [1,3,5,..., 2 * size2)
    // First subrange is equal [0, 1, ..., 2 * size2)
    // The remaining elements are equal to [2 * size2 - 1, 2 * size2 + 1, ..., 2 * size2)
    const auto mid = out.begin() + 2 * size2;
    CHECK(thrust::equal(out.begin(), mid, cuda::counting_iterator{0}));
    CHECK(thrust::equal(mid, out.end(), cuda::strided_iterator{cuda::counting_iterator{2 * size2}, 2}));
  }

  cuda::std::fill(policy, out.begin(), out.end(), -1);
  { // second range non-contiguous:
    const auto in  = cuda::strided_iterator{cuda::counting_iterator{1}, 2};
    const auto res = cuda::std::merge(policy, in1.begin(), in1.end(), in, in + size2, out.begin());
    CHECK(res == out.end());

    // in1 = [0,2,4,..., 2 * size1), in2 = [1,3,5,..., 2 * size2)
    // First subrange is equal [0, 1, ..., 2 * size2)
    // The remaining elements are equal to [2 * size2 - 1, 2 * size2 + 1, ..., 2 * size2)
    const auto mid = out.begin() + 2 * size2;
    CHECK(thrust::equal(out.begin(), mid, cuda::counting_iterator{0}));
    CHECK(thrust::equal(mid, out.end(), cuda::strided_iterator{cuda::counting_iterator{2 * size2}, 2}));
  }
}

C2H_TEST("cuda::std::merge", "[parallel algorithm]")
{
  thrust::device_vector<int> in1(size1, thrust::no_init);
  thrust::device_vector<int> in2(size2, thrust::no_init);
  thrust::device_vector<int> out(size1 + size2, thrust::no_init);

  cuda::strided_iterator iter1{cuda::counting_iterator{0}, 2}; // [0,2,4,..., 2 * size1)
  cuda::strided_iterator iter2{cuda::counting_iterator{1}, 2}; // [1,3,5,..., 2 * size2)
  cuda::std::copy_n(cuda::execution::__cub_par_unseq, iter1, size1, in1.begin());
  cuda::std::copy_n(cuda::execution::__cub_par_unseq, iter2, size2, in2.begin());

  SECTION("with default stream")
  {
    const auto policy = cuda::execution::__cub_par_unseq;
    test_merge(policy, in1, in2, out);
  }

  SECTION("with provided stream")
  {
    cuda::stream stream{cuda::device_ref{0}};
    const auto policy = cuda::execution::__cub_par_unseq.with(cuda::get_stream, stream);
    test_merge(policy, in1, in2, out);
  }

  SECTION("with provided memory_resource")
  {
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(cuda::device_ref{0});
    const auto policy = cuda::execution::__cub_par_unseq.with(cuda::mr::get_memory_resource, device_resource);
    test_merge(policy, in1, in2, out);
  }

  SECTION("with provided stream and memory_resource")
  {
    cuda::stream stream{cuda::device_ref{0}};
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(stream.device());
    const auto policy = cuda::execution::__cub_par_unseq.with(cuda::mr::get_memory_resource, device_resource)
                          .with(cuda::get_stream, stream);
    test_merge(policy, in1, in2, out);
  }
}
