//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class BinaryPred>
// bool equal(ExecutionPolicy&& exec
//            ForwardIterator1 first1,
//            ForwardIterator1 last1,
//            ForwardIterator2 first2,
//            BinaryPred pred);

// template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class BinaryPred>
// bool equal(ExecutionPolicy&& exec
//            ForwardIterator1 first1,
//            ForwardIterator1 last1,
//            ForwardIterator2 first2,
//            ForwardIterator2 first2,
//            BinaryPred pred);

#include <cuda/iterator>
#include <cuda/memory_pool>
#include <cuda/std/__pstl_algorithm>
#include <cuda/std/execution>
#include <cuda/std/functional>
#include <cuda/stream>

#include <testing.cuh>
#include <utility.cuh>

inline constexpr short size = 1000;

template <class Policy>
void test_equal(const Policy& policy)
{
  { // empty should not access anything
    const auto res = cuda::std::equal(
      policy,
      static_cast<int*>(nullptr),
      static_cast<int*>(nullptr),
      cuda::counting_iterator{short{0}},
      cuda::std::equal_to<>{});
    CHECK(res);
  }

  { // same type
    const auto res = cuda::std::equal(
      policy,
      cuda::counting_iterator{short{0}},
      cuda::counting_iterator{size},
      cuda::counting_iterator{short{0}},
      cuda::std::equal_to<>{});
    CHECK(res);
  }

  { // convertible pred
    const auto res = cuda::std::equal(
      policy,
      cuda::counting_iterator{short{0}},
      cuda::counting_iterator{short{size}},
      cuda::counting_iterator{short{0}},
      cuda::std::equal_to<int>{});
    CHECK(res);
  }

  { // convertible type
    const auto res = cuda::std::equal(
      policy,
      cuda::counting_iterator{short{0}},
      cuda::counting_iterator{size},
      cuda::counting_iterator{int{0}},
      cuda::std::equal_to<>{});
    CHECK(res);
  }

  { // convertible type and pred
    const auto res = cuda::std::equal(
      policy,
      cuda::counting_iterator{short{0}},
      cuda::counting_iterator{short{size}},
      cuda::counting_iterator{int{0}},
      cuda::std::equal_to<cuda::std::ptrdiff_t>{});
    CHECK(res);
  }
}

C2H_TEST("cuda::std::equal(first1, last1, first2, pred)", "[parallel algorithm]")
{
  SECTION("with default stream")
  {
    const auto policy = cuda::execution::__cub_par_unseq;
    test_equal(policy);
  }

  SECTION("with provided stream")
  {
    cuda::stream stream{cuda::device_ref{0}};
    const auto policy = cuda::execution::__cub_par_unseq.with_stream(stream);
    test_equal(policy);
  }

  SECTION("with provided memory_resource")
  {
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(cuda::device_ref{0});
    const auto policy = cuda::execution::__cub_par_unseq.with_memory_resource(device_resource);
    test_equal(policy);
  }

  SECTION("with provided stream and memory_resource")
  {
    cuda::stream stream{cuda::device_ref{0}};
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(stream.device());
    const auto policy = cuda::execution::__cub_par_unseq.with_stream(stream).with_memory_resource(device_resource);
    test_equal(policy);
  }
}

template <class Policy>
void test_equal2(const Policy& policy)
{
  { // empty should not access anything, even if both are empty
    const auto res = cuda::std::equal(
      policy,
      static_cast<int*>(nullptr),
      static_cast<int*>(nullptr),
      static_cast<short*>(nullptr),
      static_cast<short*>(nullptr),
      cuda::std::equal_to<>{});
    CHECK(res);
  }

  { // empty should not access anything, but if one is nonempty it fails
    const auto res = cuda::std::equal(
      policy,
      static_cast<int*>(nullptr),
      static_cast<int*>(nullptr),
      cuda::counting_iterator{short{0}},
      cuda::counting_iterator{size},
      cuda::std::equal_to<>{});
    CHECK(!res);
  }

  { // same type
    const auto res = cuda::std::equal(
      policy,
      cuda::counting_iterator{short{0}},
      cuda::counting_iterator{size},
      cuda::counting_iterator{short{0}},
      cuda::counting_iterator{size},
      cuda::std::equal_to<>{});
    CHECK(res);
  }

  { // convertible pred
    const auto res = cuda::std::equal(
      policy,
      cuda::counting_iterator{int{0}},
      cuda::counting_iterator{int{size}},
      cuda::counting_iterator{int{0}},
      cuda::counting_iterator{int{size}},
      cuda::std::equal_to<cuda::std::ptrdiff_t>{});
    CHECK(res);
  }

  { // convertible type
    const auto res = cuda::std::equal(
      policy,
      cuda::counting_iterator{short{0}},
      cuda::counting_iterator{size},
      cuda::counting_iterator{int{0}},
      cuda::counting_iterator{int{size}},
      cuda::std::equal_to<>{});
    CHECK(res);
  }

  { // convertible pred
    const auto res = cuda::std::equal(
      policy,
      cuda::counting_iterator{short{0}},
      cuda::counting_iterator{short{size}},
      cuda::counting_iterator{int{0}},
      cuda::counting_iterator{int{size}},
      cuda::std::equal_to<cuda::std::ptrdiff_t>{});
    CHECK(res);
  }
}

C2H_TEST("cuda::std::equal(first1, last1, first2, last2, pred)", "[parallel algorithm]")
{
  SECTION("with default stream")
  {
    const auto policy = cuda::execution::__cub_par_unseq;
    test_equal2(policy);
  }

  SECTION("with provided stream")
  {
    cuda::stream stream{cuda::device_ref{0}};
    const auto policy = cuda::execution::__cub_par_unseq.with_stream(stream);
    test_equal2(policy);
  }

  SECTION("with provided memory_resource")
  {
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(cuda::device_ref{0});
    const auto policy = cuda::execution::__cub_par_unseq.with_memory_resource(device_resource);
    test_equal2(policy);
  }

  SECTION("with provided stream and memory_resource")
  {
    cuda::stream stream{cuda::device_ref{0}};
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(stream.device());
    const auto policy = cuda::execution::__cub_par_unseq.with_stream(stream).with_memory_resource(device_resource);
    test_equal2(policy);
  }
}
