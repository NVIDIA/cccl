//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template <class Policy, class InputIterator, class Generator>
// void generate_n(const Policy& policy,
//                 InputIterator first,
//                 iterator_difference_t<InputIterator> count,
//                 Generator gen);

#include <thrust/device_vector.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>

#include <cuda/iterator>
#include <cuda/memory_pool>
#include <cuda/std/algorithm>
#include <cuda/std/execution>
#include <cuda/std/functional>
#include <cuda/stream>

#include <testing.cuh>
#include <utility.cuh>

#include "test_macros.h"

inline constexpr int size = 1000;

template <class T = int>
struct gen_val
{
  int val_;

  constexpr gen_val(const int val) noexcept
      : val_(val)
  {}

  TEST_DEVICE_FUNC constexpr T operator()() const noexcept
  {
    return static_cast<T>(val_);
  }
};

#include "test_iterators.h"
#include "test_macros.h"
#include "test_pstl.h"

template <class Policy, class T>
void test_generate_n(const Policy& policy, c2h::device_vector<T>& output)
{
  { // empty should not access anything
    const auto res = cuda::std::generate_n(policy, static_cast<T*>(nullptr), 0, gen_val{42});
    CHECK(res == nullptr);
  }

  cuda::std::fill(policy, output.begin(), output.end(), 0);
  { // contiguous iterators
    const auto res = cuda::std::generate_n(policy, output.begin(), size, gen_val<T>{42});
    CHECK(cuda::std::equal(policy, output.begin(), output.end(), cuda::constant_iterator{static_cast<T>(42)}));
    CHECK(res == output.end());
  }

  cuda::std::fill(policy, output.begin(), output.end(), 0);
  T* raw_pointer = thrust::raw_pointer_cast(output.data());
  { // random access iterators
    const auto res = cuda::std::generate_n(policy, random_access_iterator{raw_pointer}, size, gen_val<T>{42});
    CHECK(cuda::std::equal(policy, output.begin(), output.end(), cuda::constant_iterator{static_cast<T>(42)}));
    CHECK(res == random_access_iterator{raw_pointer + size});
  }

  cuda::std::fill(policy, output.begin(), output.end(), 0);
  { // contiguous iterators, convertible generator result
    const auto res = cuda::std::generate_n(policy, output.begin(), size, gen_val{42});
    CHECK(cuda::std::equal(policy, output.begin(), output.end(), cuda::constant_iterator{static_cast<T>(42)}));
    CHECK(res == output.end());
  }

  cuda::std::fill(policy, output.begin(), output.end(), 0);
  { // random access iterators, convertible generator result
    const auto res = cuda::std::generate_n(policy, random_access_iterator{raw_pointer}, size, gen_val{42});
    CHECK(cuda::std::equal(policy, output.begin(), output.end(), cuda::constant_iterator{static_cast<T>(42)}));
    CHECK(res == random_access_iterator{raw_pointer + size});
  }
}

C2H_TEST("cuda::std::generate_n", "[parallel algorithm]", all_types)
{
  using T = typename c2h::get<0, TestType>;
  c2h::device_vector<T> output(size, thrust::no_init);

  SECTION("with default stream")
  {
    const auto policy = cuda::execution::gpu;
    test_generate_n(policy, output);
  }

  SECTION("with provided stream")
  {
    cuda::stream stream{cuda::device_ref{0}};
    const auto policy = cuda::execution::gpu.with(cuda::get_stream, stream);
    test_generate_n(policy, output);
  }

  SECTION("with provided memory_resource")
  {
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(cuda::device_ref{0});
    const auto policy = cuda::execution::gpu.with(cuda::mr::get_memory_resource, device_resource);
    test_generate_n(policy, output);
  }

  SECTION("with provided stream and memory_resource")
  {
    cuda::stream stream{cuda::device_ref{0}};
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(stream.device());
    const auto policy =
      cuda::execution::gpu.with(cuda::mr::get_memory_resource, device_resource).with(cuda::get_stream, stream);
    test_generate_n(policy, output);
  }
}
