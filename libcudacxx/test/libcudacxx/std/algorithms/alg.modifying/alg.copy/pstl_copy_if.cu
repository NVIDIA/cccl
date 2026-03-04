//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template <class Policy, class InputIterator, class OutputIterator>
// void copy_if(const Policy&  policy,
//              InputIterator  first,
//              InputIterator  last,
//              OutoutIterator result);

#include <thrust/device_vector.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>

#include <cuda/cmath>
#include <cuda/iterator>
#include <cuda/memory_pool>
#include <cuda/std/__pstl_algorithm>
#include <cuda/std/execution>
#include <cuda/stream>

#include <testing.cuh>
#include <utility.cuh>

inline constexpr int size = 10000;

struct is_even
{
  __device__ constexpr bool operator()(const int& val) const noexcept
  {
    return (val % 2) == 0;
  }
};

struct check_is_even
{
  __device__ void operator()(const cuda::std::ptrdiff_t pos, const int value) const noexcept
  {
    _CCCL_VERIFY(static_cast<int>(pos * 2) == value, "Invalid position");
    _CCCL_VERIFY(is_even{}(value), "Not even");
  }
};

template <class Policy>
void test_copy_if(const Policy& policy, const thrust::device_vector<int>& input, thrust::device_vector<int>& output)
{
  { // empty should not access anything
    const auto res = cuda::std::copy_if(
      policy, static_cast<int*>(nullptr), static_cast<int*>(nullptr), static_cast<int*>(nullptr), is_even{});
    CHECK(res == nullptr);
  }

  { // Same input output type
    { // With random_access iterator
      thrust::fill(output.begin(), output.end(), -1);
      const auto res = cuda::std::copy_if(
        policy, cuda::counting_iterator{0}, cuda::counting_iterator{size}, output.begin(), is_even{});
      CHECK(thrust::equal(output.begin(), output.end(), cuda::strided_iterator{cuda::counting_iterator{0}, 2}));
      CHECK(res == output.end());
    }

    { // With contiguous iterator
      thrust::fill(output.begin(), output.end(), -1);
      const auto res = cuda::std::copy_if(policy, input.begin(), input.end(), output.begin(), is_even{});
      CHECK(thrust::equal(output.begin(), output.end(), cuda::strided_iterator{cuda::counting_iterator{0}, 2}));
      CHECK(res == output.end());
    }

    { // With pointer
      thrust::fill(output.begin(), output.end(), -1);
      auto ptr       = thrust::raw_pointer_cast(input.data());
      const auto res = cuda::std::copy_if(policy, ptr, ptr + size, output.begin(), is_even{});
      CHECK(thrust::equal(output.begin(), output.end(), cuda::strided_iterator{cuda::counting_iterator{0}, 2}));
      CHECK(res == output.end());
    }
  }

  { // Different input type
    thrust::fill(output.begin(), output.end(), -1);
    const auto res = cuda::std::copy_if(
      policy, cuda::counting_iterator{short{0}}, cuda::counting_iterator{short{size}}, output.begin(), is_even{});
    CHECK(thrust::equal(output.begin(), output.end(), cuda::strided_iterator{cuda::counting_iterator{0}, 2}));
    CHECK(res == output.end());
  }

  { // Random access output type
    const auto res = cuda::std::copy_if(
      policy,
      cuda::counting_iterator{0},
      cuda::counting_iterator{size},
      cuda::tabulate_output_iterator{check_is_even{}, 0},
      is_even{});
    CHECK(res == cuda::tabulate_output_iterator{check_is_even{}, size / 2});
  }

  { // Random access output type with conversion during assignment
    const auto res = cuda::std::copy_if(
      policy,
      cuda::counting_iterator{short{0}},
      cuda::counting_iterator{short{size}},
      cuda::tabulate_output_iterator{check_is_even{}, 0},
      is_even{});
    CHECK(res == cuda::tabulate_output_iterator{check_is_even{}, size / 2});
  }
}

C2H_TEST("cuda::std::copy_if", "[parallel algorithm]")
{
  thrust::device_vector<int> output(size / 2, thrust::no_init);
  thrust::device_vector<int> input(size, thrust::no_init);
  thrust::sequence(input.begin(), input.end(), 0);

  SECTION("with default stream")
  {
    const auto policy = cuda::execution::__cub_par_unseq;
    test_copy_if(policy, input, output);
  }

  SECTION("with provided stream")
  {
    cuda::stream stream{cuda::device_ref{0}};
    const auto policy = cuda::execution::__cub_par_unseq.with_stream(stream);
    test_copy_if(policy, input, output);
  }

  SECTION("with provided memory_resource")
  {
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(cuda::device_ref{0});
    const auto policy = cuda::execution::__cub_par_unseq.with_memory_resource(device_resource);
    test_copy_if(policy, input, output);
  }

  SECTION("with provided stream and memory_resource")
  {
    cuda::stream stream{cuda::device_ref{0}};
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(stream.device());
    const auto policy = cuda::execution::__cub_par_unseq.with_memory_resource(device_resource).with_stream(stream);
    test_copy_if(policy, input, output);
  }
}
