//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template <class Policy, class InputIterator, class OutputIterator, class Size>
// void copy_n(const Policy&  policy,
//             InputIterator  first,
//             Size           count,
//             OutoutIterator result);

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

inline constexpr int size = 1000;

C2H_TEST("cuda::std::copy_n", "[parallel algorithm]")
{
  thrust::device_vector<int> output(size, thrust::no_init);
  thrust::device_vector<int> input(size, thrust::no_init);
  thrust::sequence(input.begin(), input.end(), 0);

  SECTION("with default stream")
  {
    const auto policy = cuda::execution::__cub_par_unseq;

    thrust::fill(output.begin(), output.end(), -1);
    { // With non-contiguous iterator
      cuda::std::copy_n(policy, cuda::counting_iterator{0}, size, output.begin());
      CHECK(thrust::equal(output.begin(), output.end(), cuda::counting_iterator{0}));
    }

    thrust::fill(output.begin(), output.end(), -1);
    { // With contiguous iterator
      cuda::std::copy_n(policy, input.begin(), size, output.begin());
      CHECK(thrust::equal(output.begin(), output.end(), cuda::counting_iterator{0}));
    }
  }

  SECTION("with provided stream")
  {
    cuda::stream stream{cuda::device_ref{0}};
    const auto policy = cuda::execution::__cub_par_unseq.with_stream(stream);

    thrust::fill(output.begin(), output.end(), -1);
    { // With non-contiguous iterator
      cuda::std::copy_n(policy, cuda::counting_iterator{0}, size, output.begin());
      CHECK(thrust::equal(output.begin(), output.end(), cuda::counting_iterator{0}));
    }

    thrust::fill(output.begin(), output.end(), -1);
    { // With contiguous iterator
      cuda::std::copy_n(policy, input.begin(), size, output.begin());
      CHECK(thrust::equal(output.begin(), output.end(), cuda::counting_iterator{0}));
    }
  }

  SECTION("with provided memory_resource")
  {
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(cuda::device_ref{0});
    const auto policy = cuda::execution::__cub_par_unseq.with_memory_resource(device_resource);

    thrust::fill(output.begin(), output.end(), -1);
    { // With non-contiguous iterator
      cuda::std::copy_n(policy, cuda::counting_iterator{0}, size, output.begin());
      CHECK(thrust::equal(output.begin(), output.end(), cuda::counting_iterator{0}));
    }

    thrust::fill(output.begin(), output.end(), -1);
    { // With contiguous iterator
      cuda::std::copy_n(policy, input.begin(), size, output.begin());
      CHECK(thrust::equal(output.begin(), output.end(), cuda::counting_iterator{0}));
    }
  }

  SECTION("with provided stream and memory_resource")
  {
    cuda::stream stream{cuda::device_ref{0}};
    cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(stream.device());
    const auto policy = cuda::execution::__cub_par_unseq.with_memory_resource(device_resource).with_stream(stream);

    thrust::fill(output.begin(), output.end(), -1);
    { // With non-contiguous iterator
      cuda::std::copy_n(policy, cuda::counting_iterator{0}, size, output.begin());
      CHECK(thrust::equal(output.begin(), output.end(), cuda::counting_iterator{0}));
    }

    thrust::fill(output.begin(), output.end(), -1);
    { // With contiguous iterator
      cuda::std::copy_n(policy, input.begin(), size, output.begin());
      CHECK(thrust::equal(output.begin(), output.end(), cuda::counting_iterator{0}));
    }
  }
}
