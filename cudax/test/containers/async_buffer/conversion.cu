//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/memory_resource>
#include <cuda/std/__algorithm_>
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/initializer_list>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>

#include <cuda/experimental/container.cuh>

#include "helper.h"
#include "test_resources.h"
#include "types.h"

#if _CCCL_CUDACC_AT_LEAST(12, 6)
using test_types = c2h::type_list<cuda::std::tuple<int, cuda::mr::host_accessible>,
                                  cuda::std::tuple<unsigned long long, cuda::mr::device_accessible>,
                                  cuda::std::tuple<int, cuda::mr::host_accessible, cuda::mr::device_accessible>>;
#else
using test_types = c2h::type_list<cuda::std::tuple<int, cuda::mr::device_accessible>>;
#endif

C2H_CCCLRT_TEST("cudax::async_buffer conversion", "[container][async_buffer]", test_types)
{
  using TestT    = c2h::get<0, TestType>;
  using Resource = typename extract_properties<TestT>::resource;
  using Buffer   = typename extract_properties<TestT>::async_buffer;
  using T        = typename Buffer::value_type;

  cudax::stream stream{cuda::device_ref{0}};
  Resource resource{};

  // Convert from a async_buffer that has more properties than the current one
  using MatchingBuffer   = typename extract_properties<TestT>::matching_vector;
  using MatchingResource = typename extract_properties<TestT>::matching_resource;

  SECTION("cudax::async_buffer construction with matching async_buffer")
  {
    { // can be copy constructed from empty input
      const MatchingBuffer input{stream, resource, 0, cudax::no_init};
      Buffer buf(input);
      CUDAX_CHECK(buf.empty());
      CUDAX_CHECK(input.empty());
    }

    { // can be copy constructed from non-empty input
      const MatchingBuffer input{stream, resource, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      Buffer buf(input);
      CUDAX_CHECK(!buf.empty());
      CUDAX_CHECK(equal_range(buf));
      CUDAX_CHECK(equal_range(input));
    }

    { // can be move constructed with empty input
      MatchingBuffer input{stream, resource, 0, cudax::no_init};
      Buffer buf(cuda::std::move(input));
      CUDAX_CHECK(buf.empty());
      CUDAX_CHECK(input.empty());
    }

    { // can be move constructed from non-empty input
      MatchingBuffer input{stream, resource, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};

      // ensure that we steal the data
      const auto* allocation = input.data();
      Buffer buf(cuda::std::move(input));
      CUDAX_CHECK(buf.size() == 6);
      CUDAX_CHECK(buf.data() == allocation);
      CUDAX_CHECK(input.size() == 0);
      CUDAX_CHECK(input.data() == nullptr);
      CUDAX_CHECK(equal_range(buf));
    }
  }
}
