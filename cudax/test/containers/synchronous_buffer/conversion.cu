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

TEMPLATE_TEST_CASE("cudax::synchronous_buffer conversion",
                   "[container][synchronous_buffer]",
                   cuda::std::tuple<cuda::mr::host_accessible>,
                   cuda::std::tuple<cuda::mr::device_accessible>,
                   (cuda::std::tuple<cuda::mr::host_accessible, cuda::mr::device_accessible>) )
{
  using Resource = typename extract_properties<TestType>::resource;
  using Buffer   = typename extract_properties<TestType>::synchronous_buffer;
  using T        = typename Buffer::value_type;

  Resource resource{};

  // Convert from a synchronous_buffer that has more properties than the current one
  using MatchingBuffer   = typename extract_properties<TestType>::matching_buffer;
  using MatchingResource = typename extract_properties<TestType>::matching_resource;
  MatchingResource matching_resource{resource};

  SECTION("cudax::synchronous_buffer construction with matching synchronous_buffer")
  {
    { // can be copy constructed from empty input
      const MatchingBuffer input{matching_resource, 0};
      Buffer buf(input);
      CUDAX_CHECK(buf.empty());
      CUDAX_CHECK(input.empty());
    }

    { // can be copy constructed from non-empty input
      const MatchingBuffer input{matching_resource, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      Buffer buf(input);
      CUDAX_CHECK(!buf.empty());
      CUDAX_CHECK(equal_range(buf));
      CUDAX_CHECK(equal_range(input));
    }

    { // can be move constructed with empty input
      MatchingBuffer input{matching_resource, 0};
      Buffer buf(cuda::std::move(input));
      CUDAX_CHECK(buf.empty());
      CUDAX_CHECK(input.empty());
    }

    { // can be move constructed from non-empty input
      MatchingBuffer input{matching_resource, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};

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
