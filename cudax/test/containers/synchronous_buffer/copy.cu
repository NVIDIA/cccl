//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/memory_resource>
#include <cuda/std/__algorithm_>
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/initializer_list>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <cuda/experimental/container.cuh>

#include "helper.h"
#include "types.h"

// TODO: only device accessible resource
TEMPLATE_TEST_CASE("cudax::synchronous_buffer make_synchronous_buffer",
                   "[container][synchronous_buffer]",
                   cuda::std::tuple<cuda::mr::host_accessible>,
                   cuda::std::tuple<cuda::mr::device_accessible>,
                   (cuda::std::tuple<cuda::mr::host_accessible, cuda::mr::device_accessible>) )
{
  using Resource = typename extract_properties<TestType>::resource;
  using Buffer   = typename extract_properties<TestType>::synchronous_buffer;
  using T        = typename Buffer::value_type;

  Resource resource{};

  using MatchingAnyResource = typename extract_properties<TestType>::matching_any_resource;
  MatchingAnyResource matching_resource{resource};

  SECTION("Same resource")
  {
    { // empty input
      const Buffer input{resource};
      const Buffer buf = cudax::make_synchronous_buffer(input, input.get_memory_resource());
      CUDAX_CHECK(buf.empty());
      CUDAX_CHECK(buf.data() == nullptr);
    }

    { // non-empty input
      const Buffer input{resource, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      const Buffer buf = cudax::make_synchronous_buffer(input, input.get_memory_resource());
      CUDAX_CHECK(!buf.empty());
      CUDAX_CHECK(equal_range(buf));
    }

    { // empty input
      const Buffer input{resource};
      const Buffer buf = cudax::make_synchronous_buffer(input);
      CUDAX_CHECK(buf.empty());
      CUDAX_CHECK(buf.data() == nullptr);
    }

    { // non-empty input
      const Buffer input{resource, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      const Buffer buf = cudax::make_synchronous_buffer(input);
      CUDAX_CHECK(!buf.empty());
      CUDAX_CHECK(equal_range(buf));
    }
  }

  SECTION("Different resource")
  {
    { // empty input
      const Buffer input{resource};
      const auto buf = cudax::make_synchronous_buffer(input, matching_resource);
      static_assert(!cuda::std::is_same_v<Buffer, cuda::std::remove_const_t<decltype(buf)>>);
      CUDAX_CHECK(buf.empty());
      CUDAX_CHECK(buf.data() == nullptr);
    }

    { // non-empty input
      const Buffer input{resource, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      const auto buf = cudax::make_synchronous_buffer(input, matching_resource);
      static_assert(!cuda::std::is_same_v<Buffer, cuda::std::remove_const_t<decltype(buf)>>);
      CUDAX_CHECK(!buf.empty());
      CUDAX_CHECK(equal_range(buf));
    }

    { // empty input
      const Buffer input{resource};
      const auto buf = cudax::make_synchronous_buffer(input, matching_resource);
      static_assert(!cuda::std::is_same_v<Buffer, cuda::std::remove_const_t<decltype(buf)>>);
      CUDAX_CHECK(buf.empty());
      CUDAX_CHECK(buf.data() == nullptr);
    }

    { // non-empty input
      const Buffer input{resource, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      const auto buf = cudax::make_synchronous_buffer(input, matching_resource);
      static_assert(!cuda::std::is_same_v<Buffer, cuda::std::remove_const_t<decltype(buf)>>);
      CUDAX_CHECK(!buf.empty());
      CUDAX_CHECK(equal_range(buf));
    }
  }
}
