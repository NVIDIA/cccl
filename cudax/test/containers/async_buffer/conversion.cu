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

TEMPLATE_TEST_CASE("cudax::async_buffer conversion",
                   "[container][async_buffer]",
                   cuda::std::tuple<cuda::mr::host_accessible>,
                   cuda::std::tuple<cuda::mr::device_accessible>,
                   (cuda::std::tuple<cuda::mr::host_accessible, cuda::mr::device_accessible>) )
{
  using Env      = typename extract_properties<TestType>::env;
  using Resource = typename extract_properties<TestType>::resource;
  using Buffer   = typename extract_properties<TestType>::async_buffer;
  using T        = typename Buffer::value_type;

  cudax::stream stream{};
  Resource resource{};
  Env env{resource, stream};

  // Convert from a async_buffer that has more properties than the current one
  using MatchingBuffer   = typename extract_properties<TestType>::matching_vector;
  using MatchingResource = typename extract_properties<TestType>::matching_resource;
  Env matching_env{MatchingResource{resource}, stream};

  SECTION("cudax::async_buffer construction with matching async_buffer")
  {
    { // can be copy constructed from empty input
      const MatchingBuffer input{matching_env, 0};
      Buffer buf(input);
      CHECK(buf.empty());
      CHECK(input.empty());
    }

    { // can be copy constructed from non-empty input
      const MatchingBuffer input{matching_env, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      Buffer buf(input);
      CHECK(!buf.empty());
      CHECK(equal_range(buf));
      CHECK(equal_range(input));
    }

    { // can be move constructed with empty input
      MatchingBuffer input{matching_env, 0};
      Buffer buf(cuda::std::move(input));
      CHECK(buf.empty());
      CHECK(input.empty());
    }

    { // can be move constructed from non-empty input
      MatchingBuffer input{matching_env, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};

      // ensure that we steal the data
      const auto* allocation = input.data();
      Buffer buf(cuda::std::move(input));
      CHECK(buf.size() == 6);
      CHECK(buf.data() == allocation);
      CHECK(input.size() == 0);
      CHECK(input.data() == nullptr);
      CHECK(equal_range(buf));
    }
  }

#if 0
  SECTION("cudax::async_buffer copy assignment of matching async_buffer")
  {
    { // Can be assigned an empty input, no allocation
      const MatchingBuffer input{matching_env};
      Buffer buf{env};
      buf = input;
      CHECK(buf.empty());
      CHECK(buf.data() == nullptr);
    }

    { // Can be assigned an empty input, shrinking
      const MatchingBuffer input{matching_env};
      Buffer buf{env, 4, T(-2)};
      buf = input;
      CHECK(buf.empty());
      CHECK(buf.data() == nullptr);
    }

    { // Can be assigned a non-empty input, shrinking
      const MatchingBuffer input{matching_env, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      Buffer buf{env, 42, T(-2)};
      buf = input;
      CHECK(!buf.empty());
      CHECK(equal_range(buf));
    }

    { // Can be assigned a non-empty input, growing from empty with reallocation
      const MatchingBuffer input{matching_env, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      Buffer buf{env};
      buf = input;
      CHECK(buf.size() == 6);
      CHECK(equal_range(buf));
    }

    { // Can be assigned a non-empty input, growing with reallocation
      const MatchingBuffer input{matching_env, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      Buffer buf{env, 4, T(-2)};
      buf = input;
      CHECK(buf.size() == 6);
      CHECK(equal_range(buf));
    }
  }

  SECTION("cudax::async_buffer move-assignment matching async_buffer")
  {
    { // Can be move-assigned an empty input
      MatchingBuffer input{matching_env};
      CHECK(input.empty());
      CHECK(input.data() == nullptr);

      Buffer buf{env};
      buf = cuda::std::move(input);
      CHECK(buf.empty());
      CHECK(buf.data() == nullptr);
      CHECK(input.empty());
      CHECK(input.data() == nullptr);
    }

    { // Can be move-assigned an empty input, shrinking
      MatchingBuffer input{matching_env};
      CHECK(input.empty());
      CHECK(input.data() == nullptr);

      Buffer buf{env, 4};
      buf = cuda::std::move(input);
      CHECK(buf.empty());
      CHECK(buf.data() == nullptr);
      CHECK(input.empty());
      CHECK(input.data() == nullptr);
    }

    { // Can be move-assigned a non-empty input, shrinking
      MatchingBuffer input{matching_env, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      Buffer buf{env, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      buf = cuda::std::move(input);
      CHECK(buf.size() == 6);
      CHECK(equal_range(buf));
      CHECK(input.empty());
      CHECK(input.data() == nullptr);
    }

    { // Can be move-assigned an non-empty input growing from empty
      MatchingBuffer input{matching_env, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      Buffer buf{env};
      buf = cuda::std::move(input);
      CHECK(buf.size() == 6);
      CHECK(equal_range(buf));
      CHECK(input.empty());
      CHECK(input.data() == nullptr);
    }
  }
#endif
}
