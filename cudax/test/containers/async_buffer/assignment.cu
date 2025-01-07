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
#include <catch2/catch.hpp>

// TODO: only device accessible resource
TEMPLATE_TEST_CASE("cudax::async_buffer assignment",
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

  using MatchingResource = typename extract_properties<TestType>::matching_resource;
  Env matching_env{MatchingResource{resource}, stream};

  SECTION("cudax::async_buffer copy-assignment")
  {
    { // Can be copy-assigned an empty input
      const Buffer input{env};
      Buffer buf{env};
      buf = input;
      CHECK(buf.empty());
      CHECK(buf.data() == nullptr);
    }
    { // Can be copy-assigned an empty input, shrinking
      const Buffer input{env};
      Buffer buf{env, 4};
      buf = input;
      CHECK(buf.empty());
      CHECK(buf.data() != nullptr);
    }

    { // Can be copy-assigned a non-empty input, shrinking
      const Buffer input{env, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      Buffer buf{env, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      buf = input;
      CHECK(!buf.empty());
      CHECK(equal_range(buf));
    }

    { // Can be copy-assigned a non-empty input, growing from empty with reallocation
      const Buffer input{env, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      Buffer buf{env};
      buf = input;
      CHECK(buf.size() == 6);
      CHECK(equal_range(buf));
    }

    { // Can be copy-assigned a non-empty input, growing with reallocation
      const Buffer input{env, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      Buffer buf{env, 2};
      buf = input;
      CHECK(buf.size() == 6);
      CHECK(equal_range(buf));
    }
  }

  SECTION("cudax::async_buffer copy-assignment different resource")
  {
    { // Can be copy-assigned an empty input
      const Buffer input{matching_env};
      Buffer buf{env};
      buf = input;
      CHECK(buf.empty());
      CHECK(buf.data() == nullptr);
    }

    { // Can be copy-assigned an empty input, shrinking
      const Buffer input{matching_env};
      Buffer buf{env, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      buf = input;
      CHECK(buf.empty());
      CHECK(buf.data() == nullptr);
    }

    { // Can be copy-assigned a non-empty input, shrinking
      const Buffer input{matching_env, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      Buffer buf{env, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      buf = input;
      CHECK(buf.size() == 6);
      CHECK(equal_range(buf));
    }

    { // Can be copy-assigned an non-empty input growing from empty without capacity
      const Buffer input{matching_env, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      Buffer buf{env};
      buf = input;
      CHECK(buf.size() == 6);
      CHECK(equal_range(buf));
    }
  }

  SECTION("cudax::async_buffer move-assignment")
  {
    { // Can be move-assigned an empty input
      Buffer input{env};
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
      Buffer input{env};
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
      Buffer input{env, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      Buffer buf{env, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      buf = cuda::std::move(input);
      CHECK(buf.size() == 6);
      CHECK(equal_range(buf));
      CHECK(input.empty());
      CHECK(input.data() == nullptr);
    }

    { // Can be move-assigned an non-empty input growing from empty
      Buffer input{env, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      Buffer buf{env};
      buf = cuda::std::move(input);
      CHECK(buf.size() == 6);
      CHECK(equal_range(buf));
      CHECK(input.empty());
      CHECK(input.data() == nullptr);
    }
  }

  SECTION("cudax::async_buffer assignment initializer_list")
  {
    { // Can be assigned an empty initializer_list
      Buffer buf{env};
      buf = {};
      CHECK(buf.empty());
      CHECK(buf.data() == nullptr);
    }
    { // Can be assigned an empty initializer_list, shrinking
      Buffer buf{env, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      auto* old_ptr = buf.data();
      buf           = {};
      CHECK(buf.empty());
      CHECK(buf.data() == old_ptr);
    }
    { // Can be assigned a non-empty initializer_list, from empty
      Buffer buf{env};
      buf = {T(1), T(42), T(1337), T(0), T(12), T(-1)};
      CHECK(buf.size() == 6);
      CHECK(equal_range(buf));
    }

    { // Can be assigned a non-empty initializer_list, shrinking
      Buffer buf{env, 42};
      buf = {T(1), T(42), T(1337), T(0), T(12), T(-1)};
      CHECK(buf.size() == 42);
      CHECK(equal_range(buf));
    }

    { // Can be assigned a non-empty initializer_list, growing from non empty
      Buffer buf{env, {T(0), T(42)}};
      buf = {T(1), T(42), T(1337), T(0), T(12), T(-1)};
      CHECK(buf.size() == 6);
      CHECK(equal_range(buf));
    }
  }

#if 0 // Implement exceptions
#  ifndef TEST_HAS_NO_EXCEPTIONS
  SECTION("cudax::async_buffer assignment exceptions")
  { // assignment throws std::bad_alloc
    constexpr size_t capacity = 4;
    using Buffer              = cudax::async_buffer<int, capacity>;
    Buffer too_small{};

    try
    {
      cuda::std::initializer_list<int> input{0, 1, 2, 3, 4, 5, 6};
      too_small = input;
    }
    catch (const std::bad_alloc&)
    {}
    catch (...)
    {
      CHECK(false);
    }
  }
#  endif // !TEST_HAS_NO_EXCEPTIONS
#endif
}
