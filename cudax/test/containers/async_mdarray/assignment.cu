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
TEMPLATE_TEST_CASE("cudax::async_mdarray assignment",
                   "[container][async_mdarray]",
                   cuda::std::tuple<cuda::mr::host_accessible>,
                   cuda::std::tuple<cuda::mr::device_accessible>,
                   (cuda::std::tuple<cuda::mr::host_accessible, cuda::mr::device_accessible>) )
{
  using Env      = typename extract_properties<TestType>::env;
  using Resource = typename extract_properties<TestType>::resource;
  using Array    = typename extract_properties<TestType>::async_mdarray;
  using T        = typename Array::value_type;

  cudax::stream stream{};
  Resource resource{};
  Env env{resource, stream};

  using MatchingResource = typename extract_properties<TestType>::matching_resource;
  Env matching_env{MatchingResource{resource}, stream};

  SECTION("cudax::async_mdarray copy-assignment")
  {
    { // Can be copy-assigned an empty input
      const Array input{env};
      Array vec{env};
      vec = input;
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
    }
    { // Can be copy-assigned an empty input, shrinking
      const Array input{env};
      CHECK(input.empty());
      Array vec{env, cuda::std::dims<1>{42}};
      vec = input;
      CHECK(vec.empty());
    }

    { // Can be copy-assigned a non-empty input, shrinking
      const Array input{env, cuda::std::dims<1>{6}, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      Array vec{env, cuda::std::dims<1>{42}};
      vec = input;
      CHECK(!vec.empty());
      CHECK(equal_range(vec));
    }

    { // Can be copy-assigned a non-empty input, growing from empty with reallocation
      const Array input{env, cuda::std::dims<1>{6}, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      Array vec{env};
      vec = input;
      CHECK(vec.size() == 6);
      CHECK(equal_range(vec));
    }

    { // Can be copy-assigned a non-empty input, growing with reallocation
      const Array input{env, cuda::std::dims<1>{6}, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      Array vec{env, cuda::std::dims<1>{2}};
      vec = input;
      CHECK(vec.size() == 6);
      CHECK(equal_range(vec));
    }
  }

  SECTION("cudax::async_mdarray copy-assignment different resource")
  {
    { // Can be copy-assigned an empty input
      const Array input{matching_env};
      Array vec{env};
      vec = input;
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
    }

    { // Can be copy-assigned an empty input, shrinking
      const Array input{matching_env};
      Array vec{env, cuda::std::dims<1>{42}};
      vec = input;
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
    }

    { // Can be copy-assigned a non-empty input, shrinking
      const Array input{matching_env, cuda::std::dims<1>{6}, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      Array vec{env, cuda::std::dims<1>{42}};
      vec = input;
      CHECK(vec.size() == 6);
      CHECK(equal_range(vec));
    }

    { // Can be copy-assigned an non-empty input growing from empty without size
      const Array input{matching_env, cuda::std::dims<1>{6}, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      Array vec{env};
      vec = input;
      CHECK(vec.size() == 6);
      CHECK(equal_range(vec));
    }
  }

  SECTION("cudax::async_mdarray move-assignment")
  {
    { // Can be move-assigned an empty input
      Array input{env};
      CHECK(input.empty());
      CHECK(input.data() == nullptr);

      Array vec{env};
      vec = cuda::std::move(input);
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
      CHECK(input.empty());
      CHECK(input.data() == nullptr);
    }

    { // Can be move-assigned an empty input, shrinking
      Array input{env};
      CHECK(input.empty());
      CHECK(input.data() == nullptr);

      Array vec{env, cuda::std::dims<1>{42}};
      vec = cuda::std::move(input);
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
      CHECK(input.empty());
      CHECK(input.data() == nullptr);
    }

    { // Can be move-assigned a non-empty input, shrinking
      Array input{env, cuda::std::dims<1>{6}, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      Array vec{env, cuda::std::dims<1>{42}};
      vec = cuda::std::move(input);
      CHECK(vec.size() == 6);
      CHECK(equal_range(vec));
      CHECK(input.empty());
      CHECK(input.data() == nullptr);
    }

    { // Can be move-assigned an non-empty input growing from empty
      Array input{env, cuda::std::dims<1>{6}, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      Array vec{env};
      vec = cuda::std::move(input);
      CHECK(vec.size() == 6);
      CHECK(equal_range(vec));
      CHECK(input.empty());
      CHECK(input.data() == nullptr);
    }
  }

  SECTION("cudax::async_mdarray assignment initializer_list")
  {
    { // Can be assigned an empty initializer_list
      Array vec{env};
      vec = {};
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
    }

    { // Can be assigned an empty initializer_list, shrinking
      Array vec{env, cuda::std::dims<1>{6}, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      auto* old_ptr = vec.data();
      vec           = {};
      CHECK(vec.empty());
    }

    { // Can be assigned a non-empty initializer_list, from empty
      Array vec{env};
      vec = {T(1), T(42), T(1337), T(0), T(12), T(-1)};
      CHECK(vec.size() == 6);
      CHECK(equal_range(vec));
    }

    { // Can be assigned a non-empty initializer_list, shrinking
      Array vec{env, cuda::std::dims<1>{42}};
      vec = {T(1), T(42), T(1337), T(0), T(12), T(-1)};
      CHECK(vec.size() == 6);
      CHECK(equal_range(vec));
    }

    { // Can be assigned a non-empty initializer_list, growing from non empty
      Array vec{env, cuda::std::dims<1>{2}};
      vec = {T(1), T(42), T(1337), T(0), T(12), T(-1)};
      CHECK(vec.size() == 6);
      CHECK(equal_range(vec));
    }
  }
}
