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

#include <cuda/experimental/vector.cuh>

#include "helper.h"
#include "test_resources.h"
#include "types.h"
#include <catch2/catch.hpp>

TEMPLATE_TEST_CASE("cudax::vector conversion",
                   "[container][vector]",
                   cuda::std::tuple<cuda::mr::host_accessible>,
                   cuda::std::tuple<cuda::mr::device_accessible>,
                   (cuda::std::tuple<cuda::mr::host_accessible, cuda::mr::device_accessible>) )
{
  using Resource     = typename extract_properties<TestType>::resource;
  using Resource_ref = typename extract_properties<TestType>::resource_ref;
  using Vector       = typename extract_properties<TestType>::vector;
  using T            = typename Vector::value_type;

  Resource raw_resource{};
  Resource_ref resource{raw_resource};

  // Convert from a vector that has more properties than the current one
  using MatchingVector       = typename extract_properties<TestType>::matching_vector;
  using MatchingResource     = typename extract_properties<TestType>::matching_resource;
  using MatchingResource_Ref = typename extract_properties<TestType>::matching_resource_ref;
  MatchingResource raw_matching_resource{raw_resource};
  MatchingResource_Ref matching_resource{raw_matching_resource};

  SECTION("cudax::vector construction with matching vector")
  {
    { // can be copy constructed from empty input
      const MatchingVector input{matching_resource, 0};
      Vector vec(input);
      CHECK(vec.empty());
      CHECK(input.empty());
    }

    { // can be copy constructed from non-empty input
      const MatchingVector input{matching_resource, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      Vector vec(input);
      CHECK(!vec.empty());
      CHECK(equal_range(vec));
      CHECK(equal_range(input));
    }

    { // can be move constructed with empty input
      MatchingVector input{matching_resource, 0};
      Vector vec(cuda::std::move(input));
      CHECK(vec.empty());
      CHECK(input.empty());
    }

    { // can be move constructed from non-empty input
      MatchingVector input{matching_resource, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};

      // ensure that we steal the data
      const auto* allocation = input.data();
      Vector vec(cuda::std::move(input));
      CHECK(vec.capacity() == 6);
      CHECK(vec.data() == allocation);
      CHECK(input.capacity() == 0);
      CHECK(input.data() == nullptr);
      CHECK(equal_range(vec));
    }
  }

  SECTION("cudax::vector copy assignment of matching vector")
  {
    { // Can be assigned an empty input, no allocation
      const MatchingVector input{matching_resource};
      Vector vec{resource};
      vec = input;
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
    }

    { // Can be assigned an empty input, shrinking
      const MatchingVector input{matching_resource};
      Vector vec{resource, 4, T(-2)};
      vec = input;
      CHECK(vec.empty());
      CHECK(vec.data() != nullptr);
    }

    { // Can be assigned a non-empty input, shrinking
      const MatchingVector input{matching_resource, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      Vector vec{resource, 42, T(-2)};
      vec = input;
      CHECK(!vec.empty());
      CHECK(equal_range(vec));
    }

    { // Can be assigned an non-empty input growing from empty no reallocation
      const MatchingVector input{matching_resource, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      Vector vec{resource, 42, T(-2)};
      vec.clear();
      vec = input;
      CHECK(vec.capacity() == 42);
      CHECK(equal_range(vec));
    }

    { // Can be assigned an non-empty input growing from non-empty no reallocation
      const MatchingVector input{matching_resource, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      Vector vec{resource, 42, T(-2)};
      vec.resize(2);
      vec = input;
      CHECK(vec.capacity() == 42);
      CHECK(equal_range(vec));
    }

    { // Can be assigned a non-empty input, growing from empty with reallocation
      const MatchingVector input{matching_resource, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      Vector vec{resource};
      vec = input;
      CHECK(vec.capacity() == 6);
      CHECK(equal_range(vec));
    }

    { // Can be assigned a non-empty input, growing with reallocation
      const MatchingVector input{matching_resource, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      Vector vec{resource, 4, T(-2)};
      vec = input;
      CHECK(vec.capacity() == 6);
      CHECK(equal_range(vec));
    }
  }

  SECTION("cudax::vector move-assignment matching vector")
  {
    { // Can be move-assigned an empty input
      MatchingVector input{matching_resource};
      CHECK(input.empty());
      CHECK(input.data() == nullptr);

      Vector vec{resource};
      vec = cuda::std::move(input);
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
      CHECK(input.empty());
      CHECK(input.data() == nullptr);
    }

    { // Can be move-assigned an empty input, shrinking
      MatchingVector input{matching_resource};
      CHECK(input.empty());
      CHECK(input.data() == nullptr);

      Vector vec{resource, 4};
      vec = cuda::std::move(input);
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
      CHECK(input.empty());
      CHECK(input.data() == nullptr);
    }

    { // Can be move-assigned a non-empty input, shrinking
      MatchingVector input{matching_resource, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      Vector vec{resource, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      vec = cuda::std::move(input);
      CHECK(vec.capacity() == 6);
      CHECK(equal_range(vec));
      CHECK(input.empty());
      CHECK(input.data() == nullptr);
    }

    { // Can be move-assigned an non-empty input growing from empty
      MatchingVector input{matching_resource, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      Vector vec{resource};
      vec = cuda::std::move(input);
      CHECK(vec.capacity() == 6);
      CHECK(equal_range(vec));
      CHECK(input.empty());
      CHECK(input.data() == nullptr);
    }

    { // Can be move-assigned an non-empty input growing from non-empty
      MatchingVector input{matching_resource, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      Vector vec{resource, 42};
      vec.resize(2);
      vec = cuda::std::move(input);
      CHECK(vec.capacity() == 6);
      CHECK(equal_range(vec));
      CHECK(input.empty());
      CHECK(input.data() == nullptr);
    }
  }
}
