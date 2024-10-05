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

#include <cuda/experimental/vector.cuh>

#include "helper.h"
#include "types.h"
#include <catch2/catch.hpp>

// TODO: only device accessible resource
TEMPLATE_TEST_CASE("cudax::vector assignment",
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

  using MatchingResource     = typename extract_properties<TestType>::matching_resource;
  using MatchingResource_Ref = typename extract_properties<TestType>::matching_resource_ref;
  MatchingResource raw_matching_resource{raw_resource};
  MatchingResource_Ref matching_resource{raw_matching_resource};

  SECTION("cudax::vector copy-assignment")
  {
    { // Can be copy-assigned an empty input
      const Vector input{resource};
      Vector vec{resource};
      vec = input;
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
    }
    { // Can be copy-assigned an empty input, shrinking
      const Vector input{resource};
      Vector vec{resource, 4};
      vec = input;
      CHECK(vec.empty());
      CHECK(vec.data() != nullptr);
    }

    { // Can be copy-assigned a non-empty input, shrinking
      const Vector input{resource, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      Vector vec{resource, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      vec = input;
      CHECK(!vec.empty());
      CHECK(equal_range(vec));
    }

    { // Can be copy-assigned an non-empty input growing from empty no reallocation
      const Vector input{resource, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      Vector vec{resource, 42};
      vec.clear();
      vec = input;
      CHECK(vec.capacity() == 42);
      CHECK(equal_range(vec));
    }

    { // Can be copy-assigned an non-empty input growing from non-empty no reallocation
      const Vector input{resource, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      Vector vec{resource, 42};
      vec.resize(2);
      vec = input;
      CHECK(vec.capacity() == 42);
      CHECK(equal_range(vec));
    }

    { // Can be copy-assigned a non-empty input, growing from empty with reallocation
      const Vector input{resource, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      Vector vec{resource};
      vec = input;
      CHECK(vec.capacity() == 6);
      CHECK(equal_range(vec));
    }

    { // Can be copy-assigned a non-empty input, growing with reallocation
      const Vector input{resource, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      Vector vec{resource, 2};
      vec = input;
      CHECK(vec.capacity() == 6);
      CHECK(equal_range(vec));
    }
  }

  SECTION("cudax::vector copy-assignment different resource")
  {
    { // Can be copy-assigned an empty input
      const Vector input{matching_resource};
      Vector vec{resource};
      vec = input;
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
    }

    { // Can be copy-assigned an empty input, shrinking
      const Vector input{matching_resource};
      Vector vec{resource, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      vec = input;
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
    }

    { // Can be copy-assigned a non-empty input, shrinking
      const Vector input{matching_resource, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      Vector vec{resource, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      vec = input;
      CHECK(vec.capacity() == 6);
      CHECK(equal_range(vec));
    }

    { // Can be copy-assigned an non-empty input growing from empty without capacity
      const Vector input{matching_resource, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      Vector vec{resource};
      vec = input;
      CHECK(vec.capacity() == 6);
      CHECK(equal_range(vec));
    }

    { // Can be copy-assigned an non-empty input growing from empty with capacity
      const Vector input{matching_resource, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      Vector vec{resource, 42};
      vec.clear();
      vec = input;
      CHECK(vec.capacity() == 6);
      CHECK(equal_range(vec));
    }

    { // Can be copy-assigned an non-empty input growing from non-empty
      const Vector input{matching_resource, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      Vector vec{resource, {T(0), T(42)}};
      vec.resize(2);
      vec = input;
      CHECK(vec.capacity() == 6);
      CHECK(equal_range(vec));
    }
  }

  SECTION("cudax::vector move-assignment")
  {
    { // Can be move-assigned an empty input
      Vector input{resource};
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
      Vector input{resource};
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
      Vector input{resource, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      Vector vec{resource, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      vec = cuda::std::move(input);
      CHECK(vec.capacity() == 6);
      CHECK(equal_range(vec));
      CHECK(input.empty());
      CHECK(input.data() == nullptr);
    }

    { // Can be move-assigned an non-empty input growing from empty
      Vector input{resource, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      Vector vec{resource};
      vec = cuda::std::move(input);
      CHECK(vec.capacity() == 6);
      CHECK(equal_range(vec));
      CHECK(input.empty());
      CHECK(input.data() == nullptr);
    }

    { // Can be move-assigned an non-empty input growing from non-empty
      Vector input{resource, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      Vector vec{resource, 42};
      vec.resize(2);
      vec = cuda::std::move(input);
      CHECK(vec.capacity() == 6);
      CHECK(equal_range(vec));
      CHECK(input.empty());
      CHECK(input.data() == nullptr);
    }
  }

  SECTION("cudax::vector assignment initializer_list")
  {
    { // Can be assigned an empty initializer_list
      Vector vec{resource};
      vec = {};
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
    }
    { // Can be assigned an empty initializer_list, shrinking
      Vector vec{resource, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      auto* old_ptr = vec.data();
      vec           = {};
      CHECK(vec.empty());
      CHECK(vec.data() == old_ptr);
    }
    { // Can be assigned a non-empty initializer_list, from empty
      Vector vec{resource};
      vec = {T(1), T(42), T(1337), T(0), T(12), T(-1)};
      CHECK(vec.capacity() == 6);
      CHECK(equal_range(vec));
    }

    { // Can be assigned a non-empty initializer_list, shrinking
      Vector vec{resource, 42};
      vec = {T(1), T(42), T(1337), T(0), T(12), T(-1)};
      CHECK(vec.capacity() == 42);
      CHECK(equal_range(vec));
    }

    { // Can be assigned a non-empty initializer_list, growing from non empty
      Vector vec{resource, {T(0), T(42)}};
      vec = {T(1), T(42), T(1337), T(0), T(12), T(-1)};
      CHECK(vec.capacity() == 6);
      CHECK(equal_range(vec));
    }
  }

#if 0 // Implement exceptions
#  ifndef TEST_HAS_NO_EXCEPTIONS
  SECTION("cudax::vector assignment exceptions")
  { // assignment throws std::bad_alloc
    constexpr size_t capacity = 4;
    using Vector              = cudax::vector<int, capacity>;
    Vector too_small{};

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
