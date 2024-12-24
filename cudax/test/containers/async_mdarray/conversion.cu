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
#include <cuda/std/mdspan>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>

#include <cuda/experimental/container.cuh>

#include "helper.h"
#include "test_resources.h"
#include "types.h"
#include <catch2/catch.hpp>

template <class>
void print() = delete;

TEMPLATE_TEST_CASE("cudax::async_mdarray conversion",
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

  // Convert from a async_mdarray that has more properties than the current one
  using MatchingArray    = typename extract_properties<TestType>::matching_vector;
  using MatchingResource = typename extract_properties<TestType>::matching_resource;
  Env matching_env{MatchingResource{resource}, stream};

  SECTION("cudax::async_mdarray construction with matching async_mdarray")
  {
    { // can be copy constructed from empty input
      const MatchingArray input{matching_env};
      Array vec(input);
      CHECK(vec.empty());
      CHECK(input.empty());
    }

    { // can be copy constructed from non-empty input
      const MatchingArray input{matching_env, cuda::std::dims<1>{6}, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      Array vec(input);
      CHECK(!vec.empty());
      CHECK(equal_range(vec));
      CHECK(equal_range(input));
    }

    { // can be move constructed with empty input
      MatchingArray input{matching_env};
      Array vec(cuda::std::move(input));
      CHECK(vec.empty());
      CHECK(input.empty());
    }

    { // can be move constructed from non-empty input
      MatchingArray input{matching_env, cuda::std::dims<1>{6}, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};

      // ensure that we steal the data
      const auto* allocation = input.data();
      Array vec(cuda::std::move(input));
      CHECK(vec.size() == 6);
      CHECK(vec.data() == allocation);
      CHECK(input.size() == 0);
      CHECK(input.data() == nullptr);
      CHECK(equal_range(vec));
    }
  }

  SECTION("cudax::async_mdarray copy assignment of matching async_mdarray")
  {
    { // Can be assigned an empty input, no allocation
      const MatchingArray input{matching_env};
      Array vec{env};
      vec = input;
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
    }

    { // Can be assigned an empty input, shrinking
      const MatchingArray input{matching_env};
      Array vec{env, cuda::std::dims<1>{42}, T(-2)};
      vec = input;
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
    }

    { // Can be assigned a non-empty input, shrinking
      const MatchingArray input{matching_env, cuda::std::dims<1>{6}, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      Array vec{env, cuda::std::dims<1>{42}, T(-2)};
      vec = input;
      CHECK(!vec.empty());
      CHECK(equal_range(vec));
    }

    { // Can be assigned a non-empty input, growing from empty with reallocation
      const MatchingArray input{matching_env, cuda::std::dims<1>{6}, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      Array vec{env};
      vec = input;
      CHECK(vec.size() == 6);
      CHECK(equal_range(vec));
    }

    { // Can be assigned a non-empty input, growing with reallocation
      const MatchingArray input{matching_env, cuda::std::dims<1>{6}, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      Array vec{env, cuda::std::dims<1>{42}, T(-2)};
      vec = input;
      CHECK(vec.size() == 6);
      CHECK(equal_range(vec));
    }
  }

  SECTION("cudax::async_mdarray move-assignment matching async_mdarray")
  {
    { // Can be move-assigned an empty input
      MatchingArray input{matching_env};
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
      MatchingArray input{matching_env};
      CHECK(input.empty());
      CHECK(input.data() == nullptr);

      Array vec{env, 4};
      vec = cuda::std::move(input);
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
      CHECK(input.empty());
      CHECK(input.data() == nullptr);
    }

    { // Can be move-assigned a non-empty input, shrinking
      MatchingArray input{matching_env, cuda::std::dims<1>{6}, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      Array vec{env, cuda::std::dims<1>{42}, T(-2)};
      vec = cuda::std::move(input);
      CHECK(vec.size() == 6);
      CHECK(equal_range(vec));
      CHECK(input.empty());
      CHECK(input.data() == nullptr);
    }

    { // Can be move-assigned an non-empty input growing from empty
      MatchingArray input{matching_env, cuda::std::dims<1>{6}, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      Array vec{env};
      vec = cuda::std::move(input);
      CHECK(vec.size() == 6);
      CHECK(equal_range(vec));
      CHECK(input.empty());
      CHECK(input.data() == nullptr);
    }
  }

  SECTION("Conversion to mdspan")
  {
    Array vec{env, cuda::std::dims<1>{42}, T{42}};
    using layout         = cuda::std::layout_right;
    using accessor       = cuda::std::default_accessor<int>;
    using const_accessor = cuda::std::default_accessor<const int>;

    { // conversion with same template arguments
      const cuda::std::mdspan<int, cuda::std::dims<1>, layout, accessor> mdspan = vec;
      CHECK(vec.data() == mdspan.data_handle());
      CHECK(vec.mapping() == mdspan.mapping());
      CHECK(vec.extents() == mdspan.extents());
    }

    { // Explicit call to view with same properties and no argument
      using mdspan_t                 = cuda::std::mdspan<int, cuda::std::dims<1>, layout, accessor>;
      const cuda::std::mdspan mdspan = vec.view();
      STATIC_REQUIRE(cuda::std::is_same_v<decltype(mdspan), const mdspan_t>);
      CHECK(vec.data() == mdspan.data_handle());
      CHECK(vec.mapping() == mdspan.mapping());
      CHECK(vec.extents() == mdspan.extents());

      using const_mdspan_t                 = cuda::std::mdspan<const int, cuda::std::dims<1>, layout, const_accessor>;
      const cuda::std::mdspan const_mdspan = cuda::std::as_const(vec).view();
      STATIC_REQUIRE(cuda::std::is_same_v<decltype(const_mdspan), const const_mdspan_t>);
      CHECK(vec.data() == const_mdspan.data_handle());
      CHECK(vec.mapping() == const_mdspan.mapping());
      CHECK(vec.extents() == const_mdspan.extents());
    }

    { // Explicit call to view with same properties and accessor
      using mdspan_t    = cuda::std::mdspan<int, cuda::std::dims<1>, layout, accessor>;
      const auto mdspan = vec.view(accessor{});
      STATIC_REQUIRE(cuda::std::is_same_v<decltype(mdspan), const mdspan_t>);
      CHECK(vec.data() == mdspan.data_handle());
      CHECK(vec.mapping() == mdspan.mapping());
      CHECK(vec.extents() == mdspan.extents());

      using const_mdspan_t    = cuda::std::mdspan<const int, cuda::std::dims<1>, layout, const_accessor>;
      const auto const_mdspan = cuda::std::as_const(vec).view(const_accessor{});
      STATIC_REQUIRE(cuda::std::is_same_v<decltype(const_mdspan), const const_mdspan_t>);
      CHECK(vec.data() == const_mdspan.data_handle());
      CHECK(vec.mapping() == const_mdspan.mapping());
      CHECK(vec.extents() == const_mdspan.extents());
    }
  }

  SECTION("Conversion to mdspan with different accessor")
  {
    Array vec{env, cuda::std::dims<1>{42}, T{42}};
    using layout = cuda::std::layout_right;
    struct accessor : cuda::std::default_accessor<int>
    {};
    struct const_accessor : cuda::std::default_accessor<const int>
    {};

    { // conversion
      const cuda::std::mdspan<int, cuda::std::dims<1>, layout, accessor> mdspan = vec;
      CHECK(vec.data() == mdspan.data_handle());
      CHECK(vec.mapping() == mdspan.mapping());
      CHECK(vec.extents() == mdspan.extents());
    }

    { // Explicit call to view with same properties and accessor
      using mdspan_t    = cuda::std::mdspan<int, cuda::std::dims<1>, layout, accessor>;
      const auto mdspan = vec.view(accessor{});
      STATIC_REQUIRE(cuda::std::is_same_v<decltype(mdspan), const mdspan_t>);
      CHECK(vec.data() == mdspan.data_handle());
      CHECK(vec.mapping() == mdspan.mapping());
      CHECK(vec.extents() == mdspan.extents());

      using const_mdspan_t    = cuda::std::mdspan<const int, cuda::std::dims<1>, layout, const_accessor>;
      const auto const_mdspan = cuda::std::as_const(vec).view(const_accessor{});
      STATIC_REQUIRE(cuda::std::is_same_v<decltype(const_mdspan), const const_mdspan_t>);
      CHECK(vec.data() == const_mdspan.data_handle());
      CHECK(vec.mapping() == const_mdspan.mapping());
      CHECK(vec.extents() == const_mdspan.extents());
    }
  }

  SECTION("Conversion to mdspan with different layout")
  {
    Array vec{env, cuda::std::dims<1>{42}, T{42}};
    using layout  = cuda::std::layout_left;
    using mapping = typename layout::mapping<cuda::std::dims<1>>;
    struct accessor : cuda::std::default_accessor<int>
    {};
    struct const_accessor : cuda::std::default_accessor<const int>
    {};

    { // conversion
      const cuda::std::mdspan<int, cuda::std::dims<1>, layout, accessor> mdspan = vec;
      CHECK(vec.data() == mdspan.data_handle());
      CHECK(vec.extents() == mdspan.extents());
      STATIC_REQUIRE(cuda::std::is_same_v<decltype(mdspan.mapping()), const mapping&>);
    }
  }
}
