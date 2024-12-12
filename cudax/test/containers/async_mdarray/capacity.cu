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

TEMPLATE_TEST_CASE("cudax::async_mdarray size",
                   "[container][async_mdarray]",
                   cuda::std::tuple<cuda::mr::host_accessible>,
                   cuda::std::tuple<cuda::mr::device_accessible>,
                   (cuda::std::tuple<cuda::mr::host_accessible, cuda::mr::device_accessible>) )
{
  using Env       = typename extract_properties<TestType>::env;
  using Resource  = typename extract_properties<TestType>::resource;
  using Array     = typename extract_properties<TestType>::async_mdarray;
  using T         = typename Array::value_type;
  using size_type = typename Array::size_type;

  cudax::stream stream{};
  Env env{Resource{}, stream};

  SECTION("cudax::async_mdarray::empty")
  {
    STATIC_REQUIRE(cuda::std::is_same_v<decltype(cuda::std::declval<Array&>().empty()), bool>);
    STATIC_REQUIRE(cuda::std::is_same_v<decltype(cuda::std::declval<const Array&>().empty()), bool>);
    STATIC_REQUIRE(noexcept(cuda::std::declval<Array&>().empty()));
    STATIC_REQUIRE(noexcept(cuda::std::declval<const Array&>().empty()));

    { // Works without allocation
      Array vec{env};
      CHECK(vec.empty());
      CHECK(cuda::std::as_const(vec).empty());
    }

    { // Works with allocation
      Array vec{env, cuda::std::dims<1>{42}, cudax::uninit}; // Note we do not care about the elements just the sizes
      CHECK(!vec.empty());
      CHECK(!cuda::std::as_const(vec).empty());
    }
  }

  SECTION("cudax::async_mdarray::size")
  {
    STATIC_REQUIRE(cuda::std::is_same_v<decltype(cuda::std::declval<Array&>().size()), size_type>);
    STATIC_REQUIRE(cuda::std::is_same_v<decltype(cuda::std::declval<const Array&>().size()), size_type>);
    STATIC_REQUIRE(noexcept(cuda::std::declval<Array&>().size()));
    STATIC_REQUIRE(noexcept(cuda::std::declval<const Array&>().size()));

    { // Works without allocation
      Array vec{env};
      CHECK(vec.size() == 0);
      CHECK(cuda::std::as_const(vec).size() == 0);
    }

    { // Works with allocation
      Array vec{env, cuda::std::dims<1>{42}, cudax::uninit}; // Note we do not care about the elements just the sizes
      CHECK(vec.size() == 42);
      CHECK(cuda::std::as_const(vec).size() == 42);
    }
  }
}
