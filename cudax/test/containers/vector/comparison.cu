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
#include <cuda/std/initializer_list>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>

#include <cuda/experimental/vector.cuh>

#include "helper.h"
#include "types.h"
#include <catch2/catch.hpp>

// TODO: only device accessible resource
TEMPLATE_TEST_CASE("cudax::vector comparison",
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

  SECTION("cudax::vector equality")
  {
    { // without allocation
      Vector vec{resource};
      Vector other{resource, {T(0), T(1), T(2), T(3), T(4)}};

      static_assert(cuda::std::is_same<decltype(vec == vec), bool>::value, "");
      CHECK(vec == vec);

      static_assert(cuda::std::is_same<decltype(vec != other), bool>::value, "");
      CHECK(vec != other);
    }

    Vector lhs{resource, {T(1), T(42), T(1337), T(0)}};
    { // Compare equal vectors equal
      CHECK(lhs == lhs);
      CHECK(!(lhs != lhs));
    }

    { // lhs is shorter
      Vector rhs{resource, {T(1), T(42), T(1337), T(0), T(-1)}};
      CHECK(!(lhs == rhs));
      CHECK(lhs != rhs);
    }

    { // rhs is shorter
      Vector rhs{resource, {T(1), T(42), T(1337)}};
      CHECK(!(lhs == rhs));
      CHECK(lhs != rhs);
    }

    { // rhs is different
      Vector rhs{resource, {T(1), T(42), T(1337), T(-1)}};
      CHECK(!(lhs == rhs));
      CHECK(lhs != rhs);
    }
  }
}
