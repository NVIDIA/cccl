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

#include <cuda/experimental/container.cuh>

#include "helper.h"
#include "types.h"
#include <catch2/catch.hpp>

TEMPLATE_TEST_CASE("cudax::async_mdarray constructors",
                   "[container][async_mdarray]",
                   cuda::std::tuple<cuda::mr::host_accessible>,
                   cuda::std::tuple<cuda::mr::device_accessible>,
                   (cuda::std::tuple<cuda::mr::host_accessible, cuda::mr::device_accessible>) )
{
  using Env           = typename extract_properties<TestType>::env;
  using Resource      = typename extract_properties<TestType>::resource;
  using async_mdarray = typename extract_properties<TestType>::async_mdarray;
  using T             = typename async_mdarray::value_type;

  cudax::stream stream{};
  Env env{Resource{}, stream};

  SECTION("Construction with extent")
  {
    { // from env, no allocation
      const async_mdarray vec{env};
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
    }

    { // from env and size, no allocation
      const async_mdarray vec{env, cuda::std::dims<1>{0}};
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
    }

    { // from env, size and value, no allocation
      const async_mdarray vec{env, cuda::std::dims<1>{0}};
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
    }

    { // from env and size
      const async_mdarray vec{env, cuda::std::dims<1>{5}};
      CHECK(vec.size() == 5);
      CHECK(equal_size_value(vec, 5, T(0)));
    }

    { // from env, size and value
      const async_mdarray vec{env, cuda::std::dims<1>{5}, T{42}};
      CHECK(vec.size() == 5);
      CHECK(equal_size_value(vec, 5, T(42)));
    }
  }

  SECTION("Multidimensional construction with sizes")
  {
    using multidim_mdarray = typename change_extent<async_mdarray, cuda::std::dims<2>>::type;
    const multidim_mdarray vec{env, 2, 3};
    CHECK(vec.size() == 6);
    CHECK(equal_size_value(vec, 6, T(0)));
  }

  SECTION("Construction from range")
  {
    { // can be constructed from an empty uncommon forward range
      async_mdarray vec(env, cuda::std::dims<1>{0}, uncommon_range<T, 0>{});
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
    }

    { // can be constructed from a non-empty uncommon forward range
      async_mdarray vec(env, cuda::std::dims<1>{6}, uncommon_range<T, 6>{{T(1), T(42), T(1337), T(0), T(12), T(-1)}});
      CHECK(!vec.empty());
      CHECK(equal_range(vec));
    }

    { // can be constructed from an empty sized uncommon forward range
      async_mdarray vec(env, cuda::std::dims<1>{0}, sized_uncommon_range<T, 0>{});
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
    }

    { // can be constructed from a non-empty sized uncommon forward range
      async_mdarray vec(
        env, cuda::std::dims<1>{6}, sized_uncommon_range<T, 6>{{T(1), T(42), T(1337), T(0), T(12), T(-1)}});
      CHECK(!vec.empty());
      CHECK(equal_range(vec));
    }

    { // can be constructed from an empty random access range
      async_mdarray vec(env, cuda::std::dims<1>{0}, cuda::std::array<T, 0>{});
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
    }

    { // can be constructed from a non-empty random access range
      async_mdarray vec(env, cuda::std::dims<1>{6}, cuda::std::array<T, 6>{T(1), T(42), T(1337), T(0), T(12), T(-1)});
      CHECK(!vec.empty());
      CHECK(equal_range(vec));
    }
  }

  SECTION("copy construction")
  {
    static_assert(!cuda::std::is_nothrow_copy_constructible<async_mdarray>::value, "");
    { // can be copy constructed from empty input
      const async_mdarray input{env, cuda::std::dims<1>{0}};
      async_mdarray vec(input);
      CHECK(vec.empty());
    }

    { // can be copy constructed from non-empty input
      const async_mdarray input{env, cuda::std::dims<1>{6}, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      async_mdarray vec(input);
      CHECK(!vec.empty());
      CHECK(equal_range(vec));
    }
  }

  SECTION("move construction")
  {
    static_assert(cuda::std::is_nothrow_move_constructible<async_mdarray>::value, "");

    { // can be move constructed with empty input
      async_mdarray input{env, 0};
      async_mdarray vec(cuda::std::move(input));
      CHECK(vec.empty());
      CHECK(input.empty());
    }

    { // can be move constructed from non-empty input
      async_mdarray input{env, cuda::std::dims<1>{6}, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};

      // ensure that we steal the data
      const auto* allocation = input.data();
      async_mdarray vec(cuda::std::move(input));
      CHECK(vec.size() == 6);
      CHECK(vec.data() == allocation);
      CHECK(input.size() == 0);
      CHECK(input.data() == nullptr);
      CHECK(equal_range(vec));
    }
  }
}
