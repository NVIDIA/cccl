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
#include <cuda/std/initializer_list>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>

#include <cuda/experimental/container.cuh>

#include "helper.h"
#include "types.h"

// TODO: only device accessible resource
C2H_TEST("cudax::async_buffer comparison",
         "[container][async_buffer]",
         c2h::type_list<cuda::std::tuple<cuda::mr::host_accessible>,
                        cuda::std::tuple<cuda::mr::device_accessible>,
                        cuda::std::tuple<cuda::mr::host_accessible, cuda::mr::device_accessible>>)
{
  using TestT    = c2h::get<0, TestType>;
  using Env      = typename extract_properties<TestT>::env;
  using Resource = typename extract_properties<TestT>::resource;
  using Buffer   = typename extract_properties<TestT>::async_buffer;
  using T        = typename Buffer::value_type;

  cudax::stream stream{};
  Env env{Resource{}, stream};

  SECTION("cudax::async_buffer equality")
  {
    { // without allocation
      Buffer buf{env};
      Buffer other{env, {T(0), T(1), T(2), T(3), T(4)}};

      static_assert(cuda::std::is_same<decltype(buf == buf), bool>::value, "");
      CUDAX_CHECK(buf == buf);

      static_assert(cuda::std::is_same<decltype(buf != other), bool>::value, "");
      CUDAX_CHECK(buf != other);
    }

    Buffer lhs{env, {T(1), T(42), T(1337), T(0)}};
    { // Compare equal vectors equal
      CUDAX_CHECK(lhs == lhs);
      CUDAX_CHECK(!(lhs != lhs));
    }

    { // lhs is shorter
      Buffer rhs{env, {T(1), T(42), T(1337), T(0), T(-1)}};
      CUDAX_CHECK(!(lhs == rhs));
      CUDAX_CHECK(lhs != rhs);
    }

    { // rhs is shorter
      Buffer rhs{env, {T(1), T(42), T(1337)}};
      CUDAX_CHECK(!(lhs == rhs));
      CUDAX_CHECK(lhs != rhs);
    }

    { // rhs is different
      Buffer rhs{env, {T(1), T(42), T(1337), T(-1)}};
      CUDAX_CHECK(!(lhs == rhs));
      CUDAX_CHECK(lhs != rhs);
    }
  }
}
