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
#include <cuda/std/utility>

#include <cuda/experimental/container.cuh>

#include "helper.h"
#include "types.h"

TEMPLATE_TEST_CASE("cudax::async_buffer capacity",
                   "[container][async_buffer]",
                   cuda::std::tuple<cuda::mr::host_accessible>,
                   cuda::std::tuple<cuda::mr::device_accessible>,
                   (cuda::std::tuple<cuda::mr::host_accessible, cuda::mr::device_accessible>) )
{
  using Env       = typename extract_properties<TestType>::env;
  using Resource  = typename extract_properties<TestType>::resource;
  using Buffer    = typename extract_properties<TestType>::async_buffer;
  using T         = typename Buffer::value_type;
  using size_type = typename Buffer::size_type;

  cudax::stream stream{};
  Env env{Resource{}, stream};

  SECTION("cudax::async_buffer::empty")
  {
    STATIC_REQUIRE(cuda::std::is_same_v<decltype(cuda::std::declval<Buffer&>().empty()), bool>);
    STATIC_REQUIRE(cuda::std::is_same_v<decltype(cuda::std::declval<const Buffer&>().empty()), bool>);
    STATIC_REQUIRE(noexcept(cuda::std::declval<Buffer&>().empty()));
    STATIC_REQUIRE(noexcept(cuda::std::declval<const Buffer&>().empty()));

    { // Works without allocation
      Buffer buf{env, 0};
      CHECK(buf.empty());
      CHECK(cuda::std::as_const(buf).empty());
    }
  }

  SECTION("cudax::async_buffer::size")
  {
    STATIC_REQUIRE(cuda::std::is_same_v<decltype(cuda::std::declval<Buffer&>().size()), size_type>);
    STATIC_REQUIRE(cuda::std::is_same_v<decltype(cuda::std::declval<const Buffer&>().size()), size_type>);
    STATIC_REQUIRE(noexcept(cuda::std::declval<Buffer&>().size()));
    STATIC_REQUIRE(noexcept(cuda::std::declval<const Buffer&>().size()));

    { // Works without allocation
      Buffer buf{env, 0};
      CHECK(buf.size() == 0);
      CHECK(cuda::std::as_const(buf).size() == 0);
    }
  }
}
