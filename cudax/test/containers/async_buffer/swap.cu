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

C2H_TEST("cudax::async_buffer swap",
         "[container][async_buffer]",
         c2h::type_list<cuda::std::tuple<cuda::mr::host_accessible>,
                        cuda::std::tuple<cuda::mr::device_accessible>,
                        cuda::std::tuple<cuda::mr::host_accessible, cuda::mr::device_accessible>>)
{
  using TestT     = c2h::get<0, TestType>;
  using Env       = typename extract_properties<TestT>::env;
  using Resource  = typename extract_properties<TestT>::resource;
  using Buffer    = typename extract_properties<TestT>::async_buffer;
  using T         = typename Buffer::value_type;
  using size_type = typename Buffer::size_type;

  cudax::stream stream{};
  Env env{Resource{}, stream};
  STATIC_REQUIRE(
    cuda::std::is_same_v<decltype(cuda::std::declval<Buffer&>().swap(cuda::std::declval<Buffer&>())), void>);
  STATIC_REQUIRE(
    cuda::std::is_same_v<decltype(swap(cuda::std::declval<Buffer&>(), cuda::std::declval<Buffer&>())), void>);
  STATIC_REQUIRE(noexcept(cuda::std::declval<Buffer&>().swap(cuda::std::declval<Buffer&>())));
  STATIC_REQUIRE(noexcept(swap(cuda::std::declval<Buffer&>(), cuda::std::declval<Buffer&>())));

  // Note we do not care about the elements just the sizes
  Buffer vec_small{env, 5, cudax::uninit};

  SECTION("Can swap async_buffer")
  {
    Buffer vec_large{env, 42, cudax::uninit};

    CUDAX_CHECK(vec_large.size() == 42);
    CUDAX_CHECK(vec_small.size() == 5);
    CUDAX_CHECK(vec_large.size() == 42);
    CUDAX_CHECK(vec_small.size() == 5);

    vec_large.swap(vec_small);
    CUDAX_CHECK(vec_small.size() == 42);
    CUDAX_CHECK(vec_large.size() == 5);
    CUDAX_CHECK(vec_small.size() == 42);
    CUDAX_CHECK(vec_large.size() == 5);

    swap(vec_large, vec_small);
    CUDAX_CHECK(vec_large.size() == 42);
    CUDAX_CHECK(vec_small.size() == 5);
    CUDAX_CHECK(vec_large.size() == 42);
    CUDAX_CHECK(vec_small.size() == 5);
  }

  SECTION("Can swap async_buffer without allocation")
  {
    Buffer vec_no_allocation{env, 0, cudax::uninit};

    CUDAX_CHECK(vec_no_allocation.size() == 0);
    CUDAX_CHECK(vec_small.size() == 5);
    CUDAX_CHECK(vec_no_allocation.size() == 0);
    CUDAX_CHECK(vec_small.size() == 5);

    vec_no_allocation.swap(vec_small);
    CUDAX_CHECK(vec_small.size() == 0);
    CUDAX_CHECK(vec_no_allocation.size() == 5);
    CUDAX_CHECK(vec_small.size() == 0);
    CUDAX_CHECK(vec_no_allocation.size() == 5);

    swap(vec_no_allocation, vec_small);
    CUDAX_CHECK(vec_no_allocation.size() == 0);
    CUDAX_CHECK(vec_small.size() == 5);
    CUDAX_CHECK(vec_no_allocation.size() == 0);
    CUDAX_CHECK(vec_small.size() == 5);
  }
}
