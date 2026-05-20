//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#include <cuda/buffer>
#include <cuda/memory_resource>
#include <cuda/std/algorithm>
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/initializer_list>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "helper.h"
#include "types.h"

C2H_CCCLRT_TEST("cuda::buffer swap", "[container][buffer]", test_types)
{
  using Buffer   = c2h::get<0, TestType>;
  using Resource = typename extract_properties<Buffer>::resource;

  if (!extract_properties<Buffer>::is_resource_supported())
  {
    return;
  }

  cuda::stream stream{cuda::device_ref{0}};
  Resource resource = extract_properties<Buffer>::get_resource();
  STATIC_REQUIRE(
    cuda::std::is_same_v<decltype(cuda::std::declval<Buffer&>().swap(cuda::std::declval<Buffer&>())), void>);
  STATIC_REQUIRE(
    cuda::std::is_same_v<decltype(swap(cuda::std::declval<Buffer&>(), cuda::std::declval<Buffer&>())), void>);
  STATIC_REQUIRE(noexcept(cuda::std::declval<Buffer&>().swap(cuda::std::declval<Buffer&>())));
  STATIC_REQUIRE(noexcept(swap(cuda::std::declval<Buffer&>(), cuda::std::declval<Buffer&>())));

  // Note we do not care about the elements just the sizes
  Buffer vec_small{stream, resource, 5, cuda::no_init};

  SECTION("Can swap buffer")
  {
    Buffer vec_large{stream, resource, 42, cuda::no_init};

    CCCLRT_CHECK(vec_large.size() == 42);
    CCCLRT_CHECK(vec_small.size() == 5);
    CCCLRT_CHECK(vec_large.size() == 42);
    CCCLRT_CHECK(vec_small.size() == 5);

    vec_large.swap(vec_small);
    CCCLRT_CHECK(vec_small.size() == 42);
    CCCLRT_CHECK(vec_large.size() == 5);
    CCCLRT_CHECK(vec_small.size() == 42);
    CCCLRT_CHECK(vec_large.size() == 5);

    swap(vec_large, vec_small);
    CCCLRT_CHECK(vec_large.size() == 42);
    CCCLRT_CHECK(vec_small.size() == 5);
    CCCLRT_CHECK(vec_large.size() == 42);
    CCCLRT_CHECK(vec_small.size() == 5);
  }

  SECTION("Can swap buffer without allocation")
  {
    Buffer vec_no_allocation{stream, resource, 0, cuda::no_init};

    CCCLRT_CHECK(vec_no_allocation.size() == 0);
    CCCLRT_CHECK(vec_small.size() == 5);
    CCCLRT_CHECK(vec_no_allocation.size() == 0);
    CCCLRT_CHECK(vec_small.size() == 5);

    vec_no_allocation.swap(vec_small);
    CCCLRT_CHECK(vec_small.size() == 0);
    CCCLRT_CHECK(vec_no_allocation.size() == 5);
    CCCLRT_CHECK(vec_small.size() == 0);
    CCCLRT_CHECK(vec_no_allocation.size() == 5);

    swap(vec_no_allocation, vec_small);
    CCCLRT_CHECK(vec_no_allocation.size() == 0);
    CCCLRT_CHECK(vec_small.size() == 5);
    CCCLRT_CHECK(vec_no_allocation.size() == 0);
    CCCLRT_CHECK(vec_small.size() == 5);
  }
}
