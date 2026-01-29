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

C2H_CCCLRT_TEST("cuda::buffer capacity", "[container][buffer]", test_types)
{
  using Buffer    = c2h::get<0, TestType>;
  using Resource  = typename extract_properties<Buffer>::resource;
  using size_type = typename Buffer::size_type;

  if (!extract_properties<Buffer>::is_resource_supported())
  {
    return;
  }

  cuda::stream stream{cuda::device_ref{0}};
  Resource resource = extract_properties<Buffer>::get_resource();

  SECTION("cuda::buffer::empty")
  {
    STATIC_REQUIRE(cuda::std::is_same_v<decltype(cuda::std::declval<Buffer&>().empty()), bool>);
    STATIC_REQUIRE(cuda::std::is_same_v<decltype(cuda::std::declval<const Buffer&>().empty()), bool>);
    STATIC_REQUIRE(noexcept(cuda::std::declval<Buffer&>().empty()));
    STATIC_REQUIRE(noexcept(cuda::std::declval<const Buffer&>().empty()));

    { // Works without allocation
      Buffer buf{stream, resource, 0, cuda::no_init};
      CCCLRT_CHECK(buf.empty());
      CCCLRT_CHECK(cuda::std::as_const(buf).empty());
    }
  }

  SECTION("cuda::buffer::size")
  {
    STATIC_REQUIRE(cuda::std::is_same_v<decltype(cuda::std::declval<Buffer&>().size()), size_type>);
    STATIC_REQUIRE(cuda::std::is_same_v<decltype(cuda::std::declval<const Buffer&>().size()), size_type>);
    STATIC_REQUIRE(noexcept(cuda::std::declval<Buffer&>().size()));
    STATIC_REQUIRE(noexcept(cuda::std::declval<const Buffer&>().size()));

    { // Works without allocation
      Buffer buf{stream, resource, 0, cuda::no_init};
      CCCLRT_CHECK(buf.size() == 0);
      CCCLRT_CHECK(cuda::std::as_const(buf).size() == 0);
    }
  }
}
