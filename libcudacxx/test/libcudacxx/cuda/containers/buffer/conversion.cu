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

#include <test_resources.h>

#include "helper.h"
#include "types.h"

C2H_CCCLRT_TEST("cuda::buffer conversion", "[container][buffer]", test_types)
{
  using Buffer   = c2h::get<0, TestType>;
  using Resource = typename extract_properties<Buffer>::resource;
  using T        = typename Buffer::value_type;

  // Convert from a buffer that has more properties than the current one
  using MatchingBuffer = typename extract_properties<Buffer>::matching_buffer;

  if (!extract_properties<Buffer>::is_resource_supported())
  {
    return;
  }

  cuda::stream stream{cuda::device_ref{0}};
  Resource resource = extract_properties<Buffer>::get_resource();

  SECTION("cuda::buffer construction with matching buffer")
  {
    { // can be copy constructed from empty input
      const MatchingBuffer input{stream, resource, 0, cuda::no_init};
      Buffer buf(input);
      CHECK(buf.empty());
      CHECK(input.empty());
    }

    { // can be copy constructed from non-empty input
      const MatchingBuffer input{stream, resource, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      Buffer buf(input);
      CHECK(!buf.empty());
      CHECK(equal_range(buf));
      CHECK(equal_range(input));
    }

    { // can be move constructed with empty input
      MatchingBuffer input{stream, resource, 0, cuda::no_init};
      Buffer buf(cuda::std::move(input));
      CHECK(buf.empty());
      CHECK(input.empty());
    }

    { // can be move constructed from non-empty input
      MatchingBuffer input{stream, resource, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};

      // ensure that we steal the data
      const auto* allocation = input.data();
      Buffer buf(cuda::std::move(input));
      CHECK(buf.size() == 6);
      CHECK(buf.data() == allocation);
      CHECK(input.size() == 0);
      CHECK(input.data() == nullptr);
      CHECK(equal_range(buf));
    }
  }
}
