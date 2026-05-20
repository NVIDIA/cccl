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
#include <cuda/devices>
#include <cuda/memory_resource>
#include <cuda/std/algorithm>
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/initializer_list>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>

#include <stdexcept>

#include "helper.h"
#include "types.h"

C2H_CCCLRT_TEST("cuda::buffer access and stream", "[container][buffer]", test_types)
{
  using Buffer   = c2h::get<0, TestType>;
  using Resource = typename extract_properties<Buffer>::resource;
  using T        = typename Buffer::value_type;

  if (!extract_properties<Buffer>::is_resource_supported())
  {
    return;
  }

  cuda::stream stream{cuda::device_ref{0}};
  Resource resource = extract_properties<Buffer>::get_resource();

  SECTION("cuda::buffer::get_unsynchronized")
  {
    static_assert(cuda::std::is_same_v<decltype(cuda::std::declval<Buffer&>().get_unsynchronized(1ull)),
                                       typename Buffer::reference>);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::declval<const Buffer&>().get_unsynchronized(1ull)),
                                       typename Buffer::const_reference>);

    {
      Buffer buf{stream, resource, {T(1), T(42), T(1337), T(0)}};
      buf.stream().sync();
      auto& res = buf.get_unsynchronized(2);
      CCCLRT_CHECK(compare_value<Buffer>(res, T(1337)));
      CCCLRT_CHECK(static_cast<size_t>(cuda::std::addressof(res) - buf.data()) == 2);
      assign_value<Buffer>(res, T(4));

      auto& const_res = cuda::std::as_const(buf).get_unsynchronized(2);
      CCCLRT_CHECK(compare_value<Buffer>(const_res, T(4)));
      CCCLRT_CHECK(static_cast<size_t>(cuda::std::addressof(const_res) - buf.data()) == 2);
    }
  }

  SECTION("cuda::buffer::data")
  {
    static_assert(cuda::std::is_same_v<decltype(cuda::std::declval<Buffer&>().data()), typename Buffer::pointer>);
    static_assert(
      cuda::std::is_same_v<decltype(cuda::std::declval<const Buffer&>().data()), typename Buffer::const_pointer>);

    { // Works without allocation
      Buffer buf{stream, resource};
      buf.stream().sync();
      CCCLRT_CHECK(buf.data() == nullptr);
      CCCLRT_CHECK(cuda::std::as_const(buf).data() == nullptr);
    }

    { // Works with allocation
      Buffer buf{stream, resource, {T(1), T(42), T(1337), T(0)}};
      buf.stream().sync();
      CCCLRT_CHECK(buf.data() != nullptr);
      CCCLRT_CHECK(cuda::std::as_const(buf).data() != nullptr);
      CCCLRT_CHECK(cuda::std::as_const(buf).data() == buf.data());
    }
  }

  SECTION("cuda::buffer::stream")
  {
    Buffer buf{stream, resource, {T(1), T(42), T(1337), T(0)}};
    CCCLRT_CHECK(buf.stream() == stream);

    {
      cuda::stream other_stream{cuda::device_ref{0}};
      buf.set_stream(other_stream);
      CCCLRT_CHECK(buf.stream() == other_stream);
      buf.set_stream(stream);
    }

    CCCLRT_CHECK(buf.stream() == stream);
    buf.destroy(stream);
  }
}
