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

#if _CCCL_CTK_AT_LEAST(12, 6)
using test_types = c2h::type_list<cuda::std::tuple<int, cuda::mr::host_accessible>,
                                  cuda::std::tuple<unsigned long long, cuda::mr::device_accessible>,
                                  cuda::std::tuple<int, cuda::mr::host_accessible, cuda::mr::device_accessible>>;
#else // ^^^ _CCCL_CTK_AT_LEAST(12, 6) ^^^ / vvv _CCCL_CTK_BELOW(12, 6) vvv
using test_types = c2h::type_list<cuda::std::tuple<int, cuda::mr::device_accessible>>;
#endif // ^^^ _CCCL_CTK_BELOW(12, 6) ^^^

C2H_CCCLRT_TEST("cuda::buffer access and stream", "[container][buffer]", test_types)
{
  using TestT           = c2h::get<0, TestType>;
  using Resource        = typename extract_properties<TestT>::resource;
  using Buffer          = typename extract_properties<TestT>::buffer;
  using T               = typename Buffer::value_type;
  using reference       = typename Buffer::reference;
  using const_reference = typename Buffer::const_reference;
  using pointer         = typename Buffer::pointer;
  using const_pointer   = typename Buffer::const_pointer;

  cuda::stream stream{cuda::device_ref{0}};
  Resource resource = extract_properties<TestT>::get_resource();

  SECTION("cuda::buffer::get_unsynchronized")
  {
    static_assert(cuda::std::is_same_v<decltype(cuda::std::declval<Buffer&>().get_unsynchronized(1ull)), reference>);
    static_assert(
      cuda::std::is_same_v<decltype(cuda::std::declval<const Buffer&>().get_unsynchronized(1ull)), const_reference>);

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
    static_assert(cuda::std::is_same_v<decltype(cuda::std::declval<Buffer&>().data()), pointer>);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::declval<const Buffer&>().data()), const_pointer>);

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

  SECTION("cuda::buffer::memory_resource")
  {
    static_assert(noexcept(cuda::std::declval<const Buffer&>().memory_resource()));

    { // Returns the resource used during construction
      Buffer buf{stream, resource, {T(1), T(42), T(1337), T(0)}};
      const auto& mr = buf.memory_resource();
      CCCLRT_CHECK(mr == resource);
    }

    { // Works with empty buffer
      Buffer buf{stream, resource, 0, cuda::no_init};
      const auto& mr = buf.memory_resource();
      CCCLRT_CHECK(mr == resource);
    }

    { // Returns same resource after move assignment
      Buffer buf1{stream, resource, {T(1), T(42)}};

      Resource other_resource = extract_properties<TestT>::get_resource();
      Buffer buf2{stream, other_resource, {T(99), T(88)}};

      buf1 = cuda::std::move(buf2);
      // After move assignment, buf1 should have buf2's resource
      const auto& mr_after = buf1.memory_resource();
      CCCLRT_CHECK(mr_after == other_resource);
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

  SECTION("cuda::buffer::destroy")
  {
    { // destroy with explicit stream
      Buffer buf{stream, resource, {T(1), T(42), T(1337), T(0)}};
      CCCLRT_CHECK(!buf.empty());
      CCCLRT_CHECK(buf.data() != nullptr);

      cuda::stream destroy_stream{cuda::device_ref{0}};
      destroy_stream.wait(stream);
      buf.destroy(destroy_stream);
      CCCLRT_CHECK(buf.empty());
      CCCLRT_CHECK(buf.data() == nullptr);
    }

    { // destroy without explicit stream (uses stored stream)
      Buffer buf{stream, resource, {T(1), T(42), T(1337), T(0)}};
      CCCLRT_CHECK(!buf.empty());
      CCCLRT_CHECK(buf.data() != nullptr);

      buf.destroy();
      CCCLRT_CHECK(buf.empty());
      CCCLRT_CHECK(buf.data() == nullptr);
    }

    { // destroy empty buffer
      Buffer buf{stream, resource, 0, cuda::no_init};
      CCCLRT_CHECK(buf.empty());
      CCCLRT_CHECK(buf.data() == nullptr);

      buf.destroy();
      CCCLRT_CHECK(buf.empty());
      CCCLRT_CHECK(buf.data() == nullptr);
    }

    { // destroy and then move assign (should be valid)
      Buffer buf1{stream, resource, {T(1), T(42), T(1337)}};
      Buffer buf2{stream, resource, {T(99), T(88)}};

      buf1.destroy();
      CCCLRT_CHECK(buf1.empty());

      buf1 = cuda::std::move(buf2);
      CCCLRT_CHECK(buf1.size() == 2);
      CCCLRT_CHECK(buf2.empty());
    }

    { // destroy and then destroy again (should be safe)
      Buffer buf{stream, resource, {T(1), T(42)}};
      buf.destroy();
      CCCLRT_CHECK(buf.empty());

      buf.destroy();
      CCCLRT_CHECK(buf.empty());
    }
  }
}
