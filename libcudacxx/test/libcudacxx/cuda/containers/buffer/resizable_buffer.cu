//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/__container/resizable_buffer.h>
#include <cuda/buffer>
#include <cuda/devices>
#include <cuda/memory_pool>
#include <cuda/std/array>
#include <cuda/std/cstddef>
#include <cuda/std/span>
#include <cuda/std/string_view>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <stdexcept>

#include <cuda_runtime_api.h>

#include "helper.h"

using buffer_t = cuda::__resizable_buffer<int, cuda::mr::device_accessible>;
using base_t   = cuda::buffer<int, cuda::mr::device_accessible>;

template <typename _Iter>
__global__ void check_prefix_kernel(_Iter ptr, cuda::std::size_t size)
{
  for (cuda::std::size_t i = 0; i != size; ++i)
  {
    if (ptr[i] != device_data[i])
    {
      __trap();
    }
  }
}

template <cuda::std::size_t _Size>
void check_prefix(const buffer_t& buf, const cuda::std::array<int, _Size>&)
{
  REQUIRE(buf.size() >= _Size);

  cuda::__ensure_current_context guard{buf.stream()};
  check_prefix_kernel<<<1, 1, 0, buf.stream().get()>>>(buf.begin(), _Size);
  REQUIRE(::cudaGetLastError() == ::cudaSuccess);
  buf.stream().sync();
}

C2H_CCCLRT_TEST("cuda::__resizable_buffer capacity and resize", "[container][buffer]")
{
  static_assert(!cuda::std::is_copy_constructible_v<buffer_t>);
  static_assert(!cuda::std::is_copy_assignable_v<buffer_t>);
  static_assert(cuda::std::is_nothrow_move_constructible_v<buffer_t>);
  static_assert(cuda::std::is_constructible_v<buffer_t, base_t&&>);

  cuda::device_ref device{0};
  cuda::stream stream{device};
  auto resource = cuda::device_default_memory_pool(device);
  static_assert(cuda::std::is_constructible_v<buffer_t, cuda::stream_ref, decltype(resource)&>);

  SECTION("empty construction")
  {
    buffer_t buf{stream, resource};
    REQUIRE(buf.empty());
    REQUIRE(buf.size() == 0);
    REQUIRE(buf.capacity() == 0);
    REQUIRE(buf.capacity_bytes() == 0);
    REQUIRE(buf.data() == nullptr);
  }

  SECTION("construction sets capacity to size")
  {
    cuda::std::array<int, 6> values{1, 42, 1337, 0, 12, -1};
    buffer_t buf{stream, resource, values.begin(), values.end()};

    REQUIRE(buf.size() == values.size());
    REQUIRE(buf.capacity() == values.size());
    REQUIRE(buf.capacity_bytes() == values.size() * sizeof(int));
    REQUIRE(buf.end() == buf.begin() + buf.size());
    REQUIRE(cuda::std::span<int>{buf}.size() == buf.size());
    REQUIRE(static_cast<const base_t&>(buf).size() == buf.size());
    check_prefix(buf, values);
  }

  SECTION("construction from an existing buffer")
  {
    cuda::std::array<int, 6> values{1, 42, 1337, 0, 12, -1};
    base_t base{stream, resource, values.begin(), values.end()};
    const auto* const allocation = base.data();

    buffer_t buf{cuda::std::move(base)};

    REQUIRE(buf.size() == values.size());
    REQUIRE(buf.capacity() == values.size());
    REQUIRE(buf.data() == allocation);
    check_prefix(buf, values);
  }

  SECTION("resize without reallocation")
  {
    cuda::std::array<int, 6> values{1, 42, 1337, 0, 12, -1};
    buffer_t buf{stream, resource, values.begin(), values.end()};
    const auto* const allocation = buf.data();

    buf.resize(3, cuda::no_init);
    REQUIRE(buf.size() == 3);
    REQUIRE(buf.capacity() == values.size());
    REQUIRE(buf.data() == allocation);
    REQUIRE(buf.end() == buf.begin() + buf.size());
    REQUIRE(static_cast<const base_t&>(buf).size() == buf.size());
    check_prefix(buf, cuda::std::array<int, 3>{1, 42, 1337});

    buf.resize(5, cuda::no_init);
    REQUIRE(buf.size() == 5);
    REQUIRE(buf.capacity() == values.size());
    REQUIRE(buf.data() == allocation);
    REQUIRE(buf.end() == buf.begin() + buf.size());
    REQUIRE(static_cast<const base_t&>(buf).size() == buf.size());
    check_prefix(buf, cuda::std::array<int, 3>{1, 42, 1337});

    bool caught = false;
    try
    {
      buf.resize(7, cuda::no_init);
    }
    catch (const std::invalid_argument& error)
    {
      caught = true;
      REQUIRE(cuda::std::string_view{error.what()}
              == "cuda::__resizable_buffer::resize requires an explicit stream to grow beyond capacity");
    }
    REQUIRE(caught);
  }

  SECTION("resize with reallocation")
  {
    cuda::std::array<int, 6> values{1, 42, 1337, 0, 12, -1};
    buffer_t buf{stream, resource, values.begin(), values.end()};
    buf.resize(stream, 3, cuda::no_init);

    cuda::stream other_stream{device};
    buf.resize(other_stream, 9, cuda::no_init);

    REQUIRE(buf.size() == 9);
    REQUIRE(buf.capacity() == 9);
    REQUIRE(buf.stream() == other_stream);
    REQUIRE(buf.end() == buf.begin() + buf.size());
    REQUIRE(static_cast<const base_t&>(buf).size() == buf.size());
    check_prefix(buf, cuda::std::array<int, 3>{1, 42, 1337});

    const auto* const allocation = buf.data();
    buf.resize(other_stream, 2, cuda::no_init);
    REQUIRE(buf.size() == 2);
    REQUIRE(buf.capacity() == 9);
    REQUIRE(buf.data() == allocation);
    REQUIRE(static_cast<const base_t&>(buf).size() == buf.size());
    check_prefix(buf, cuda::std::array<int, 2>{1, 42});
  }

  SECTION("resize from empty logical size with reallocation")
  {
    buffer_t buf{stream, resource, 3, cuda::no_init};
    buf.resize(stream, 0, cuda::no_init);

    cuda::stream other_stream{device};
    buf.resize(other_stream, 9, cuda::no_init);

    REQUIRE(buf.size() == 9);
    REQUIRE(buf.capacity() == 9);
    REQUIRE(buf.stream() == other_stream);
  }

  SECTION("resize_discard")
  {
    cuda::std::array<int, 6> values{1, 42, 1337, 0, 12, -1};
    buffer_t buf{stream, resource, values.begin(), values.end()};

    cuda::stream other_stream{device};
    buf.resize_discard(other_stream, 9, cuda::no_init);

    REQUIRE(buf.size() == 9);
    REQUIRE(buf.capacity() == 9);
    REQUIRE(buf.stream() == other_stream);
    REQUIRE(static_cast<const base_t&>(buf).size() == buf.size());
  }

  SECTION("move and swap preserve logical size and capacity")
  {
    cuda::std::array<int, 6> values{1, 42, 1337, 0, 12, -1};
    buffer_t input{stream, resource, values.begin(), values.end()};
    input.resize(stream, 4, cuda::no_init);

    buffer_t moved{cuda::std::move(input)};
    REQUIRE(moved.size() == 4);
    REQUIRE(moved.capacity() == values.size());
    REQUIRE(input.size() == 0);
    REQUIRE(input.capacity() == 0);
    check_prefix(moved, cuda::std::array<int, 4>{1, 42, 1337, 0});

    buffer_t assigned{stream, resource, 8, cuda::no_init};
    assigned.resize(stream, 2, cuda::no_init);
    assigned = cuda::std::move(moved);

    REQUIRE(assigned.size() == 4);
    REQUIRE(assigned.capacity() == values.size());
    REQUIRE(moved.size() == 0);
    REQUIRE(moved.capacity() == 0);
    check_prefix(assigned, cuda::std::array<int, 4>{1, 42, 1337, 0});

    buffer_t other{stream, resource, 2, cuda::no_init};
    other.resize(stream, 1, cuda::no_init);
    swap(assigned, other);

    REQUIRE(assigned.size() == 1);
    REQUIRE(assigned.capacity() == 2);
    REQUIRE(other.size() == 4);
    REQUIRE(other.capacity() == values.size());
    check_prefix(other, cuda::std::array<int, 4>{1, 42, 1337, 0});
  }
}
