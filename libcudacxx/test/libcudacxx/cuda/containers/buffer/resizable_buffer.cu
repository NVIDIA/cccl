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
#include <cuda/std/algorithm>
#include <cuda/std/array>
#include <cuda/std/cstddef>
#include <cuda/std/span>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <cuda_runtime_api.h>

#include <stdexcept>

#include "helper.h"

using buffer_t = cuda::__resizable_buffer<int, cuda::mr::device_accessible>;
using base_t   = cuda::buffer<int, cuda::mr::device_accessible>;

template <cuda::std::size_t _Size>
void check_prefix(const buffer_t& buf, const cuda::std::array<int, _Size>& expected)
{
  CCCLRT_REQUIRE(buf.size() >= expected.size());

  cuda::std::array<int, _Size> actual{};
  buf.stream().sync();
  CCCLRT_REQUIRE(
    ::cudaMemcpy(actual.data(), buf.data(), sizeof(int) * expected.size(), ::cudaMemcpyDefault) == ::cudaSuccess);
  CCCLRT_CHECK(cuda::std::equal(actual.begin(), actual.end(), expected.begin(), expected.end()));
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
    CCCLRT_CHECK(buf.empty());
    CCCLRT_CHECK(buf.size() == 0);
    CCCLRT_CHECK(buf.capacity() == 0);
    CCCLRT_CHECK(buf.capacity_bytes() == 0);
    CCCLRT_CHECK(buf.data() == nullptr);
  }

  SECTION("construction sets capacity to size")
  {
    cuda::std::array<int, 6> values{1, 42, 1337, 0, 12, -1};
    buffer_t buf{stream, resource, values.begin(), values.end()};

    CCCLRT_CHECK(buf.size() == values.size());
    CCCLRT_CHECK(buf.capacity() == values.size());
    CCCLRT_CHECK(buf.capacity_bytes() == values.size() * sizeof(int));
    CCCLRT_CHECK(buf.end() == buf.begin() + buf.size());
    CCCLRT_CHECK(cuda::std::span<int>{buf}.size() == buf.size());
    CCCLRT_CHECK(static_cast<const base_t&>(buf).size() == buf.size());
    check_prefix(buf, values);
  }

  SECTION("construction from an existing buffer")
  {
    cuda::std::array<int, 6> values{1, 42, 1337, 0, 12, -1};
    base_t base{stream, resource, values.begin(), values.end()};
    const auto* const allocation = base.data();

    buffer_t buf{cuda::std::move(base)};

    CCCLRT_CHECK(buf.size() == values.size());
    CCCLRT_CHECK(buf.capacity() == values.size());
    CCCLRT_CHECK(buf.data() == allocation);
    check_prefix(buf, values);
  }

  SECTION("resize without reallocation")
  {
    cuda::std::array<int, 6> values{1, 42, 1337, 0, 12, -1};
    buffer_t buf{stream, resource, values.begin(), values.end()};
    const auto* const allocation = buf.data();

    buf.resize(3);
    CCCLRT_CHECK(buf.size() == 3);
    CCCLRT_CHECK(buf.capacity() == values.size());
    CCCLRT_CHECK(buf.data() == allocation);
    CCCLRT_CHECK(buf.end() == buf.begin() + buf.size());
    CCCLRT_CHECK(static_cast<const base_t&>(buf).size() == buf.size());
    check_prefix(buf, cuda::std::array<int, 3>{1, 42, 1337});

    buf.resize(5, cuda::no_init);
    CCCLRT_CHECK(buf.size() == 5);
    CCCLRT_CHECK(buf.capacity() == values.size());
    CCCLRT_CHECK(buf.data() == allocation);
    CCCLRT_CHECK(buf.end() == buf.begin() + buf.size());
    CCCLRT_CHECK(static_cast<const base_t&>(buf).size() == buf.size());
    check_prefix(buf, cuda::std::array<int, 3>{1, 42, 1337});

    CHECK_THROWS_AS(buf.resize(6), std::invalid_argument);
    CHECK_THROWS_AS(buf.resize(7, cuda::no_init), std::invalid_argument);
  }

  SECTION("resize with reallocation")
  {
    cuda::std::array<int, 6> values{1, 42, 1337, 0, 12, -1};
    buffer_t buf{stream, resource, values.begin(), values.end()};
    buf.resize(3);

    cuda::stream other_stream{device};
    buf.resize(other_stream, 9, cuda::no_init);

    CCCLRT_CHECK(buf.size() == 9);
    CCCLRT_CHECK(buf.capacity() == 9);
    CCCLRT_CHECK(buf.stream() == other_stream);
    CCCLRT_CHECK(buf.end() == buf.begin() + buf.size());
    CCCLRT_CHECK(static_cast<const base_t&>(buf).size() == buf.size());
    check_prefix(buf, cuda::std::array<int, 3>{1, 42, 1337});

    const auto* const allocation = buf.data();
    buf.resize(2);
    CCCLRT_CHECK(buf.size() == 2);
    CCCLRT_CHECK(buf.capacity() == 9);
    CCCLRT_CHECK(buf.data() == allocation);
    CCCLRT_CHECK(static_cast<const base_t&>(buf).size() == buf.size());
    check_prefix(buf, cuda::std::array<int, 2>{1, 42});
  }

  SECTION("resize from empty logical size with reallocation")
  {
    buffer_t buf{stream, resource, 3, cuda::no_init};
    buf.resize(0);

    cuda::stream other_stream{device};
    buf.resize(other_stream, 9, cuda::no_init);

    CCCLRT_CHECK(buf.size() == 9);
    CCCLRT_CHECK(buf.capacity() == 9);
    CCCLRT_CHECK(buf.stream() == other_stream);
  }

  SECTION("resize_discard")
  {
    cuda::std::array<int, 6> values{1, 42, 1337, 0, 12, -1};
    buffer_t buf{stream, resource, values.begin(), values.end()};

    cuda::stream other_stream{device};
    buf.resize_discard(other_stream, 9, cuda::no_init);

    CCCLRT_CHECK(buf.size() == 9);
    CCCLRT_CHECK(buf.capacity() == 9);
    CCCLRT_CHECK(buf.stream() == other_stream);
    CCCLRT_CHECK(static_cast<const base_t&>(buf).size() == buf.size());
  }

  SECTION("move and swap preserve logical size and capacity")
  {
    cuda::std::array<int, 6> values{1, 42, 1337, 0, 12, -1};
    buffer_t input{stream, resource, values.begin(), values.end()};
    input.resize(4);

    buffer_t moved{cuda::std::move(input)};
    CCCLRT_CHECK(moved.size() == 4);
    CCCLRT_CHECK(moved.capacity() == values.size());
    CCCLRT_CHECK(input.size() == 0);
    CCCLRT_CHECK(input.capacity() == 0);
    check_prefix(moved, cuda::std::array<int, 4>{1, 42, 1337, 0});

    buffer_t assigned{stream, resource, 8, cuda::no_init};
    assigned.resize(2);
    assigned = cuda::std::move(moved);

    CCCLRT_CHECK(assigned.size() == 4);
    CCCLRT_CHECK(assigned.capacity() == values.size());
    CCCLRT_CHECK(moved.size() == 0);
    CCCLRT_CHECK(moved.capacity() == 0);
    check_prefix(assigned, cuda::std::array<int, 4>{1, 42, 1337, 0});

    buffer_t other{stream, resource, 2, cuda::no_init};
    other.resize(1);
    swap(assigned, other);

    CCCLRT_CHECK(assigned.size() == 1);
    CCCLRT_CHECK(assigned.capacity() == 2);
    CCCLRT_CHECK(other.size() == 4);
    CCCLRT_CHECK(other.capacity() == values.size());
    check_prefix(other, cuda::std::array<int, 4>{1, 42, 1337, 0});
  }
}
