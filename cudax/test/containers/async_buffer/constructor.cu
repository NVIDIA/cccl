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

#include <cuda/experimental/container.cuh>

#include <stdexcept>

#include "helper.h"
#include "types.h"

#if _CCCL_CUDACC_AT_LEAST(12, 6)
using test_types = c2h::type_list<cuda::std::tuple<int, cuda::mr::host_accessible>,
                                  cuda::std::tuple<unsigned long long, cuda::mr::device_accessible>,
                                  cuda::std::tuple<int, cuda::mr::host_accessible, cuda::mr::device_accessible>>;
#else
using test_types = c2h::type_list<cuda::std::tuple<int, cuda::mr::device_accessible>>;
#endif

C2H_CCCLRT_TEST("cudax::async_buffer constructors", "[container][async_buffer]", test_types)
{
  using TestT    = c2h::get<0, TestType>;
  using Resource = typename extract_properties<TestT>::resource;
  using Buffer   = typename extract_properties<TestT>::async_buffer;
  using T        = typename Buffer::value_type;

  cudax::stream stream{cuda::device_ref{0}};
  Resource resource{};

  SECTION("Construction with explicit size")
  {
    { // from stream and resource, no allocation
      const Buffer buf{stream, resource};
      CUDAX_CHECK(buf.empty());
      CUDAX_CHECK(buf.data() == nullptr);
    }
    {
      const auto buf = cudax::make_async_buffer<T>(stream, resource);
      CUDAX_CHECK(buf.empty());
      CUDAX_CHECK(buf.data() == nullptr);
    }

    {
      const auto buf = cudax::make_async_buffer(stream, Resource{}, 0, T{42});
      CUDAX_CHECK(buf.empty());
      CUDAX_CHECK(buf.data() == nullptr);
    }

    {
      const auto buf = cudax::make_async_buffer(stream, Resource{}, 5, T{42});
      CUDAX_CHECK(buf.size() == 5);
      CUDAX_CHECK(equal_size_value(buf, 5, T(42)));
    }
  }

  { // from size with no_init, no allocation
    SECTION("from size with no_init, no allocation")
    {
      const Buffer buf{stream, resource, 0, cudax::no_init};
      CUDAX_CHECK(buf.empty());
      CUDAX_CHECK(buf.data() == nullptr);
    }
    {
      const auto buf = cudax::make_async_buffer<T>(stream, resource, 0, cudax::no_init);
      CUDAX_CHECK(buf.empty());
      CUDAX_CHECK(buf.data() == nullptr);
    }

    { // from size with no_init
      const Buffer buf{stream, resource, 5, cudax::no_init};
      CUDAX_CHECK(buf.size() == 5);
      CUDAX_CHECK(buf.data() != nullptr);
    }
    {
      const auto buf = cudax::make_async_buffer<T>(stream, resource, 5, cudax::no_init);
      CUDAX_CHECK(buf.size() == 5);
      CUDAX_CHECK(buf.data() != nullptr);
    }
  }

  SECTION("Construction from iterators")
  {
    const cuda::std::array<T, 6> input{T(1), T(42), T(1337), T(0), T(12), T(-1)};
    { // can be constructed from two equal input iterators
      Buffer buf(stream, resource, input.begin(), input.begin());
      CUDAX_CHECK(buf.empty());
      CUDAX_CHECK(buf.data() == nullptr);
    }
    {
      const auto buf = cudax::make_async_buffer<T>(stream, resource, input.begin(), input.begin());
      CUDAX_CHECK(buf.empty());
      CUDAX_CHECK(buf.data() == nullptr);
    }

    { // can be constructed from two input iterators
      Buffer buf(stream, resource, input.begin(), input.end());
      CUDAX_CHECK(buf.size() == 6);
      CUDAX_CHECK(equal_range(buf));
    }
    {
      const auto buf = cudax::make_async_buffer<T>(stream, resource, input.begin(), input.end());
      CUDAX_CHECK(buf.size() == 6);
      CUDAX_CHECK(equal_range(buf));
    }
  }

  SECTION("Construction from range")
  {
    { // can be constructed from an empty random access range
      Buffer buf(stream, resource, cuda::std::array<T, 0>{});
      CUDAX_CHECK(buf.empty());
      CUDAX_CHECK(buf.data() == nullptr);
    }
    {
      const auto buf = cudax::make_async_buffer<T>(stream, resource, cuda::std::array<T, 0>{});
      CUDAX_CHECK(buf.empty());
      CUDAX_CHECK(buf.data() == nullptr);
    }

    { // can be constructed from a non-empty random access range
      Buffer buf(stream, resource, cuda::std::array<T, 6>{T(1), T(42), T(1337), T(0), T(12), T(-1)});
      CUDAX_CHECK(!buf.empty());
      CUDAX_CHECK(equal_range(buf));
    }
    {
      const auto buf =
        cudax::make_async_buffer<T>(stream, resource, cuda::std::array<T, 6>{T(1), T(42), T(1337), T(0), T(12), T(-1)});
      CUDAX_CHECK(!buf.empty());
      CUDAX_CHECK(equal_range(buf));
    }
  }

  SECTION("Construction from initializer_list")
  {
    { // can be constructed from an empty initializer_list
      const cuda::std::initializer_list<T> input{};
      Buffer buf(stream, resource, input);
      CUDAX_CHECK(buf.empty());
      CUDAX_CHECK(buf.data() == nullptr);
    }
    {
      const auto buf = cudax::make_async_buffer(stream, resource, cuda::std::initializer_list<T>{});
      CUDAX_CHECK(buf.empty());
      CUDAX_CHECK(buf.data() == nullptr);
    }

    { // can be constructed from a non-empty initializer_list
      const cuda::std::initializer_list<T> input{T(1), T(42), T(1337), T(0), T(12), T(-1)};
      Buffer buf(stream, resource, input);
      CUDAX_CHECK(buf.size() == 6);
      CUDAX_CHECK(equal_range(buf));
    }
    {
      const auto buf = cudax::make_async_buffer(
        stream, resource, cuda::std::initializer_list<T>{T(1), T(42), T(1337), T(0), T(12), T(-1)});
      CUDAX_CHECK(buf.size() == 6);
      CUDAX_CHECK(equal_range(buf));
    }
  }

  SECTION("copy construction")
  {
    static_assert(!cuda::std::is_nothrow_copy_constructible<Buffer>::value, "");
    { // can be copy constructed from empty input
      const Buffer input{stream, resource, 0, cudax::no_init};
      Buffer buf(input);
      CUDAX_CHECK(buf.empty());
    }

    { // can be copy constructed from non-empty input
      const Buffer input{stream, resource, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      Buffer buf(input);
      CUDAX_CHECK(!buf.empty());
      CUDAX_CHECK(equal_range(buf));
    }
  }

  SECTION("move construction")
  {
    static_assert(cuda::std::is_nothrow_move_constructible<Buffer>::value, "");

    { // can be move constructed with empty input
      Buffer input{stream, resource, 0, cudax::no_init};
      Buffer buf(cuda::std::move(input));
      CUDAX_CHECK(buf.empty());
      CUDAX_CHECK(input.empty());
    }

    { // can be move constructed from non-empty input
      Buffer input{stream, resource, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};

      // ensure that we steal the data
      const auto* allocation = input.data();
      Buffer buf(cuda::std::move(input));
      CUDAX_CHECK(buf.size() == 6);
      CUDAX_CHECK(buf.data() == allocation);
      CUDAX_CHECK(input.size() == 0);
      CUDAX_CHECK(input.data() == nullptr);
      CUDAX_CHECK(equal_range(buf));
    }
  }
  stream.sync();

#if 0 // Implement exception handling
#  if _CCCL_HAS_EXCEPTIONS()
  SECTION("Exception handling throwing bad_alloc")
  {
    using async_buffer = cudax::async_buffer<int>;

    try
    {
      async_buffer too_small(2 * capacity);
    }
    catch (const std::bad_alloc&)
    {}
    catch (...)
    {
      CUDAX_CHECK(false);
    }

    try
    {
      async_buffer too_small(2 * capacity, 42);
    }
    catch (const std::bad_alloc&)
    {}
    catch (...)
    {
      CUDAX_CHECK(false);
    }

    try
    {
      cuda::std::array<int, 2 * capacity> input{0, 1, 2, 3, 4, 5, 6, 7};
      async_buffer too_small(input.begin(), input.end());
    }
    catch (const std::bad_alloc&)
    {}
    catch (...)
    {
      CUDAX_CHECK(false);
    }

    try
    {
      cuda::std::initializer_list<int> input{0, 1, 2, 3, 4, 5, 6};
      async_buffer too_small(input);
    }
    catch (const std::bad_alloc&)
    {}
    catch (...)
    {
      CUDAX_CHECK(false);
    }

    try
    {
      uncommon_range<int, 2 * capacity> input{{0, 1, 2, 3, 4, 5, 6, 7}};
      async_buffer too_small(input);
    }
    catch (const std::bad_alloc&)
    {}
    catch (...)
    {
      CUDAX_CHECK(false);
    }

    try
    {
      sized_uncommon_range<int, 2 * capacity> input{{0, 1, 2, 3, 4, 5, 6, 7}};
      async_buffer too_small(input);
    }
    catch (const std::bad_alloc&)
    {}
    catch (...)
    {
      CUDAX_CHECK(false);
    }

    try
    {
      cuda::std::array<int, 2 * capacity> input{0, 1, 2, 3, 4, 5, 6, 7};
      async_buffer too_small(input);
    }
    catch (const std::bad_alloc&)
    {}
    catch (...)
    {
      CUDAX_CHECK(false);
    }
  }
#  endif // _CCCL_HAS_EXCEPTIONS()
#endif // 0
}
