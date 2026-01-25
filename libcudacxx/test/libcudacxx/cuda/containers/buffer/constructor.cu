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

// Checks if the offsetting resource wrapper correctly got passed the alignment by the buffer constructor.
template <typename T>
bool check_offseted_pointer(const T* ptr)
{
  return (ptr != nullptr)
      && (reinterpret_cast<std::uintptr_t>(ptr) % cuda::mr::default_cuda_malloc_alignment == alignof(T));
}

C2H_CCCLRT_TEST("cuda::buffer constructors", "[container][buffer]", test_types)
{
  using TestT    = c2h::get<0, TestType>;
  using Resource = typename extract_properties<TestT>::resource;
  using Buffer   = typename extract_properties<TestT>::buffer;
  using T        = typename Buffer::value_type;

  cuda::stream stream{cuda::device_ref{0}};
  Resource resource = extract_properties<TestT>::get_resource();

  SECTION("Construction with explicit size")
  {
    { // from stream and resource, no allocation
      const Buffer buf{stream, resource};
      CCCLRT_CHECK(buf.empty());
      CCCLRT_CHECK(buf.data() == nullptr);
    }

    {
      const auto buf = cuda::make_buffer<T>(stream, resource);
      CCCLRT_CHECK(buf.empty());
      CCCLRT_CHECK(buf.data() == nullptr);
    }

    {
      const auto buf = cuda::make_buffer(stream, extract_properties<TestT>::get_resource(), 0, T{42});
      CCCLRT_CHECK(buf.empty());
      CCCLRT_CHECK(buf.data() == nullptr);
    }

    {
      const auto buf = cuda::make_buffer(stream, extract_properties<TestT>::get_resource(), 5, T{42});
      CCCLRT_CHECK(check_offseted_pointer(buf.data()));
      CCCLRT_CHECK(buf.size() == 5);
      CCCLRT_CHECK(equal_size_value(buf, 5, T(42)));
    }
  }

  { // from size with no_init, no allocation
    SECTION("from size with no_init, no allocation")
    {
      const Buffer buf{stream, resource, 0, cuda::no_init};
      CCCLRT_CHECK(buf.empty());
      CCCLRT_CHECK(buf.data() == nullptr);
    }
    {
      const auto buf = cuda::make_buffer<T>(stream, resource, 0, cuda::no_init);
      CCCLRT_CHECK(buf.empty());
      CCCLRT_CHECK(buf.data() == nullptr);
    }

    { // from size with no_init
      const Buffer buf{stream, resource, 5, cuda::no_init};
      CCCLRT_CHECK(check_offseted_pointer(buf.data()));
      CCCLRT_CHECK(buf.size() == 5);
    }
    {
      const auto buf = cuda::make_buffer<T>(stream, resource, 5, cuda::no_init);
      CCCLRT_CHECK(check_offseted_pointer(buf.data()));
      CCCLRT_CHECK(buf.size() == 5);
    }
  }

  SECTION("Construction from iterators")
  {
    const cuda::std::array<T, 6> input{T(1), T(42), T(1337), T(0), T(12), T(-1)};
    { // can be constructed from two equal input iterators
      Buffer buf(stream, resource, input.begin(), input.begin());
      CCCLRT_CHECK(buf.empty());
      CCCLRT_CHECK(buf.data() == nullptr);
    }
    {
      const auto buf = cuda::make_buffer<T>(stream, resource, input.begin(), input.begin());
      CCCLRT_CHECK(buf.empty());
      CCCLRT_CHECK(buf.data() == nullptr);
    }

    { // can be constructed from two input iterators
      Buffer buf(stream, resource, input.begin(), input.end());
      CCCLRT_CHECK(check_offseted_pointer(buf.data()));
      CCCLRT_CHECK(buf.size() == 6);
      CCCLRT_CHECK(equal_range(buf));
    }
    {
      const auto buf = cuda::make_buffer<T>(stream, resource, input.begin(), input.end());
      CCCLRT_CHECK(check_offseted_pointer(buf.data()));
      CCCLRT_CHECK(buf.size() == 6);
      CCCLRT_CHECK(equal_range(buf));
    }
  }

  SECTION("Construction from range")
  {
    { // can be constructed from an empty random access range
      Buffer buf(stream, resource, cuda::std::array<T, 0>{});
      CCCLRT_CHECK(buf.empty());
      CCCLRT_CHECK(buf.data() == nullptr);
    }
    {
      const auto buf = cuda::make_buffer<T>(stream, resource, cuda::std::array<T, 0>{});
      CCCLRT_CHECK(buf.empty());
      CCCLRT_CHECK(buf.data() == nullptr);
    }

    { // can be constructed from a non-empty random access range
      Buffer buf(stream, resource, cuda::std::array<T, 6>{T(1), T(42), T(1337), T(0), T(12), T(-1)});
      CCCLRT_CHECK(check_offseted_pointer(buf.data()));
      CCCLRT_CHECK(!buf.empty());
      CCCLRT_CHECK(equal_range(buf));
    }
    {
      const auto buf =
        cuda::make_buffer<T>(stream, resource, cuda::std::array<T, 6>{T(1), T(42), T(1337), T(0), T(12), T(-1)});
      CCCLRT_CHECK(check_offseted_pointer(buf.data()));
      CCCLRT_CHECK(!buf.empty());
      CCCLRT_CHECK(equal_range(buf));
    }
  }

  SECTION("Construction from initializer_list")
  {
    { // can be constructed from an empty initializer_list
      const cuda::std::initializer_list<T> input{};
      Buffer buf(stream, resource, input);
      CCCLRT_CHECK(buf.empty());
      CCCLRT_CHECK(buf.data() == nullptr);
    }
    {
      const auto buf = cuda::make_buffer(stream, resource, cuda::std::initializer_list<T>{});
      CCCLRT_CHECK(buf.empty());
      CCCLRT_CHECK(buf.data() == nullptr);
    }

    { // can be constructed from a non-empty initializer_list
      const cuda::std::initializer_list<T> input{T(1), T(42), T(1337), T(0), T(12), T(-1)};
      Buffer buf(stream, resource, input);
      CCCLRT_CHECK(check_offseted_pointer(buf.data()));
      CCCLRT_CHECK(buf.size() == 6);
      CCCLRT_CHECK(equal_range(buf));
    }
    {
      const auto buf =
        cuda::make_buffer(stream, resource, cuda::std::initializer_list<T>{T(1), T(42), T(1337), T(0), T(12), T(-1)});
      CCCLRT_CHECK(check_offseted_pointer(buf.data()));
      CCCLRT_CHECK(buf.size() == 6);
      CCCLRT_CHECK(equal_range(buf));
    }
  }

  SECTION("copy construction")
  {
    static_assert(!cuda::std::is_nothrow_copy_constructible<Buffer>::value, "");
    { // can be copy constructed from empty input
      const Buffer input{stream, resource, 0, cuda::no_init};
      Buffer buf(input);
      CCCLRT_CHECK(buf.empty());
    }

    { // can be copy constructed from non-empty input
      const Buffer input{stream, resource, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      Buffer buf(input);
      CCCLRT_CHECK(check_offseted_pointer(buf.data()));
      CCCLRT_CHECK(!buf.empty());
      CCCLRT_CHECK(equal_range(buf));
    }
  }

  SECTION("move construction")
  {
    static_assert(cuda::std::is_nothrow_move_constructible<Buffer>::value, "");

    { // can be move constructed with empty input
      Buffer input{stream, resource, 0, cuda::no_init};
      Buffer buf(cuda::std::move(input));
      CCCLRT_CHECK(buf.empty());
      CCCLRT_CHECK(input.empty());
    }

    { // can be move constructed from non-empty input
      Buffer input{stream, resource, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};

      // ensure that we steal the data
      const auto* allocation = input.data();
      Buffer buf(cuda::std::move(input));
      CCCLRT_CHECK(buf.size() == 6);
      CCCLRT_CHECK(buf.data() == allocation);
      CCCLRT_CHECK(input.size() == 0);
      CCCLRT_CHECK(input.data() == nullptr);
      CCCLRT_CHECK(equal_range(buf));
    }
  }

  SECTION("move assignment")
  {
    { // can be move assigned with empty input
      Buffer input{stream, resource, 0, cuda::no_init};
      Buffer buf{stream, resource, {T(1), T(42), T(1337)}};
      buf = cuda::std::move(input);
      CCCLRT_CHECK(buf.empty());
      CCCLRT_CHECK(input.empty());
    }

    { // can be move assigned from non-empty input
      Buffer input{stream, resource, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      Buffer buf{stream, resource, {T(99), T(88)}};

      // ensure that we steal the data
      const auto* allocation = input.data();
      buf                    = cuda::std::move(input);
      CCCLRT_CHECK(buf.size() == 6);
      CCCLRT_CHECK(buf.data() == allocation);
      CCCLRT_CHECK(input.size() == 0);
      CCCLRT_CHECK(input.data() == nullptr);
      CCCLRT_CHECK(equal_range(buf));
    }

    { // can be move assigned to empty buffer
      Buffer input{stream, resource, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      Buffer buf{stream, resource, 0, cuda::no_init};

      const auto* allocation = input.data();
      buf                    = cuda::std::move(input);
      CCCLRT_CHECK(buf.size() == 6);
      CCCLRT_CHECK(buf.data() == allocation);
      CCCLRT_CHECK(input.size() == 0);
      CCCLRT_CHECK(input.data() == nullptr);
      CCCLRT_CHECK(equal_range(buf));
    }

    { // self move assignment
      Buffer buf{stream, resource, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      const auto* allocation = buf.data();
      const auto size        = buf.size();
      test::assign(buf, cuda::std::move(buf));
      CCCLRT_CHECK(buf.size() == size);
      CCCLRT_CHECK(buf.data() == allocation);
      CCCLRT_CHECK(equal_range(buf));
    }
  }
  stream.sync();

#if 0 // Implement exception handling
#  if _CCCL_HAS_EXCEPTIONS()
  SECTION("Exception handling throwing bad_alloc")
  {
    using buffer = cuda::buffer<int>;

    try
    {
      buffer too_small(2 * capacity);
    }
    catch (const std::bad_alloc&)
    {}
    catch (...)
    {
      CCCLRT_CHECK(false);
    }

    try
    {
      buffer too_small(2 * capacity, 42);
    }
    catch (const std::bad_alloc&)
    {}
    catch (...)
    {
      CCCLRT_CHECK(false);
    }

    try
    {
      cuda::std::array<int, 2 * capacity> input{0, 1, 2, 3, 4, 5, 6, 7};
      buffer too_small(input.begin(), input.end());
    }
    catch (const std::bad_alloc&)
    {}
    catch (...)
    {
      CCCLRT_CHECK(false);
    }

    try
    {
      cuda::std::initializer_list<int> input{0, 1, 2, 3, 4, 5, 6};
      buffer too_small(input);
    }
    catch (const std::bad_alloc&)
    {}
    catch (...)
    {
      CCCLRT_CHECK(false);
    }

    try
    {
      uncommon_range<int, 2 * capacity> input{{0, 1, 2, 3, 4, 5, 6, 7}};
      buffer too_small(input);
    }
    catch (const std::bad_alloc&)
    {}
    catch (...)
    {
      CCCLRT_CHECK(false);
    }

    try
    {
      sized_uncommon_range<int, 2 * capacity> input{{0, 1, 2, 3, 4, 5, 6, 7}};
      buffer too_small(input);
    }
    catch (const std::bad_alloc&)
    {}
    catch (...)
    {
      CCCLRT_CHECK(false);
    }

    try
    {
      cuda::std::array<int, 2 * capacity> input{0, 1, 2, 3, 4, 5, 6, 7};
      buffer too_small(input);
    }
    catch (const std::bad_alloc&)
    {}
    catch (...)
    {
      CCCLRT_CHECK(false);
    }
  }
#  endif // _CCCL_HAS_EXCEPTIONS()
#endif // 0
}

C2H_CCCLRT_TEST("cuda::buffer constructors with legacy resource", "[container][buffer]")
{
  cuda::stream stream{cuda::device_ref{0}};
  cuda::mr::legacy_pinned_memory_resource resource;
  auto input = compare_data_initializer_list;
  cuda::buffer<int, cuda::mr::device_accessible> buffer{stream, resource, input};
  CCCLRT_CHECK(equal_range(buffer));
  STATIC_CHECK(!decltype(buffer)::properties_list::has_property(cuda::mr::host_accessible{}));
  STATIC_CHECK(decltype(buffer)::properties_list::has_property(cuda::mr::device_accessible{}));

  cuda::buffer<int, cuda::mr::host_accessible> buffer2{stream, resource, input};
  auto buf2 = cuda::make_buffer(stream, resource, buffer2);
  CCCLRT_CHECK(equal_range(buffer2));
  STATIC_CHECK(decltype(buffer2)::properties_list::has_property(cuda::mr::host_accessible{}));
  STATIC_CHECK(!decltype(buffer2)::properties_list::has_property(cuda::mr::device_accessible{}));
}

#if _CCCL_CTK_AT_LEAST(12, 6)
C2H_CCCLRT_TEST("cuda::make_buffer narrowing properties", "[container][buffer]")
{
  auto resource = cuda::pinned_default_memory_pool();
  cuda::stream stream{cuda::device_ref{0}};

  auto buf = cuda::make_buffer<int>(stream, resource, 0, cuda::no_init);

  auto input      = compare_data_initializer_list;
  auto buf_host   = cuda::make_buffer<int, cuda::mr::host_accessible>(stream, resource, input);
  auto buf_device = cuda::make_buffer<int, cuda::mr::device_accessible>(stream, resource, 2, 42);

  STATIC_CHECK(decltype(buf)::properties_list::has_property(cuda::mr::host_accessible{}));
  STATIC_CHECK(decltype(buf)::properties_list::has_property(cuda::mr::device_accessible{}));
  STATIC_CHECK(decltype(buf_host)::properties_list::has_property(cuda::mr::host_accessible{}));
  STATIC_CHECK(!decltype(buf_host)::properties_list::has_property(cuda::mr::device_accessible{}));
  STATIC_CHECK(decltype(buf_device)::properties_list::has_property(cuda::mr::device_accessible{}));
  STATIC_CHECK(!decltype(buf_device)::properties_list::has_property(cuda::mr::host_accessible{}));

  CCCLRT_CHECK(buf.empty());
  CCCLRT_CHECK(equal_range(buf_host));
  CCCLRT_CHECK(buf_device.size() == 2);
}
#endif // ^^^ _CCCL_CTK_AT_LEAST(12, 6) ^^^

C2H_CCCLRT_TEST("cuda::make_buffer with memory_pool_ref", "[container][buffer]")
{
  cuda::stream stream{cuda::device_ref{0}};
  cuda::device_memory_pool pool{cuda::device_ref{0}};
  auto buf = cuda::make_buffer(stream, pool.as_ref(), 10, 42);
  CCCLRT_CHECK(buf.size() == 10);
  static_assert(decltype(buf)::properties_list::has_property(cuda::mr::device_accessible{}));
  static_assert(!decltype(buf)::properties_list::has_property(cuda::mr::host_accessible{}));
}

C2H_CCCLRT_TEST("cuda::make_buffer with shared_resource", "[container][buffer]")
{
  cuda::stream stream{cuda::device_ref{0}};
  auto shared_res = cuda::mr::make_shared_resource<cuda::device_memory_pool>(cuda::device_ref{0});
  auto buf        = cuda::make_buffer(stream, shared_res, 10, 42);
  CCCLRT_CHECK(buf.size() == 10);
  static_assert(decltype(buf)::properties_list::has_property(cuda::mr::device_accessible{}));
  static_assert(!decltype(buf)::properties_list::has_property(cuda::mr::host_accessible{}));
}
