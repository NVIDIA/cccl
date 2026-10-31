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
#include <cuda/memory_resource>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <cstddef>
#include <stdexcept>
#include <unordered_map>

#include <test_resources.h>

#include "helper.h"

struct allocation_log
{
  std::unordered_map<void*, cuda::std::pair<std::size_t, std::size_t>> live;
  int mismatched_deallocations = 0;

  void on_allocate(void* ptr, std::size_t size, std::size_t alignment)
  {
    live[ptr] = {size, alignment};
  }

  void on_deallocate(void* ptr, std::size_t size, std::size_t alignment)
  {
    const auto it = live.find(ptr);
    if (it == live.end() || it->second != cuda::std::pair<std::size_t, std::size_t>{size, alignment})
    {
      ++mismatched_deallocations;
      return;
    }
    live.erase(it);
  }
};

//! @brief Resource adapter that checks every deallocation matches the size and
//! alignment of the corresponding allocation.
template <typename Resource>
struct tracking_resource
    : ::cuda::mr::__copy_default_queries<Resource>
    , ::cuda::forward_property<tracking_resource<Resource>, Resource>
{
  Resource resource_;
  allocation_log* log_;

  tracking_resource(Resource resource, allocation_log& log) noexcept
      : resource_(cuda::std::move(resource))
      , log_(&log)
  {}

  void* allocate_sync(std::size_t size, std::size_t alignment)
  {
    void* ptr = resource_.allocate_sync(size, alignment);
    log_->on_allocate(ptr, size, alignment);
    return ptr;
  }
  void deallocate_sync(void* ptr, std::size_t size, std::size_t alignment)
  {
    log_->on_deallocate(ptr, size, alignment);
    resource_.deallocate_sync(ptr, size, alignment);
  }
  void* allocate(cuda::stream_ref stream, std::size_t size, std::size_t alignment)
  {
    void* ptr = resource_.allocate(stream, size, alignment);
    log_->on_allocate(ptr, size, alignment);
    return ptr;
  }
  void deallocate(cuda::stream_ref stream, void* ptr, std::size_t size, std::size_t alignment)
  {
    log_->on_deallocate(ptr, size, alignment);
    resource_.deallocate(stream, ptr, size, alignment);
  }

  bool operator==(const tracking_resource& other) const
  {
    return log_ == other.log_;
  }
  bool operator!=(const tracking_resource& other) const
  {
    return log_ != other.log_;
  }

  Resource& upstream_resource() noexcept
  {
    return resource_;
  }
  const Resource& upstream_resource() const noexcept
  {
    return resource_;
  }
};

template <class Buf, class = void>
inline constexpr bool has_release = false;
template <class Buf>
inline constexpr bool has_release<Buf, cuda::std::void_t<decltype(cuda::std::declval<Buf>().release())>> = true;

template <class Buf, class Byte, class = void>
inline constexpr bool has_as_bytes = false;
template <class Buf, class Byte>
inline constexpr bool
  has_as_bytes<Buf, Byte, cuda::std::void_t<decltype(cuda::std::declval<Buf>().template as_bytes<Byte>())>> = true;

template <class Buf, class U, class = void>
inline constexpr bool has_as_type = false;
template <class Buf, class U>
inline constexpr bool has_as_type<Buf, U, cuda::std::void_t<decltype(cuda::std::declval<Buf>().template as_type<U>())>> =
  true;

C2H_CCCLRT_TEST("cuda::buffer release, acquire and byte conversion", "[container][buffer]", test_types)
{
  using Buffer     = c2h::get<0, TestType>;
  using T          = typename Buffer::value_type;
  using ByteBuffer = decltype(cuda::std::declval<Buffer>().template as_bytes<cuda::std::byte>());

  static_assert(has_release<Buffer>);
  static_assert(!has_release<Buffer&>);
  static_assert(cuda::std::is_nothrow_invocable_v<decltype(&Buffer::release), Buffer&&>);

  static_assert(has_as_bytes<Buffer, cuda::std::byte>);
  static_assert(has_as_bytes<Buffer, std::byte>);
  static_assert(has_as_bytes<Buffer, char>);
  static_assert(has_as_bytes<Buffer, unsigned char>);
  static_assert(!has_as_bytes<Buffer, signed char>);
  static_assert(!has_as_bytes<Buffer, int>);
  static_assert(!has_as_bytes<Buffer&, cuda::std::byte>);

  static_assert(!has_as_type<Buffer, float>);
  static_assert(has_as_type<ByteBuffer, T>);
  static_assert(!has_as_type<ByteBuffer&, T>);
  static_assert(cuda::std::is_same_v<typename ByteBuffer::properties_list, typename Buffer::properties_list>);

  if (!extract_properties<Buffer>::is_resource_supported())
  {
    return;
  }

  const cuda::stream stream{cuda::device_ref{0}};
  allocation_log log;
  {
    tracking_resource resource{extract_properties<Buffer>::get_resource(), log};

    SECTION("release and acquire round trip")
    {
      Buffer buf{stream, resource, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      const auto* allocation = buf.data();
      const auto alignment   = buf.alignment();

      auto released = cuda::std::move(buf).release();
      CCCLRT_CHECK(released.ptr == allocation);
      CCCLRT_CHECK(released.size == 6);
      CCCLRT_CHECK(released.alignment == alignment);
      CCCLRT_CHECK(released.stream == cuda::stream_ref{stream});
      CCCLRT_CHECK(buf.empty());
      CCCLRT_CHECK(buf.data() == nullptr);
      CCCLRT_CHECK(log.live.size() == 1);

      Buffer acquired =
        Buffer::acquire(released.stream, cuda::std::move(released.mr), released.ptr, released.size, released.alignment);
      CCCLRT_CHECK(acquired.data() == allocation);
      CCCLRT_CHECK(acquired.alignment() == alignment);
      CCCLRT_CHECK(acquired.stream() == cuda::stream_ref{stream});
      CCCLRT_CHECK(equal_range(acquired));
    }

    SECTION("release of an empty buffer")
    {
      Buffer buf{stream, resource, 0, cuda::no_init};
      auto released = cuda::std::move(buf).release();
      CCCLRT_CHECK(released.ptr == nullptr);
      CCCLRT_CHECK(released.size == 0);

      Buffer acquired = Buffer::acquire(released.stream, cuda::std::move(released.mr), released.ptr, released.size);
      CCCLRT_CHECK(acquired.empty());
    }

    SECTION("acquire rejects invalid allocations")
    {
      CHECK_THROWS_AS((void) Buffer::acquire(stream, resource, nullptr, 1), std::invalid_argument);
      CHECK_THROWS_AS((void) Buffer::acquire(stream, resource, reinterpret_cast<T*>(1), 1, 2 * alignof(T)),
                      std::invalid_argument);
      CHECK_THROWS_AS((void) Buffer::acquire(stream, resource, nullptr, 0, 3), std::invalid_argument);
    }

    SECTION("as_bytes and as_type round trip with every byte type")
    {
      const ::cuda::std::size_t alignment = ::cuda::mr::default_cuda_malloc_alignment / 2;
      const auto env                      = ::cuda::std::execution::prop{::cuda::allocation_alignment, alignment};

      auto check = [&](auto tag) {
        using Byte = typename decltype(tag)::type;

        Buffer buf{stream, resource, {T(1), T(42), T(1337), T(0), T(12), T(-1)}, env};
        const void* allocation = buf.data();

        auto bytes = cuda::std::move(buf).template as_bytes<Byte>();
        static_assert(cuda::std::is_same_v<typename decltype(bytes)::value_type, Byte>);
        CCCLRT_CHECK(static_cast<const void*>(bytes.data()) == allocation);
        CCCLRT_CHECK(bytes.size() == 6 * sizeof(T));
        CCCLRT_CHECK(bytes.alignment() == alignment);
        CCCLRT_CHECK(bytes.stream() == cuda::stream_ref{stream});
        CCCLRT_CHECK(buf.empty());

        Buffer typed = cuda::std::move(bytes).template as_type<T>();
        CCCLRT_CHECK(static_cast<const void*>(typed.data()) == allocation);
        CCCLRT_CHECK(typed.size() == 6);
        CCCLRT_CHECK(typed.alignment() == alignment);
        CCCLRT_CHECK(bytes.empty());
        CCCLRT_CHECK(equal_range(typed));
      };
      check(cuda::std::type_identity<cuda::std::byte>{});
      check(cuda::std::type_identity<std::byte>{});
      check(cuda::std::type_identity<char>{});
      check(cuda::std::type_identity<unsigned char>{});
    }

    SECTION("as_bytes of an empty buffer")
    {
      Buffer buf{stream, resource, 0, cuda::no_init};
      auto bytes = cuda::std::move(buf).template as_bytes<cuda::std::byte>();
      CCCLRT_CHECK(bytes.empty());
      CCCLRT_CHECK(bytes.data() == nullptr);
    }

    SECTION("as_type rejects incompatible size or alignment and leaves the buffer unchanged")
    {
      using Int4 = cuda::std::int32_t;

      { // size is not a multiple of sizeof(U)
        const auto env = ::cuda::std::execution::prop{::cuda::allocation_alignment, alignof(Int4)};
        ByteBuffer bytes{stream, resource, 5, cuda::no_init, env};
        const auto* allocation = bytes.data();
        CHECK_THROWS_AS((void) cuda::std::move(bytes).template as_type<Int4>(), std::invalid_argument);
        CCCLRT_CHECK(bytes.data() == allocation);
        CCCLRT_CHECK(bytes.size() == 5);
      }

      { // alignment is less than alignof(U)
        ByteBuffer bytes{stream, resource, 8, cuda::no_init};
        CCCLRT_CHECK(bytes.alignment() < alignof(Int4));
        const auto* allocation = bytes.data();
        CHECK_THROWS_AS((void) cuda::std::move(bytes).template as_type<Int4>(), std::invalid_argument);
        CCCLRT_CHECK(bytes.data() == allocation);
        CCCLRT_CHECK(bytes.size() == 8);
      }
    }
  }
  stream.sync();
  CCCLRT_CHECK(log.live.empty());
  CCCLRT_CHECK(log.mismatched_deallocations == 0);
}

C2H_CCCLRT_TEST("cuda::__resizable_buffer disallows release and byte conversion", "[container][buffer]")
{
  using ResizableBuffer = cuda::__resizable_buffer<int, cuda::mr::device_accessible>;
  static_assert(!has_release<ResizableBuffer>);
  static_assert(!has_as_bytes<ResizableBuffer, cuda::std::byte>);
  static_assert(!has_as_type<cuda::__resizable_buffer<cuda::std::byte, cuda::mr::device_accessible>, int>);
}
