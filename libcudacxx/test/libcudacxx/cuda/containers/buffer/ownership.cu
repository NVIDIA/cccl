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
#include <cuda/std/limits>
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
  using Buffer               = c2h::get<0, TestType>;
  using BufferProperties     = typename Buffer::properties_list;
  using T                    = typename Buffer::value_type;
  using ByteBuffer           = decltype(cuda::std::declval<Buffer>().template as_bytes<cuda::std::byte>());
  using ByteBufferProperties = typename Buffer::properties_list;

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
  static_assert(cuda::std::is_same_v<ByteBufferProperties, BufferProperties>);

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
      CHECK(released.ptr == allocation);
      CHECK(released.capacity == 6);
      CHECK(released.size == 6);
      CHECK(released.alignment == alignment);
      CHECK(released.stream == cuda::stream_ref{stream});
      CHECK(buf.empty());
      CHECK(buf.data() == nullptr);
      CHECK(log.live.size() == 1);

      Buffer acquired = cuda::__acquire_buffer(cuda::std::move(released));
      CHECK(acquired.data() == allocation);
      CHECK(acquired.alignment() == alignment);
      CHECK(acquired.stream() == cuda::stream_ref{stream});
      CHECK(equal_range(acquired));
    }

    SECTION("release of an empty buffer")
    {
      Buffer buf{stream, resource, 0, cuda::no_init};
      auto released = cuda::std::move(buf).release();
      CHECK(released.ptr == nullptr);
      CHECK(released.capacity == 0);
      CHECK(released.size == 0);

      Buffer const acquired = cuda::__acquire_buffer(cuda::std::move(released));
      CHECK(acquired.empty());
    }

    SECTION("acquire rejects invalid allocations")
    {
      using Info = cuda::__allocation_info<T, cuda::mr::device_accessible>;

      T* ptr = reinterpret_cast<T*>(sizeof(T)); // NOLINT(performance-no-int-to-ptr)

      // Null buffer with non-zero size
      CHECK_THROWS_AS((void) cuda::__acquire_buffer(Info{stream, resource, nullptr, 1, 1, alignof(T)}),
                      std::invalid_argument);
      // Misligned pointer
      CHECK_THROWS_AS(
        (void) cuda::__acquire_buffer(Info{stream, resource, reinterpret_cast<T*>(1), 1, 1, 2 * alignof(T)}),
        std::invalid_argument);
      // Invalid alignment
      CHECK_THROWS_AS((void) cuda::__acquire_buffer(Info{stream, resource, nullptr, 0, 0, 3}), std::invalid_argument);
      // Capacity/size mismatch (TODO: lift this when buffers are resizable)
      CHECK_THROWS_AS((void) cuda::__acquire_buffer(Info{stream, resource, ptr, 2, 1, alignof(T)}),
                      std::invalid_argument);
      // Capacity less than size
      CHECK_THROWS_AS((void) cuda::__acquire_buffer(Info{stream, resource, ptr, 1, 2, alignof(T)}),
                      std::invalid_argument);
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
        CHECK(static_cast<const void*>(bytes.data()) == allocation);
        CHECK(bytes.size() == 6 * sizeof(T));
        CHECK(bytes.alignment() == alignment);
        CHECK(bytes.stream() == cuda::stream_ref{stream});
        CHECK(buf.empty());

        Buffer typed = cuda::std::move(bytes).template as_type<T>();
        CHECK(static_cast<const void*>(typed.data()) == allocation);
        CHECK(typed.size() == 6);
        CHECK(typed.alignment() == alignment);
        CHECK(bytes.empty());
        CHECK(equal_range(typed));
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
      CHECK(bytes.empty());
      CHECK(bytes.data() == nullptr);
    }

    SECTION("as_type rejects incompatible size or alignment and leaves the buffer unchanged")
    {
      using Int4 = cuda::std::int32_t;

      { // size is not a multiple of sizeof(U)
        const auto env = ::cuda::std::execution::prop{::cuda::allocation_alignment, alignof(Int4)};
        ByteBuffer bytes{stream, resource, 5, cuda::no_init, env};
        const auto* allocation = bytes.data();
        CHECK_THROWS_AS((void) cuda::std::move(bytes).template as_type<Int4>(), std::invalid_argument);
        CHECK(bytes.data() == allocation);
        CHECK(bytes.size() == 5);
      }

      { // alignment is less than alignof(U)
        ByteBuffer bytes{stream, resource, 8, cuda::no_init};
        CHECK(bytes.alignment() < alignof(Int4));
        const auto* allocation = bytes.data();
        CHECK_THROWS_AS((void) cuda::std::move(bytes).template as_type<Int4>(), std::invalid_argument);
        CHECK(bytes.data() == allocation);
        CHECK(bytes.size() == 8);
      }
    }
  }
  stream.sync();
  CHECK(log.live.empty());
  CHECK(log.mismatched_deallocations == 0);
}

template <class Buffer>
struct resizable_buffer_type;

template <class T, class... Properties>
struct resizable_buffer_type<cuda::buffer<T, Properties...>>
{
  using type      = cuda::__resizable_buffer<T, Properties...>;
  using byte_type = cuda::__resizable_buffer<cuda::std::byte, Properties...>;
};

C2H_CCCLRT_TEST("cuda::__resizable_buffer release, acquire and byte conversion", "[container][buffer]", test_types)
{
  using Base       = c2h::get<0, TestType>;
  using Buffer     = typename resizable_buffer_type<Base>::type;
  using T          = typename Buffer::value_type;
  using ByteBuffer = typename resizable_buffer_type<Base>::byte_type;

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
  static_assert(noexcept(cuda::std::declval<Buffer>().template as_bytes<cuda::std::byte>()));
  static_assert(!has_as_type<Buffer, float>);
  static_assert(has_as_type<ByteBuffer, T>);
  static_assert(!has_as_type<ByteBuffer&, T>);
  static_assert(
    cuda::std::is_same_v<decltype(cuda::std::declval<Buffer>().template as_bytes<cuda::std::byte>()), ByteBuffer>);
  static_assert(cuda::std::is_same_v<typename ByteBuffer::properties_list, typename Buffer::properties_list>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::declval<ByteBuffer>().template as_type<T>()), Buffer>);

  if (!extract_properties<Base>::is_resource_supported())
  {
    return;
  }

  const cuda::stream stream{cuda::device_ref{0}};
  allocation_log log;
  {
    tracking_resource resource{extract_properties<Base>::get_resource(), log};
    const cuda::std::size_t alignment = cuda::mr::default_cuda_malloc_alignment / 2;
    const auto env                    = cuda::std::execution::prop{cuda::allocation_alignment, alignment};

    SECTION("release and acquire preserve spare capacity")
    {
      Buffer buf{stream, resource, {T(1), T(42), T(1337), T(0), T(12), T(-1)}, env};
      const auto* allocation = buf.data();
      const auto size        = GENERATE(cuda::std::size_t{0}, cuda::std::size_t{3}, cuda::std::size_t{6});
      buf.resize_unsynchronized(size, cuda::no_init);

      auto released = cuda::std::move(buf).release();
      CHECK(released.ptr == allocation);
      CHECK(released.size == size);
      CHECK(released.capacity == 6);
      CHECK(released.alignment == alignment);
      CHECK(released.stream == cuda::stream_ref{stream});
      CHECK(buf.data() == nullptr);
      CHECK(buf.size() == 0);
      CHECK(buf.capacity() == 0);
      CHECK(log.live.size() == 1);

      Buffer acquired = cuda::__acquire_resizable_buffer(cuda::std::move(released));
      CHECK(acquired.data() == allocation);
      CHECK(acquired.size() == size);
      CHECK(acquired.capacity() == 6);
      CHECK(acquired.alignment() == alignment);
      CHECK(acquired.stream() == cuda::stream_ref{stream});
      acquired.resize_unsynchronized(6, cuda::no_init);
      CHECK(equal_range(acquired));
      acquired.resize_unsynchronized(size, cuda::no_init);
    }

    SECTION("byte conversion preserves spare capacity with every byte type")
    {
      const auto size = GENERATE(cuda::std::size_t{0}, cuda::std::size_t{3}, cuda::std::size_t{6});
      auto check      = [&](auto tag) {
        using Byte = typename decltype(tag)::type;
        Buffer buf{stream, resource, {T(1), T(42), T(1337), T(0), T(12), T(-1)}, env};
        const void* allocation = buf.data();
        buf.resize_unsynchronized(size, cuda::no_init);

        auto bytes = cuda::std::move(buf).template as_bytes<Byte>();
        using Expected =
          typename resizable_buffer_type<decltype(cuda::std::declval<Base>().template as_bytes<Byte>())>::type;
        static_assert(cuda::std::is_same_v<decltype(bytes), Expected>);
        CHECK(static_cast<const void*>(bytes.data()) == allocation);
        CHECK(bytes.size() == size * sizeof(T));
        CHECK(bytes.capacity() == 6 * sizeof(T));
        CHECK(bytes.alignment() == alignment);
        CHECK(bytes.stream() == cuda::stream_ref{stream});
        CHECK(buf.size() == 0);
        CHECK(buf.capacity() == 0);

        Buffer typed = cuda::std::move(bytes).template as_type<T>();
        CHECK(static_cast<const void*>(typed.data()) == allocation);
        CHECK(typed.size() == size);
        CHECK(typed.capacity() == 6);
        CHECK(typed.alignment() == alignment);
        CHECK(typed.stream() == cuda::stream_ref{stream});
        CHECK(bytes.data() == nullptr);
        CHECK(bytes.size() == 0);
        CHECK(bytes.capacity() == 0);
        typed.resize_unsynchronized(6, cuda::no_init);
        CHECK(equal_range(typed));
        typed.resize_unsynchronized(size, cuda::no_init);
      };
      check(cuda::std::type_identity<cuda::std::byte>{});
      check(cuda::std::type_identity<std::byte>{});
      check(cuda::std::type_identity<char>{});
      check(cuda::std::type_identity<unsigned char>{});
    }

    SECTION("deallocation uses capacity after ownership transfers")
    {
      Buffer buf{stream, resource, 512, cuda::no_init, env};
      buf.resize_unsynchronized(1, cuda::no_init);
      auto released      = cuda::std::move(buf).release();
      auto acquired      = cuda::__acquire_resizable_buffer(cuda::std::move(released));
      auto bytes         = cuda::std::move(acquired).template as_bytes<cuda::std::byte>();
      Buffer const typed = cuda::std::move(bytes).template as_type<T>();
      CHECK(typed.size() == 1);
      CHECK(typed.capacity() == 512);
    }

    SECTION("empty allocation round trips")
    {
      Buffer buf{stream, resource, 0, cuda::no_init, env};
      auto released = cuda::std::move(buf).release();
      CHECK(released.ptr == nullptr);
      CHECK(released.size == 0);
      CHECK(released.capacity == 0);
      auto acquired = cuda::__acquire_resizable_buffer(cuda::std::move(released));
      auto bytes    = cuda::std::move(acquired).template as_bytes<cuda::std::byte>();
      Buffer typed  = cuda::std::move(bytes).template as_type<T>();
      CHECK(typed.data() == nullptr);
      CHECK(typed.size() == 0);
      CHECK(typed.capacity() == 0);
      CHECK(typed.alignment() == alignment);
      CHECK(typed.stream() == cuda::stream_ref{stream});
    }

    SECTION("as_type rejects incompatible size, capacity or alignment without consuming the buffer")
    {
      using Int4                   = cuda::std::int32_t;
      const auto capacity          = GENERATE(cuda::std::size_t{5}, cuda::std::size_t{8});
      const cuda::std::size_t size = capacity == 5 ? 4 : 5;
      ByteBuffer bytes{stream, resource, capacity, cuda::no_init, env};
      bytes.resize_unsynchronized(size, cuda::no_init);
      const auto* allocation = bytes.data();
      CHECK_THROWS_AS((void) cuda::std::move(bytes).template as_type<Int4>(), std::invalid_argument);
      CHECK(bytes.data() == allocation);
      CHECK(bytes.size() == size);
      CHECK(bytes.capacity() == capacity);
      CHECK(bytes.alignment() == alignment);
      CHECK(bytes.stream() == cuda::stream_ref{stream});

      ByteBuffer unaligned{stream, resource, 8, cuda::no_init};
      unaligned.resize_unsynchronized(4, cuda::no_init);
      const auto* unaligned_allocation = unaligned.data();
      CHECK_THROWS_AS((void) cuda::std::move(unaligned).template as_type<Int4>(), std::invalid_argument);
      CHECK(unaligned.data() == unaligned_allocation);
      CHECK(unaligned.size() == 4);
      CHECK(unaligned.capacity() == 8);
    }

    SECTION("acquire rejects invalid allocations")
    {
      using Info = typename Buffer::__allocation_info_t;

      T* ptr = reinterpret_cast<T*>(alignment); // NOLINT(performance-no-int-to-ptr)

      // Capacity less than size
      CHECK_THROWS_AS((void) cuda::__acquire_resizable_buffer(Info{stream, resource, ptr, 1, 2, alignment}),
                      std::invalid_argument);
      // Null buffer with non-zero capacity, even when size is zero
      CHECK_THROWS_AS((void) cuda::__acquire_resizable_buffer(Info{stream, resource, nullptr, 1, 0, alignment}),
                      std::invalid_argument);
      // Invalid alignment
      CHECK_THROWS_AS((void) cuda::__acquire_resizable_buffer(Info{stream, resource, nullptr, 0, 0, 3}),
                      std::invalid_argument);
      // Misaligned pointer
      CHECK_THROWS_AS(
        (void) cuda::__acquire_resizable_buffer(Info{stream, resource, reinterpret_cast<T*>(1), 1, 0, alignment}),
        std::invalid_argument);
      // Allocation size overflows due to capacity, even when size is zero
      CHECK_THROWS_AS((void) cuda::__acquire_resizable_buffer(
                        Info{stream, resource, ptr, cuda::std::numeric_limits<cuda::std::size_t>::max(), 0, alignment}),
                      std::invalid_argument);
    }
  }
  stream.sync();
  CHECK(log.live.empty());
  CHECK(log.mismatched_deallocations == 0);
}
