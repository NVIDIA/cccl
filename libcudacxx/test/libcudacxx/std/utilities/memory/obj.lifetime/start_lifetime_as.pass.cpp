//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// #include <memory>

// template<class T> T* start_lifetime_as(void* p) noexcept;
// template<class T> const T* start_lifetime_as(const void* p) noexcept;
// template<class T> volatile T* start_lifetime_as(volatile void* p) noexcept;
// template<class T> const volatile T* start_lifetime_as(const volatile void* p) noexcept;
// template<class T> T* start_lifetime_as_array(void* p, size_t n) noexcept;
// (and the const, volatile and const volatile overloads of the latter)

#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/cstring>
#include <cuda/std/memory>
#include <cuda/std/type_traits>

#include "test_macros.h"

struct aggregate
{
  int a;
  float b;
  char c;
};

// trivially copyable, but not default constructible
struct no_default_ctor
{
  int value;

  TEST_HOST_DEVICE_FUNC explicit no_default_ctor(int v)
      : value(v)
  {}
};

// not an implicit-lifetime type, but arrays of it are
struct not_implicit_lifetime
{
  TEST_HOST_DEVICE_FUNC not_implicit_lifetime() {}
  TEST_HOST_DEVICE_FUNC ~not_implicit_lifetime() {}
};

template <class T>
TEST_HOST_DEVICE_FUNC void test_object(const T& source)
{
  alignas(T) unsigned char buffer[sizeof(T)];
  cuda::std::memcpy(buffer, &source, sizeof(T));

  void* p                  = buffer;
  const void* cp           = buffer;
  volatile void* vp        = buffer;
  const volatile void* cvp = buffer;

  static_assert(cuda::std::is_same_v<T*, decltype(cuda::std::start_lifetime_as<T>(p))>);
  static_assert(cuda::std::is_same_v<const T*, decltype(cuda::std::start_lifetime_as<T>(cp))>);
  static_assert(cuda::std::is_same_v<volatile T*, decltype(cuda::std::start_lifetime_as<T>(vp))>);
  static_assert(cuda::std::is_same_v<const volatile T*, decltype(cuda::std::start_lifetime_as<T>(cvp))>);
  static_assert(noexcept(cuda::std::start_lifetime_as<T>(p)));

  // the object has the value representation of the bytes in the storage
  T* result = cuda::std::start_lifetime_as<T>(p);
  assert(static_cast<void*>(result) == p);
  assert(cuda::std::memcmp(result, &source, sizeof(T)) == 0);

  const T* cresult = cuda::std::start_lifetime_as<T>(cp);
  assert(static_cast<const void*>(cresult) == cp);
  assert(cuda::std::memcmp(cresult, &source, sizeof(T)) == 0);

  volatile T* vresult = cuda::std::start_lifetime_as<T>(vp);
  assert(static_cast<volatile void*>(vresult) == vp);

  const volatile T* cvresult = cuda::std::start_lifetime_as<T>(cvp);
  assert(static_cast<const volatile void*>(cvresult) == cvp);
}

template <class T, cuda::std::size_t N>
TEST_HOST_DEVICE_FUNC void test_array(const T (&source)[N])
{
  alignas(T) unsigned char buffer[sizeof(T) * N];
  cuda::std::memcpy(buffer, source, sizeof(T) * N);

  void* p                  = buffer;
  const void* cp           = buffer;
  volatile void* vp        = buffer;
  const volatile void* cvp = buffer;

  static_assert(cuda::std::is_same_v<T*, decltype(cuda::std::start_lifetime_as_array<T>(p, N))>);
  static_assert(cuda::std::is_same_v<const T*, decltype(cuda::std::start_lifetime_as_array<T>(cp, N))>);
  static_assert(cuda::std::is_same_v<volatile T*, decltype(cuda::std::start_lifetime_as_array<T>(vp, N))>);
  static_assert(cuda::std::is_same_v<const volatile T*, decltype(cuda::std::start_lifetime_as_array<T>(cvp, N))>);
  static_assert(noexcept(cuda::std::start_lifetime_as_array<T>(p, N)));

  T* result = cuda::std::start_lifetime_as_array<T>(p, N);
  assert(static_cast<void*>(result) == p);
  assert(cuda::std::memcmp(result, source, sizeof(T) * N) == 0);

  // an empty array has no effect and returns a pointer to the storage, which may be null
  assert(cuda::std::start_lifetime_as_array<T>(p, 0) == static_cast<T*>(p));
  assert(cuda::std::start_lifetime_as_array<T>(static_cast<void*>(nullptr), 0) == nullptr);
}

// The storage is modified through another type between two calls. The second call creates a new object with the
// modified bytes, so the compiler must not reuse the value read through the first object or reorder the accesses.
_CCCL_NOINLINE TEST_HOST_DEVICE_FUNC float reread_after_store(void* p, void* q)
{
  const float a                                         = *cuda::std::start_lifetime_as<float>(p);
  *cuda::std::start_lifetime_as<cuda::std::uint32_t>(q) = 0x40000000u; // 2.0f
  const float b                                         = *cuda::std::start_lifetime_as<float>(p);
  return a + b;
}

TEST_HOST_DEVICE_FUNC void test_reread_after_store()
{
  static_assert(sizeof(float) == sizeof(cuda::std::uint32_t));
  alignas(float) unsigned char buffer[sizeof(float)];
  const float one = 1.0f;
  cuda::std::memcpy(buffer, &one, sizeof(float));
  assert(reread_after_store(buffer, buffer) == 3.0f);
}

TEST_HOST_DEVICE_FUNC void test()
{
  test_object(42);
  test_object(3.5);
  test_object(aggregate{1, 2.0f, 'c'});
  test_object(no_default_ctor{7});

  const cuda::std::uint32_t words[] = {1u, 2u, 3u, 4u};
  test_array(words);
  const aggregate aggregates[] = {{1, 2.0f, 'a'}, {3, 4.0f, 'b'}};
  test_array(aggregates);

  // start_lifetime_as_array only requires a complete type
  alignas(not_implicit_lifetime) unsigned char buffer[sizeof(not_implicit_lifetime) * 2];
  assert(static_cast<void*>(cuda::std::start_lifetime_as_array<not_implicit_lifetime>(buffer, 2)) == buffer);

  test_reread_after_store();
}

int main(int, char**)
{
  test();
  return 0;
}
