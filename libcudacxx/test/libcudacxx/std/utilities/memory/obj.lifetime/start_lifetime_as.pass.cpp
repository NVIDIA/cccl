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

  // an empty array has no effect and returns a pointer to the storage
  assert(cuda::std::start_lifetime_as_array<T>(p, 0) == static_cast<T*>(p));
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
}

int main(int, char**)
{
  test();
  return 0;
}
