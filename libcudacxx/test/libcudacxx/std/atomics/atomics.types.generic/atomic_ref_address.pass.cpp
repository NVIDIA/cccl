//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: libcpp-has-no-threads, pre-sm-60
// UNSUPPORTED: windows && pre-sm-70

// <cuda/std/atomic>

// constexpr address-return-type address() const noexcept;

#include <cuda/std/atomic>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <typename T>
struct TestAddress
{
  void operator()() const
  {
    alignas(cuda::std::atomic_ref<T>::required_alignment) T x(T(1));
    const cuda::std::atomic_ref<T> a(x);

    using AddressReturnT = void*;

    auto p = a.address();
    static_assert(cuda::std::is_same_v<decltype(p), AddressReturnT>, "address() return type mismatch");
    assert(cuda::std::addressof(x) == p);

    static_assert(noexcept(a.address()));
  }
};

template <typename T>
struct TestAddressConst
{
  void operator()() const
  {
    alignas(cuda::std::atomic_ref<const T>::required_alignment) T x(T(1));
    const cuda::std::atomic_ref<const T> a(x);

    using AddressReturnT = const void*;

    auto p = a.address();
    static_assert(cuda::std::is_same_v<decltype(p), AddressReturnT>, "address() return type mismatch");
    assert(cuda::std::addressof(x) == p);

    static_assert(noexcept(a.address()));
  }
};

template <typename T>
struct TestAddressVolatile
{
  void operator()() const
  {
    if constexpr (cuda::std::atomic_ref<T>::is_always_lock_free)
    {
      alignas(cuda::std::atomic_ref<volatile T>::required_alignment) volatile T x(T(1));
      const cuda::std::atomic_ref<volatile T> a(x);

      using AddressReturnT = volatile void*;

      auto p = a.address();
      static_assert(cuda::std::is_same_v<decltype(p), AddressReturnT>, "address() return type mismatch");
      assert(cuda::std::addressof(x) == p);

      static_assert(noexcept(a.address()));
    }
  }
};

template <typename T>
struct TestAddressCV
{
  void operator()() const
  {
    if constexpr (cuda::std::atomic_ref<T>::is_always_lock_free)
    {
      alignas(cuda::std::atomic_ref<const volatile T>::required_alignment) const volatile T x(T(1));
      const cuda::std::atomic_ref<const volatile T> a(x);

      using AddressReturnT = const volatile void*;

      auto p = a.address();
      static_assert(cuda::std::is_same_v<decltype(p), AddressReturnT>, "address() return type mismatch");
      assert(cuda::std::addressof(x) == p);

      static_assert(noexcept(a.address()));
    }
  }
};

int main(int, char**)
{
#if _CCCL_STD_VER >= 2026
  TestAddress<int>()();
  TestAddress<float>()();
  TestAddress<int*>()();

  TestAddressConst<int>()();
  TestAddressConst<float>()();
  TestAddressConst<int*>()();

  TestAddressVolatile<int>()();
  TestAddressVolatile<float>()();

  TestAddressCV<int>()();
  TestAddressCV<float>()();
#else
  // atomic_ref::address() is only available in C++26
#endif // _CCCL_STD_VER >= 2026

  return 0;
}
