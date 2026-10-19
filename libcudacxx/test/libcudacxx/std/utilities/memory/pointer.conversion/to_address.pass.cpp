//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <memory>

// template <class T> constexpr T* to_address(T* p) noexcept;
// template <class Ptr> constexpr auto to_address(const Ptr& p) noexcept;

#include <cuda/std/cassert>
#include <cuda/std/memory>
#include <cuda/std/utility>

#include "test_macros.h"

struct Irrelevant;

struct P1
{
  using element_type = Irrelevant;
  TEST_FUNC constexpr explicit P1(int* p)
      : p_(p)
  {}
  TEST_FUNC constexpr int* operator->() const
  {
    return p_;
  }
  int* p_;
};

struct P2
{
  using element_type = Irrelevant;
  TEST_FUNC constexpr explicit P2(int* p)
      : p_(p)
  {}
  TEST_FUNC constexpr P1 operator->() const
  {
    return p_;
  }
  P1 p_;
};

struct P3
{
  TEST_FUNC constexpr explicit P3(int* p)
      : p_(p)
  {}
  int* p_;
};

template <>
struct cuda::std::pointer_traits<P3>
{
  TEST_FUNC static constexpr int* to_address(const P3& p)
  {
    return p.p_;
  }
};

struct P4
{
  TEST_FUNC constexpr explicit P4(int* p)
      : p_(p)
  {}
  TEST_FUNC int* operator->() const; // should never be called
  int* p_;
};

template <>
struct cuda::std::pointer_traits<P4>
{
  TEST_FUNC static constexpr int* to_address(const P4& p)
  {
    return p.p_;
  }
};

struct P5
{
  using element_type = Irrelevant;
  TEST_FUNC int const* const& operator->() const;
};

struct P6
{};

template <>
struct cuda::std::pointer_traits<P6>
{
  TEST_FUNC static int const* const& to_address(const P6&);
};

#if _CCCL_HAS_HOST_STD_LIB()
struct STDP6
{};

template <>
struct cuda::std::pointer_traits<STDP6>
{
  TEST_FUNC static int const* const& to_address(const STDP6&);
};
#endif // _CCCL_HAS_HOST_STD_LIB()

// Taken from a build breakage caused in Clang
namespace P7
{
template <typename T>
struct CanProxy;
template <typename T>
struct CanQual
{
  TEST_FUNC CanProxy<T> operator->() const
  {
    return CanProxy<T>();
  }
};
template <typename T>
struct CanProxy
{
  TEST_FUNC const CanProxy<T>* operator->() const
  {
    return nullptr;
  }
};
} // namespace P7

namespace P8
{
template <class T>
struct FancyPtrA
{
  using element_type = Irrelevant;
  T* p_;
  TEST_FUNC constexpr FancyPtrA(T* p)
      : p_(p)
  {}
  TEST_FUNC T& operator*() const;
  TEST_FUNC constexpr T* operator->() const
  {
    return p_;
  }
};
template <class T>
struct FancyPtrB
{
  T* p_;
  TEST_FUNC constexpr FancyPtrB(T* p)
      : p_(p)
  {}
  TEST_FUNC T& operator*() const;
};
} // namespace P8

template <class T>
struct cuda::std::pointer_traits<P8::FancyPtrB<T>>
{
  TEST_FUNC static constexpr T* to_address(const P8::FancyPtrB<T>& p)
  {
    return p.p_;
  }
};

struct Incomplete;
template <class T>
struct Holder
{
  T t;
};

TEST_FUNC constexpr bool test()
{
  int i = 0;
  static_assert(noexcept(cuda::std::to_address(&i)));
  assert(cuda::std::to_address(&i) == &i);
  P1 p1(&i);
  static_assert(noexcept(cuda::std::to_address(p1)));
  assert(cuda::std::to_address(p1) == &i);
  P2 p2(&i);
  static_assert(noexcept(cuda::std::to_address(p2)));
  assert(cuda::std::to_address(p2) == &i);
  P3 p3(&i);
  static_assert(noexcept(cuda::std::to_address(p3)));
  assert(cuda::std::to_address(p3) == &i);
  P4 p4(&i);
  static_assert(noexcept(cuda::std::to_address(p4)));
  assert(cuda::std::to_address(p4) == &i);

  static_assert(cuda::std::is_same_v<decltype(cuda::std::to_address(cuda::std::declval<int const*>())), int const*>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::to_address(cuda::std::declval<P5>())), int const*>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::to_address(cuda::std::declval<P6>())), int const*>);

  P7::CanQual<int>* p7 = nullptr;
  assert(cuda::std::to_address(p7) == nullptr);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::to_address(p7)), P7::CanQual<int>*>);

  Holder<Incomplete>* p8_nil            = nullptr; // for C++03 compatibility
  P8::FancyPtrA<Holder<Incomplete>> p8a = p8_nil;
  assert(cuda::std::to_address(p8a) == p8_nil);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::to_address(p8a)), decltype(p8_nil)>);

  P8::FancyPtrB<Holder<Incomplete>> p8b = p8_nil;
  assert(cuda::std::to_address(p8b) == p8_nil);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::to_address(p8b)), decltype(p8_nil)>);

  int p9[2] = {};
  assert(cuda::std::to_address(p9) == p9);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::to_address(p9)), int*>);

  const int p10[2] = {};
  assert(cuda::std::to_address(p10) == p10);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::to_address(p10)), const int*>);

  int (*p11)() = nullptr;
  assert(cuda::std::to_address(&p11) == &p11);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::to_address(&p11)), int (**)()>);

  // See https://llvm.org/PR67449
  {
    struct S
    {};
    S* p = nullptr;
    assert(cuda::std::to_address<S>(p) == p);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::to_address<S>(p)), S*>);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
