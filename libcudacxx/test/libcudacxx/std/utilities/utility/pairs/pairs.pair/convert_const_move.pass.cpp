//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <utility>

// template <class T1, class T2> struct pair

// template <class U, class V> pair(const pair<U, V>&& p);

#include <cuda/std/cassert>
#include <cuda/std/utility>

#include "archetypes.h"
#include "test_convertible.h"
#include "test_macros.h"
using namespace ImplicitTypes; // Get implicitly archetypes

template <class T1, class U1, bool CanCopy = true, bool CanConvert = CanCopy>
TEST_FUNC constexpr void test_pair_rv()
{
  using P1  = cuda::std::pair<T1, int>;
  using P2  = cuda::std::pair<int, T1>;
  using UP1 = const cuda::std::pair<U1, int>&&;
  using UP2 = const cuda::std::pair<int, U1>&&;
  static_assert(cuda::std::is_constructible_v<P1, UP1> == CanCopy);
  static_assert(test_convertible<P1, UP1>() == CanConvert);
  static_assert(cuda::std::is_constructible_v<P2, UP2> == CanCopy);
  static_assert(test_convertible<P2, UP2>() == CanConvert);
}

template <class T, class U>
struct DPair : public cuda::std::pair<T, U>
{
  using Base = cuda::std::pair<T, U>;
  using Base::Base;
};

struct ExplicitT
{
  TEST_FUNC constexpr explicit ExplicitT(int x)
      : value(x)
  {}
  int value;
};

struct ImplicitT
{
  TEST_FUNC constexpr ImplicitT(int x)
      : value(x)
  {}
  int value;
};

// Move construction poisons the source. Copy construction leaves it unchanged.
struct MoveSensitive
{
  int value;

  TEST_FUNC constexpr MoveSensitive(int v)
      : value(v)
  {}
  TEST_FUNC constexpr MoveSensitive(const MoveSensitive& other) noexcept
      : value(other.value)
  {}
  TEST_FUNC constexpr MoveSensitive(MoveSensitive&& other)
      : value(other.value)
  {
    other.value = -1;
  }
};

// Explicit element constructors select the explicit pair(const pair<U, V>&&) overload.
struct ExplicitMoveSensitive
{
  int value;

  TEST_FUNC constexpr explicit ExplicitMoveSensitive(int v)
      : value(v)
  {}
  TEST_FUNC constexpr explicit ExplicitMoveSensitive(const ExplicitMoveSensitive& other) noexcept
      : value(other.value)
  {}
  TEST_FUNC constexpr explicit ExplicitMoveSensitive(ExplicitMoveSensitive&& other)
      : value(other.value)
  {
    other.value = -1;
  }
};

// A deleted move constructor makes pair(const pair<U, V>&&) ill-formed if reference elements are moved.
struct CopyOnlyElement
{
  int value;

  TEST_FUNC constexpr CopyOnlyElement(int v)
      : value(v)
  {}
  TEST_FUNC constexpr CopyOnlyElement(const CopyOnlyElement& other) noexcept
      : value(other.value)
  {}
  TEST_FUNC CopyOnlyElement(CopyOnlyElement&&) = delete;
};

static_assert(cuda::std::is_nothrow_copy_constructible_v<MoveSensitive>);
static_assert(!cuda::std::is_nothrow_move_constructible_v<MoveSensitive>);
static_assert(cuda::std::is_nothrow_copy_constructible_v<ExplicitMoveSensitive>);
static_assert(!cuda::std::is_nothrow_move_constructible_v<ExplicitMoveSensitive>);
static_assert(test_convertible<cuda::std::pair<MoveSensitive, MoveSensitive>,
                               const cuda::std::pair<MoveSensitive&, MoveSensitive&>&&>());
static_assert(!test_convertible<cuda::std::pair<ExplicitMoveSensitive, ExplicitMoveSensitive>,
                                const cuda::std::pair<ExplicitMoveSensitive&, ExplicitMoveSensitive&>&&>());

// pair(const pair<U, V>&&) forwards each element with forward<const U>, so an lvalue-reference element is copied.
template <class T>
TEST_FUNC constexpr void test_lvalue_ref_source()
{
  T first(1);
  T second(2);
  const cuda::std::pair<T&, T&> source(first, second);
  static_assert(cuda::std::is_nothrow_constructible_v<cuda::std::pair<T, T>, const cuda::std::pair<T&, T&>&&>);
  cuda::std::pair<T, T> dest(cuda::std::move(source));
  assert(dest.first.value == 1);
  assert(dest.second.value == 2);
  assert(first.value == 1);
  assert(second.value == 2);
}

// An rvalue-reference element stays an xvalue after forward<const U&&> and is moved.
template <class T>
TEST_FUNC constexpr void test_rvalue_ref_source()
{
  T first(1);
  T second(2);
  const cuda::std::pair<T&&, T&&> source(cuda::std::move(first), cuda::std::move(second));
  cuda::std::pair<T, T> dest(cuda::std::move(source));
  assert(dest.first.value == 1);
  assert(dest.second.value == 2);
  assert(first.value == -1);
  assert(second.value == -1);
}

// Each element is forwarded independently. A const value element is copied.
template <class T>
TEST_FUNC constexpr void test_mixed_ref_source()
{
  {
    T referenced(1);
    const cuda::std::pair<T&, T> source(referenced, T(2));
    cuda::std::pair<T, T> dest(cuda::std::move(source));
    assert(dest.first.value == 1);
    assert(dest.second.value == 2);
    assert(referenced.value == 1);
    assert(source.second.value == 2);
  }
  {
    T referenced(2);
    const cuda::std::pair<T, T&> source(T(1), referenced);
    cuda::std::pair<T, T> dest(cuda::std::move(source));
    assert(dest.first.value == 1);
    assert(dest.second.value == 2);
    assert(source.first.value == 1);
    assert(referenced.value == 2);
  }
}

TEST_FUNC constexpr void test_copy_only_ref_source()
{
  CopyOnlyElement first(1);
  CopyOnlyElement second(2);
  const cuda::std::pair<CopyOnlyElement&, CopyOnlyElement&> source(first, second);
  static_assert(cuda::std::is_nothrow_constructible_v<cuda::std::pair<CopyOnlyElement, CopyOnlyElement>,
                                                      const cuda::std::pair<CopyOnlyElement&, CopyOnlyElement&>&&>);
  cuda::std::pair<CopyOnlyElement, CopyOnlyElement> dest(cuda::std::move(source));
  assert(dest.first.value == 1);
  assert(dest.second.value == 2);
  assert(first.value == 1);
  assert(second.value == 2);
}

TEST_FUNC constexpr bool test()
{
  {
    // We allow derived types to use this constructor
    using P1 = DPair<long, long>;
    using P2 = cuda::std::pair<int, int>;
    const P1 p1(42, 101);
    const P2 p2(cuda::std::move(p1));
    assert(p2.first == 42);
    assert(p2.second == 101);
  }
  {
    const cuda::std::pair<int, int> p1(42, 43);
    const cuda::std::pair<ExplicitT, ExplicitT> p2(cuda::std::move(p1));
    assert(p2.first.value == 42);
    assert(p2.second.value == 43);
  }
  {
    const cuda::std::pair<int, int> p1(42, 43);
    const cuda::std::pair<ImplicitT, ImplicitT> p2 = cuda::std::move(p1);
    assert(p2.first.value == 42);
    assert(p2.second.value == 43);
  }
  {
    test_pair_rv<AllCtors, AllCtors>();
    test_pair_rv<AllCtors, AllCtors&>();
    test_pair_rv<AllCtors, AllCtors&&>();
    test_pair_rv<AllCtors, const AllCtors&>();
    test_pair_rv<AllCtors, const AllCtors&&>();

    test_pair_rv<ExplicitTypes::AllCtors, ExplicitTypes::AllCtors>();
    test_pair_rv<ExplicitTypes::AllCtors, ExplicitTypes::AllCtors&, true, false>();
    test_pair_rv<ExplicitTypes::AllCtors, ExplicitTypes::AllCtors&&, true, false>();
    test_pair_rv<ExplicitTypes::AllCtors, const ExplicitTypes::AllCtors&, true, false>();
    test_pair_rv<ExplicitTypes::AllCtors, const ExplicitTypes::AllCtors&&, true, false>();

    // const pair<MoveOnly, int>&& forwards const MoveOnly&&, which cannot bind to the move constructor.
    test_pair_rv<MoveOnly, MoveOnly, false>();
    test_pair_rv<MoveOnly, MoveOnly&, false>();
    test_pair_rv<MoveOnly, MoveOnly&&>();

    test_pair_rv<ExplicitTypes::MoveOnly, ExplicitTypes::MoveOnly, false>();
    test_pair_rv<ExplicitTypes::MoveOnly, ExplicitTypes::MoveOnly&, false>();
    test_pair_rv<ExplicitTypes::MoveOnly, ExplicitTypes::MoveOnly&&, true, false>();

    test_pair_rv<CopyOnly, CopyOnly>();
    test_pair_rv<CopyOnly, CopyOnly&>();
    test_pair_rv<CopyOnly, CopyOnly&&>();

    test_pair_rv<ExplicitTypes::CopyOnly, ExplicitTypes::CopyOnly>();
    test_pair_rv<ExplicitTypes::CopyOnly, ExplicitTypes::CopyOnly&, true, false>();
    test_pair_rv<ExplicitTypes::CopyOnly, ExplicitTypes::CopyOnly&&, true, false>();

    test_pair_rv<NonCopyable, NonCopyable, false>();
    test_pair_rv<NonCopyable, NonCopyable&, false>();
    test_pair_rv<NonCopyable, NonCopyable&&, false>();
    test_pair_rv<NonCopyable, const NonCopyable&, false>();
    test_pair_rv<NonCopyable, const NonCopyable&&, false>();
  }
  { // Test construction of references
    test_pair_rv<NonCopyable&, NonCopyable&>();
    test_pair_rv<NonCopyable&, NonCopyable&&>();
    test_pair_rv<NonCopyable&, NonCopyable const&, false>();
    test_pair_rv<NonCopyable const&, NonCopyable&&>();
    // const pair<NonCopyable&&, int>&& forwards the reference element as an rvalue.
    test_pair_rv<NonCopyable&&, NonCopyable&&>();

    test_pair_rv<ConvertingType&, int, false>();
    test_pair_rv<ExplicitTypes::ConvertingType&, int, false>();
#if defined(_CCCL_BUILTIN_REFERENCE_CONSTRUCTS_FROM_TEMPORARY)
    // Constructing a reference element from a temporary is deleted.
    test_pair_rv<ConvertingType&&, int, false>();
    test_pair_rv<ConvertingType const&, int, false>();
    test_pair_rv<ConvertingType const&&, int, false>();
#endif // _CCCL_BUILTIN_REFERENCE_CONSTRUCTS_FROM_TEMPORARY
    test_pair_rv<ExplicitTypes::ConvertingType&&, int, false>();
    test_pair_rv<ExplicitTypes::ConvertingType const&, int, false>();
    test_pair_rv<ExplicitTypes::ConvertingType const&&, int, false>();
  }
  {
    test_pair_rv<AllCtors, int, false>();
    test_pair_rv<ExplicitTypes::AllCtors, int, false>();
    test_pair_rv<ConvertingType, int>();
    test_pair_rv<ExplicitTypes::ConvertingType, int, true, false>();

    test_pair_rv<ConvertingType, int>();
    test_pair_rv<ConvertingType, ConvertingType>();
    test_pair_rv<ConvertingType, ConvertingType const&>();
    test_pair_rv<ConvertingType, ConvertingType&>();
    test_pair_rv<ConvertingType, ConvertingType&&>();

    test_pair_rv<ExplicitTypes::ConvertingType, int, true, false>();
    test_pair_rv<ExplicitTypes::ConvertingType, int&, true, false>();
    test_pair_rv<ExplicitTypes::ConvertingType, const int&, true, false>();
    test_pair_rv<ExplicitTypes::ConvertingType, int&&, true, false>();
    test_pair_rv<ExplicitTypes::ConvertingType, const int&&, true, false>();

    test_pair_rv<ExplicitTypes::ConvertingType, ExplicitTypes::ConvertingType>();
    test_pair_rv<ExplicitTypes::ConvertingType, ExplicitTypes::ConvertingType const&, true, false>();
    test_pair_rv<ExplicitTypes::ConvertingType, ExplicitTypes::ConvertingType&, true, false>();
    test_pair_rv<ExplicitTypes::ConvertingType, ExplicitTypes::ConvertingType&&, true, false>();
  }
  test_lvalue_ref_source<MoveSensitive>();
  test_lvalue_ref_source<ExplicitMoveSensitive>();
  test_rvalue_ref_source<MoveSensitive>();
  test_rvalue_ref_source<ExplicitMoveSensitive>();
  test_mixed_ref_source<MoveSensitive>();
  test_mixed_ref_source<ExplicitMoveSensitive>();
  test_copy_only_ref_source();
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
