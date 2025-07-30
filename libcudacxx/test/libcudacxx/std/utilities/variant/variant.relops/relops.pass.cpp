//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: msvc-19.16
// UNSUPPORTED: clang-7, clang-8

// <cuda/std/variant>

// template <class ...Types>
// constexpr bool
// operator==(variant<Types...> const&, variant<Types...> const&) noexcept;
//
// template <class ...Types>
// constexpr bool
// operator!=(variant<Types...> const&, variant<Types...> const&) noexcept;
//
// template <class ...Types>
// constexpr bool
// operator<(variant<Types...> const&, variant<Types...> const&) noexcept;
//
// template <class ...Types>
// constexpr bool
// operator>(variant<Types...> const&, variant<Types...> const&) noexcept;
//
// template <class ...Types>
// constexpr bool
// operator<=(variant<Types...> const&, variant<Types...> const&) noexcept;
//
// template <class ...Types>
// constexpr bool
// operator>=(variant<Types...> const&, variant<Types...> const&) noexcept;

#include <cuda/std/cassert>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/std/variant>

#include "test_macros.h"

#if TEST_HAS_EXCEPTIONS()
struct MakeEmptyT
{
  MakeEmptyT() = default;
  MakeEmptyT(MakeEmptyT&&)
  {
    throw 42;
  }
  MakeEmptyT& operator=(MakeEmptyT&&)
  {
    throw 42;
  }
};
inline bool operator==(const MakeEmptyT&, const MakeEmptyT&)
{
  assert(false);
  return false;
}
inline bool operator!=(const MakeEmptyT&, const MakeEmptyT&)
{
  assert(false);
  return false;
}
inline bool operator<(const MakeEmptyT&, const MakeEmptyT&)
{
  assert(false);
  return false;
}
inline bool operator<=(const MakeEmptyT&, const MakeEmptyT&)
{
  assert(false);
  return false;
}
inline bool operator>(const MakeEmptyT&, const MakeEmptyT&)
{
  assert(false);
  return false;
}
inline bool operator>=(const MakeEmptyT&, const MakeEmptyT&)
{
  assert(false);
  return false;
}

template <class Variant>
void makeEmpty(Variant& v)
{
  Variant v2(cuda::std::in_place_type<MakeEmptyT>);
  try
  {
    v = cuda::std::move(v2);
    assert(false);
  }
  catch (...)
  {
    assert(v.valueless_by_exception());
  }
}
#endif // TEST_HAS_EXCEPTIONS()

struct MyBool
{
  bool value;
  __host__ __device__ constexpr explicit MyBool(bool v)
      : value(v)
  {}
  __host__ __device__ constexpr operator bool() const noexcept
  {
    return value;
  }
};

struct ComparesToMyBool
{
  int value = 0;
};
__host__ __device__ inline constexpr MyBool operator==(const ComparesToMyBool& LHS, const ComparesToMyBool& RHS) noexcept
{
  return MyBool(LHS.value == RHS.value);
}
__host__ __device__ inline constexpr MyBool operator!=(const ComparesToMyBool& LHS, const ComparesToMyBool& RHS) noexcept
{
  return MyBool(LHS.value != RHS.value);
}
__host__ __device__ inline constexpr MyBool operator<(const ComparesToMyBool& LHS, const ComparesToMyBool& RHS) noexcept
{
  return MyBool(LHS.value < RHS.value);
}
__host__ __device__ inline constexpr MyBool operator<=(const ComparesToMyBool& LHS, const ComparesToMyBool& RHS) noexcept
{
  return MyBool(LHS.value <= RHS.value);
}
__host__ __device__ inline constexpr MyBool operator>(const ComparesToMyBool& LHS, const ComparesToMyBool& RHS) noexcept
{
  return MyBool(LHS.value > RHS.value);
}
__host__ __device__ inline constexpr MyBool operator>=(const ComparesToMyBool& LHS, const ComparesToMyBool& RHS) noexcept
{
  return MyBool(LHS.value >= RHS.value);
}

template <class T1, class T2>
__host__ __device__ void test_equality_basic()
{
  {
    using V = cuda::std::variant<T1, T2>;
    constexpr V v1(cuda::std::in_place_index<0>, T1{42});
    constexpr V v2(cuda::std::in_place_index<0>, T1{42});
    static_assert(v1 == v2, "");
    static_assert(v2 == v1, "");
    static_assert(!(v1 != v2), "");
    static_assert(!(v2 != v1), "");
  }
  {
    using V = cuda::std::variant<T1, T2>;
    constexpr V v1(cuda::std::in_place_index<0>, T1{42});
    constexpr V v2(cuda::std::in_place_index<0>, T1{43});
    static_assert(!(v1 == v2), "");
    static_assert(!(v2 == v1), "");
    static_assert(v1 != v2, "");
    static_assert(v2 != v1, "");
  }
  {
    using V = cuda::std::variant<T1, T2>;
    constexpr V v1(cuda::std::in_place_index<0>, T1{42});
    constexpr V v2(cuda::std::in_place_index<1>, T2{42});
    static_assert(!(v1 == v2), "");
    static_assert(!(v2 == v1), "");
    static_assert(v1 != v2, "");
    static_assert(v2 != v1, "");
  }
  {
    using V = cuda::std::variant<T1, T2>;
    constexpr V v1(cuda::std::in_place_index<1>, T2{42});
    constexpr V v2(cuda::std::in_place_index<1>, T2{42});
    static_assert(v1 == v2, "");
    static_assert(v2 == v1, "");
    static_assert(!(v1 != v2), "");
    static_assert(!(v2 != v1), "");
  }
}

__host__ __device__ void test_equality()
{
  test_equality_basic<int, long>();
  test_equality_basic<ComparesToMyBool, int>();
  test_equality_basic<int, ComparesToMyBool>();
  test_equality_basic<ComparesToMyBool, ComparesToMyBool>();
}

#if TEST_HAS_EXCEPTIONS()
void test_exceptions_equality()
{
  {
    using V = cuda::std::variant<int, MakeEmptyT>;
    V v1;
    V v2;
    makeEmpty(v2);
    assert(!(v1 == v2));
    assert(!(v2 == v1));
    assert(v1 != v2);
    assert(v2 != v1);
  }
  {
    using V = cuda::std::variant<int, MakeEmptyT>;
    V v1;
    makeEmpty(v1);
    V v2;
    assert(!(v1 == v2));
    assert(!(v2 == v1));
    assert(v1 != v2);
    assert(v2 != v1);
  }
  {
    using V = cuda::std::variant<int, MakeEmptyT>;
    V v1;
    makeEmpty(v1);
    V v2;
    makeEmpty(v2);
    assert(v1 == v2);
    assert(v2 == v1);
    assert(!(v1 != v2));
    assert(!(v2 != v1));
  }
}
#endif // TEST_HAS_EXCEPTIONS()

template <class Var>
__host__ __device__ constexpr bool test_less(const Var& l, const Var& r, bool expect_less, bool expect_greater)
{
  static_assert(cuda::std::is_same_v<decltype(l < r), bool>, "");
  static_assert(cuda::std::is_same_v<decltype(l <= r), bool>, "");
  static_assert(cuda::std::is_same_v<decltype(l > r), bool>, "");
  static_assert(cuda::std::is_same_v<decltype(l >= r), bool>, "");

  return ((l < r) == expect_less) && (!(l >= r) == expect_less) && ((l > r) == expect_greater)
      && (!(l <= r) == expect_greater);
}

template <class T1, class T2>
__host__ __device__ void test_relational_basic()
{
  { // same index, same value
    using V = cuda::std::variant<T1, T2>;
    constexpr V v1(cuda::std::in_place_index<0>, T1{1});
    constexpr V v2(cuda::std::in_place_index<0>, T1{1});
    static_assert(test_less(v1, v2, false, false), "");
  }
  { // same index, value < other_value
    using V = cuda::std::variant<T1, T2>;
    constexpr V v1(cuda::std::in_place_index<0>, T1{0});
    constexpr V v2(cuda::std::in_place_index<0>, T1{1});
    static_assert(test_less(v1, v2, true, false), "");
  }
  { // same index, value > other_value
    using V = cuda::std::variant<T1, T2>;
    constexpr V v1(cuda::std::in_place_index<0>, T1{1});
    constexpr V v2(cuda::std::in_place_index<0>, T1{0});
    static_assert(test_less(v1, v2, false, true), "");
  }
  { // LHS.index() < RHS.index()
    using V = cuda::std::variant<T1, T2>;
    constexpr V v1(cuda::std::in_place_index<0>, T1{0});
    constexpr V v2(cuda::std::in_place_index<1>, T2{0});
    static_assert(test_less(v1, v2, true, false), "");
  }
  { // LHS.index() > RHS.index()
    using V = cuda::std::variant<T1, T2>;
    constexpr V v1(cuda::std::in_place_index<1>, T2{0});
    constexpr V v2(cuda::std::in_place_index<0>, T1{0});
    static_assert(test_less(v1, v2, false, true), "");
  }
}

__host__ __device__ void test_relational()
{
  test_relational_basic<int, long>();
  test_relational_basic<ComparesToMyBool, int>();
  test_relational_basic<int, ComparesToMyBool>();
  test_relational_basic<ComparesToMyBool, ComparesToMyBool>();
}

#if TEST_HAS_EXCEPTIONS()
void test_exceptions_relational()
{
  { // LHS.index() < RHS.index(), RHS is empty
    using V = cuda::std::variant<int, MakeEmptyT>;
    V v1;
    V v2;
    makeEmpty(v2);
    assert(test_less(v1, v2, false, true));
  }
  { // LHS.index() > RHS.index(), LHS is empty
    using V = cuda::std::variant<int, MakeEmptyT>;
    V v1;
    makeEmpty(v1);
    V v2;
    assert(test_less(v1, v2, true, false));
  }
  { // LHS.index() == RHS.index(), LHS and RHS are empty
    using V = cuda::std::variant<int, MakeEmptyT>;
    V v1;
    makeEmpty(v1);
    V v2;
    makeEmpty(v2);
    assert(test_less(v1, v2, false, false));
  }
}
#endif

int main(int, char**)
{
  test_equality();
  test_relational();

#if TEST_HAS_EXCEPTIONS()
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions_equality();))
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions_relational();))
#endif // TEST_HAS_EXCEPTIONS()

  return 0;
}
