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

// template <size_t I, class... Types>
//   constexpr variant_alternative_t<I, variant<Types...>>&
//   get(variant<Types...>& v);
// template <size_t I, class... Types>
//   constexpr variant_alternative_t<I, variant<Types...>>&&
//   get(variant<Types...>&& v);
// template <size_t I, class... Types>
//   constexpr variant_alternative_t<I, variant<Types...>> const& get(const
//   variant<Types...>& v);
// template <size_t I, class... Types>
//  constexpr variant_alternative_t<I, variant<Types...>> const&& get(const
//  variant<Types...>&& v);

#include <cuda/std/cassert>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/std/variant>

#include "test_macros.h"
#include "test_workarounds.h"
#include "variant_test_helpers.h"

__host__ __device__ void test_const_lvalue_get()
{
  {
    using V = cuda::std::variant<int, const long>;
    constexpr V v(42);
#if !TEST_COMPILER(MSVC) && !TEST_COMPILER(GCC, <, 9)
    static_assert(!noexcept(cuda::std::get<0>(v)));
#endif // !TEST_COMPILER(MSVC) && !TEST_COMPILER(GCC, <, 9)
    static_assert(cuda::std::is_same_v<decltype(cuda::std::get<0>(v)), const int&>);
    static_assert(cuda::std::get<0>(v) == 42, "");
  }
  {
    using V = cuda::std::variant<int, const long>;
    const V v(42);
    static_assert(!noexcept(cuda::std::get<0>(v)));
    static_assert(cuda::std::is_same_v<decltype(cuda::std::get<0>(v)), const int&>);
    assert(cuda::std::get<0>(v) == 42);
  }
  {
    using V = cuda::std::variant<int, const long>;
    constexpr V v(42l);
#if !TEST_COMPILER(MSVC) && !TEST_COMPILER(GCC, <, 9)
    static_assert(!noexcept(cuda::std::get<1>(v)));
#endif // !TEST_COMPILER(MSVC) && !TEST_COMPILER(GCC, <, 9)
    static_assert(cuda::std::is_same_v<decltype(cuda::std::get<1>(v)), const long&>);
    static_assert(cuda::std::get<1>(v) == 42, "");
  }
  {
    using V = cuda::std::variant<int, const long>;
    const V v(42l);
    static_assert(!noexcept(cuda::std::get<1>(v)));
    static_assert(cuda::std::is_same_v<decltype(cuda::std::get<1>(v)), const long&>);
    assert(cuda::std::get<1>(v) == 42);
  }
// FIXME: Remove these once reference support is reinstated
#if !defined(TEST_VARIANT_HAS_NO_REFERENCES)
  {
    using V = cuda::std::variant<int&>;
    int x   = 42;
    const V v(x);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::get<0>(v)), int&>);
    assert(&cuda::std::get<0>(v) == &x);
  }
  {
    using V = cuda::std::variant<int&&>;
    int x   = 42;
    const V v(cuda::std::move(x));
    static_assert(cuda::std::is_same_v<decltype(cuda::std::get<0>(v)), int&>);
    assert(&cuda::std::get<0>(v) == &x);
  }
  {
    using V = cuda::std::variant<const int&&>;
    int x   = 42;
    const V v(cuda::std::move(x));
    static_assert(cuda::std::is_same_v<decltype(cuda::std::get<0>(v)), const int&>);
    assert(&cuda::std::get<0>(v) == &x);
  }
#endif
}

__host__ __device__ void test_lvalue_get()
{
  {
    using V = cuda::std::variant<int, const long>;
    V v(42);
    static_assert(!noexcept(cuda::std::get<0>(v)));
    static_assert(cuda::std::is_same_v<decltype(cuda::std::get<0>(v)), int&>);
    assert(cuda::std::get<0>(v) == 42);
  }
  {
    using V = cuda::std::variant<int, const long>;
    V v(42l);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::get<1>(v)), const long&>);
    assert(cuda::std::get<1>(v) == 42);
  }
// FIXME: Remove these once reference support is reinstated
#if !defined(TEST_VARIANT_HAS_NO_REFERENCES)
  {
    using V = cuda::std::variant<int&>;
    int x   = 42;
    V v(x);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::get<0>(v)), int&>);
    assert(&cuda::std::get<0>(v) == &x);
  }
  {
    using V = cuda::std::variant<const int&>;
    int x   = 42;
    V v(x);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::get<0>(v)), const int&>);
    assert(&cuda::std::get<0>(v) == &x);
  }
  {
    using V = cuda::std::variant<int&&>;
    int x   = 42;
    V v(cuda::std::move(x));
    static_assert(cuda::std::is_same_v<decltype(cuda::std::get<0>(v)), int&>);
    assert(&cuda::std::get<0>(v) == &x);
  }
  {
    using V = cuda::std::variant<const int&&>;
    int x   = 42;
    V v(cuda::std::move(x));
    static_assert(cuda::std::is_same_v<decltype(cuda::std::get<0>(v)), const int&>);
    assert(&cuda::std::get<0>(v) == &x);
  }
#endif
}

__host__ __device__ void test_rvalue_get()
{
  {
    using V = cuda::std::variant<int, const long>;
    V v(42);
    static_assert(!noexcept(cuda::std::get<0>(cuda::std::move(v))));
    static_assert(cuda::std::is_same_v<decltype(cuda::std::get<0>(cuda::std::move(v))), int&&>);
    assert(cuda::std::get<0>(cuda::std::move(v)) == 42);
  }
  {
    using V = cuda::std::variant<int, const long>;
    V v(42l);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::get<1>(cuda::std::move(v))), const long&&>);
    assert(cuda::std::get<1>(cuda::std::move(v)) == 42);
  }
// FIXME: Remove these once reference support is reinstated
#if !defined(TEST_VARIANT_HAS_NO_REFERENCES)
  {
    using V = cuda::std::variant<int&>;
    int x   = 42;
    V v(x);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::get<0>(cuda::std::move(v))), int&>);
    assert(&cuda::std::get<0>(cuda::std::move(v)) == &x);
  }
  {
    using V = cuda::std::variant<const int&>;
    int x   = 42;
    V v(x);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::get<0>(cuda::std::move(v))), const int&>);
    assert(&cuda::std::get<0>(cuda::std::move(v)) == &x);
  }
  {
    using V = cuda::std::variant<int&&>;
    int x   = 42;
    V v(cuda::std::move(x));
    static_assert(cuda::std::is_same_v<decltype(cuda::std::get<0>(cuda::std::move(v))), int&&>);
    int&& xref = cuda::std::get<0>(cuda::std::move(v));
    assert(&xref == &x);
  }
  {
    using V = cuda::std::variant<const int&&>;
    int x   = 42;
    V v(cuda::std::move(x));
    static_assert(cuda::std::is_same_v<decltype(cuda::std::get<0>(cuda::std::move(v))), const int&&>);
    const int&& xref = cuda::std::get<0>(cuda::std::move(v));
    assert(&xref == &x);
  }
#endif
}

__host__ __device__ void test_const_rvalue_get()
{
  {
    using V = cuda::std::variant<int, const long>;
    const V v(42);
    static_assert(!noexcept(cuda::std::get<0>(cuda::std::move(v))));
    static_assert(cuda::std::is_same_v<decltype(cuda::std::get<0>(cuda::std::move(v))), const int&&>);
    assert(cuda::std::get<0>(cuda::std::move(v)) == 42);
  }
  {
    using V = cuda::std::variant<int, const long>;
    const V v(42l);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::get<1>(cuda::std::move(v))), const long&&>);
    assert(cuda::std::get<1>(cuda::std::move(v)) == 42);
  }
// FIXME: Remove these once reference support is reinstated
#if !defined(TEST_VARIANT_HAS_NO_REFERENCES)
  {
    using V = cuda::std::variant<int&>;
    int x   = 42;
    const V v(x);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::get<0>(cuda::std::move(v))), int&>);
    assert(&cuda::std::get<0>(cuda::std::move(v)) == &x);
  }
  {
    using V = cuda::std::variant<const int&>;
    int x   = 42;
    const V v(x);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::get<0>(cuda::std::move(v))), const int&>);
    assert(&cuda::std::get<0>(cuda::std::move(v)) == &x);
  }
  {
    using V = cuda::std::variant<int&&>;
    int x   = 42;
    const V v(cuda::std::move(x));
    static_assert(cuda::std::is_same_v<decltype(cuda::std::get<0>(cuda::std::move(v))), int&&>);
    int&& xref = cuda::std::get<0>(cuda::std::move(v));
    assert(&xref == &x);
  }
  {
    using V = cuda::std::variant<const int&&>;
    int x   = 42;
    const V v(cuda::std::move(x));
    static_assert(cuda::std::is_same_v<decltype(cuda::std::get<0>(cuda::std::move(v))), const int&&>);
    const int&& xref = cuda::std::get<0>(cuda::std::move(v));
    assert(&xref == &x);
  }
#endif
}

template <cuda::std::size_t I>
using Idx = cuda::std::integral_constant<size_t, I>;

#if TEST_HAS_EXCEPTIONS()
void test_throws_for_all_value_categories()
{
  using V = cuda::std::variant<int, long>;
  V v0(42);
  const V& cv0 = v0;
  assert(v0.index() == 0);
  V v1(42l);
  const V& cv1 = v1;
  assert(v1.index() == 1);
  cuda::std::integral_constant<size_t, 0> zero;
  cuda::std::integral_constant<size_t, 1> one;
  auto test = [](auto idx, auto&& v) {
    using Idx = decltype(idx);
    try
    {
      TEST_IGNORE_NODISCARD cuda::std::get<Idx::value>(cuda::std::forward<decltype(v)>(v));
    }
    catch (const cuda::std::bad_variant_access&)
    {
      return true;
    }
    catch (...)
    { /* ... */
    }
    return false;
  };
  { // lvalue test cases
    assert(test(one, v0));
    assert(test(zero, v1));
  }
  { // const lvalue test cases
    assert(test(one, cv0));
    assert(test(zero, cv1));
  }
  { // rvalue test cases
    assert(test(one, cuda::std::move(v0)));
    assert(test(zero, cuda::std::move(v1)));
  }
  { // const rvalue test cases
    assert(test(one, cuda::std::move(cv0)));
    assert(test(zero, cuda::std::move(cv1)));
  }
}
#endif // TEST_HAS_EXCEPTIONS()

int main(int, char**)
{
  test_const_lvalue_get();
  test_lvalue_get();
  test_rvalue_get();
  test_const_rvalue_get();

#if TEST_HAS_EXCEPTIONS()
  NV_IF_TARGET(NV_IS_HOST, (test_throws_for_all_value_categories();))
#endif // TEST_HAS_EXCEPTIONS()

  return 0;
}
