//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// UNSUPPORTED: msvc-19.16

// <cuda/std/variant>

// template <class T, class... Types> constexpr T& get(variant<Types...>& v);
// template <class T, class... Types> constexpr T&& get(variant<Types...>&& v);
// template <class T, class... Types> constexpr const T& get(const
// variant<Types...>& v);
// template <class T, class... Types> constexpr const T&& get(const
// variant<Types...>&& v);

#include <cuda/std/cassert>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/std/variant>

#include "test_macros.h"
#include "test_workarounds.h"
#include "variant_test_helpers.h"

TEST_HOST_DEVICE void test_const_lvalue_get() {
  {
    using V = cuda::std::variant<int, const long>;
    constexpr V v(42);
#if !defined(TEST_COMPILER_MSVC) &&                                            \
    !(defined(TEST_COMPILER_GCC) && __GNUC__ < 9) &&                           \
    !defined(TEST_COMPILER_CUDACC_BELOW_11_3) && !defined(TEST_COMPILER_ICC)
    ASSERT_NOT_NOEXCEPT(cuda::std::get<int>(v));
#endif // !TEST_COMPILER_MSVC && !TEST_COMPILER_GCC && TEST_COMPILER_CUDACC_BELOW_11_3 && !TEST_COMPILER_ICC
    ASSERT_SAME_TYPE(decltype(cuda::std::get<int>(v)), const int&);
    static_assert(cuda::std::get<int>(v) == 42, "");
  }
  {
    using V = cuda::std::variant<int, const long>;
    const V v(42);
#if !defined(TEST_COMPILER_ICC)
    ASSERT_NOT_NOEXCEPT(cuda::std::get<int>(v));
#endif // !TEST_COMPILER_ICC
    ASSERT_SAME_TYPE(decltype(cuda::std::get<int>(v)), const int&);
    assert(cuda::std::get<int>(v) == 42);
  }
  {
    using V = cuda::std::variant<int, const long>;
    constexpr V v(42l);
#if !defined(TEST_COMPILER_MSVC) &&                                            \
    !(defined(TEST_COMPILER_GCC) && __GNUC__ < 9) &&                           \
    !defined(TEST_COMPILER_CUDACC_BELOW_11_3) && !defined(TEST_COMPILER_ICC)
    ASSERT_NOT_NOEXCEPT(cuda::std::get<const long>(v));
#endif // !TEST_COMPILER_MSVC && !TEST_COMPILER_GCC && TEST_COMPILER_CUDACC_BELOW_11_3 && !TEST_COMPILER_ICC
    ASSERT_SAME_TYPE(decltype(cuda::std::get<const long>(v)), const long&);
    static_assert(cuda::std::get<const long>(v) == 42, "");
  }
  {
    using V = cuda::std::variant<int, const long>;
    const V v(42l);
#if !defined(TEST_COMPILER_ICC)
    ASSERT_NOT_NOEXCEPT(cuda::std::get<const long>(v));
#endif // !TEST_COMPILER_ICC
    ASSERT_SAME_TYPE(decltype(cuda::std::get<const long>(v)), const long&);
    assert(cuda::std::get<const long>(v) == 42);
  }
// FIXME: Remove these once reference support is reinstated
#if !defined(TEST_VARIANT_HAS_NO_REFERENCES)
  {
    using V = cuda::std::variant<int&>;
    int x = 42;
    const V v(x);
    ASSERT_SAME_TYPE(decltype(cuda::std::get<int&>(v)), int&);
    assert(&cuda::std::get<int&>(v) == &x);
  }
  {
    using V = cuda::std::variant<int&&>;
    int x = 42;
    const V v(cuda::std::move(x));
    ASSERT_SAME_TYPE(decltype(cuda::std::get<int&&>(v)), int&);
    assert(&cuda::std::get<int&&>(v) == &x);
  }
  {
    using V = cuda::std::variant<const int&&>;
    int x = 42;
    const V v(cuda::std::move(x));
    ASSERT_SAME_TYPE(decltype(cuda::std::get<const int&&>(v)), const int&);
    assert(&cuda::std::get<const int&&>(v) == &x);
  }
#endif
}

TEST_HOST_DEVICE void test_lvalue_get() {
  {
    using V = cuda::std::variant<int, const long>;
    V v(42);
#if !defined(TEST_COMPILER_ICC)
    ASSERT_NOT_NOEXCEPT(cuda::std::get<int>(v));
#endif // !TEST_COMPILER_ICC
    ASSERT_SAME_TYPE(decltype(cuda::std::get<int>(v)), int&);
    assert(cuda::std::get<int>(v) == 42);
  }
  {
    using V = cuda::std::variant<int, const long>;
    V v(42l);
    ASSERT_SAME_TYPE(decltype(cuda::std::get<const long>(v)), const long&);
    assert(cuda::std::get<const long>(v) == 42);
  }
// FIXME: Remove these once reference support is reinstated
#if !defined(TEST_VARIANT_HAS_NO_REFERENCES)
  {
    using V = cuda::std::variant<int&>;
    int x = 42;
    V v(x);
    ASSERT_SAME_TYPE(decltype(cuda::std::get<int&>(v)), int&);
    assert(&cuda::std::get<int&>(v) == &x);
  }
  {
    using V = cuda::std::variant<const int&>;
    int x = 42;
    V v(x);
    ASSERT_SAME_TYPE(decltype(cuda::std::get<const int&>(v)), const int&);
    assert(&cuda::std::get<const int&>(v) == &x);
  }
  {
    using V = cuda::std::variant<int&&>;
    int x = 42;
    V v(cuda::std::move(x));
    ASSERT_SAME_TYPE(decltype(cuda::std::get<int&&>(v)), int&);
    assert(&cuda::std::get<int&&>(v) == &x);
  }
  {
    using V = cuda::std::variant<const int&&>;
    int x = 42;
    V v(cuda::std::move(x));
    ASSERT_SAME_TYPE(decltype(cuda::std::get<const int&&>(v)), const int&);
    assert(&cuda::std::get<const int&&>(v) == &x);
  }
#endif
}

TEST_HOST_DEVICE void test_rvalue_get() {
  {
    using V = cuda::std::variant<int, const long>;
    V v(42);
#if !defined(TEST_COMPILER_ICC)
    ASSERT_NOT_NOEXCEPT(cuda::std::get<int>(cuda::std::move(v)));
#endif // !TEST_COMPILER_ICC
    ASSERT_SAME_TYPE(decltype(cuda::std::get<int>(cuda::std::move(v))), int&&);
    assert(cuda::std::get<int>(cuda::std::move(v)) == 42);
  }
  {
    using V = cuda::std::variant<int, const long>;
    V v(42l);
    ASSERT_SAME_TYPE(decltype(cuda::std::get<const long>(cuda::std::move(v))),
                     const long&&);
    assert(cuda::std::get<const long>(cuda::std::move(v)) == 42);
  }
// FIXME: Remove these once reference support is reinstated
#if !defined(TEST_VARIANT_HAS_NO_REFERENCES)
  {
    using V = cuda::std::variant<int&>;
    int x = 42;
    V v(x);
    ASSERT_SAME_TYPE(decltype(cuda::std::get<int&>(cuda::std::move(v))), int&);
    assert(&cuda::std::get<int&>(cuda::std::move(v)) == &x);
  }
  {
    using V = cuda::std::variant<const int&>;
    int x = 42;
    V v(x);
    ASSERT_SAME_TYPE(decltype(cuda::std::get<const int&>(cuda::std::move(v))),
                     const int&);
    assert(&cuda::std::get<const int&>(cuda::std::move(v)) == &x);
  }
  {
    using V = cuda::std::variant<int&&>;
    int x = 42;
    V v(cuda::std::move(x));
    ASSERT_SAME_TYPE(decltype(cuda::std::get<int&&>(cuda::std::move(v))),
                     int&&);
    int&& xref = cuda::std::get<int&&>(cuda::std::move(v));
    assert(&xref == &x);
  }
  {
    using V = cuda::std::variant<const int&&>;
    int x = 42;
    V v(cuda::std::move(x));
    ASSERT_SAME_TYPE(decltype(cuda::std::get<const int&&>(cuda::std::move(v))),
                     const int&&);
    const int&& xref = cuda::std::get<const int&&>(cuda::std::move(v));
    assert(&xref == &x);
  }
#endif
}

TEST_HOST_DEVICE void test_const_rvalue_get() {
  {
    using V = cuda::std::variant<int, const long>;
    const V v(42);
#if !defined(TEST_COMPILER_ICC)
    ASSERT_NOT_NOEXCEPT(cuda::std::get<int>(cuda::std::move(v)));
#endif // !TEST_COMPILER_ICC
    ASSERT_SAME_TYPE(decltype(cuda::std::get<int>(cuda::std::move(v))),
                     const int&&);
    assert(cuda::std::get<int>(cuda::std::move(v)) == 42);
  }
  {
    using V = cuda::std::variant<int, const long>;
    const V v(42l);
    ASSERT_SAME_TYPE(decltype(cuda::std::get<const long>(cuda::std::move(v))),
                     const long&&);
    assert(cuda::std::get<const long>(cuda::std::move(v)) == 42);
  }
// FIXME: Remove these once reference support is reinstated
#if !defined(TEST_VARIANT_HAS_NO_REFERENCES)
  {
    using V = cuda::std::variant<int&>;
    int x = 42;
    const V v(x);
    ASSERT_SAME_TYPE(decltype(cuda::std::get<int&>(cuda::std::move(v))), int&);
    assert(&cuda::std::get<int&>(cuda::std::move(v)) == &x);
  }
  {
    using V = cuda::std::variant<const int&>;
    int x = 42;
    const V v(x);
    ASSERT_SAME_TYPE(decltype(cuda::std::get<const int&>(cuda::std::move(v))),
                     const int&);
    assert(&cuda::std::get<const int&>(cuda::std::move(v)) == &x);
  }
  {
    using V = cuda::std::variant<int&&>;
    int x = 42;
    const V v(cuda::std::move(x));
    ASSERT_SAME_TYPE(decltype(cuda::std::get<int&&>(cuda::std::move(v))),
                     int&&);
    int&& xref = cuda::std::get<int&&>(cuda::std::move(v));
    assert(&xref == &x);
  }
  {
    using V = cuda::std::variant<const int&&>;
    int x = 42;
    const V v(cuda::std::move(x));
    ASSERT_SAME_TYPE(decltype(cuda::std::get<const int&&>(cuda::std::move(v))),
                     const int&&);
    const int&& xref = cuda::std::get<const int&&>(cuda::std::move(v));
    assert(&xref == &x);
  }
#endif
}

template <class Tp>
struct identity {
  using type = Tp;
};

TEST_HOST_DEVICE void test_throws_for_all_value_categories() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  using V = cuda::std::variant<int, long>;
  V v0(42);
  const V& cv0 = v0;
  assert(v0.index() == 0);
  V v1(42l);
  const V& cv1 = v1;
  assert(v1.index() == 1);
  identity<int> zero;
  identity<long> one;
  auto test = [](auto idx, auto&& v) {
    using Idx = decltype(idx);
    try {
      TEST_IGNORE_NODISCARD cuda::std::get<typename Idx::type>(
          cuda::std::forward<decltype(v)>(v));
    } catch (const cuda::std::bad_variant_access&) {
      return true;
    } catch (...) { /* ... */
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
#endif
}

int main(int, char**) {
  test_const_lvalue_get();
  test_lvalue_get();
  test_rvalue_get();
  test_const_rvalue_get();
  test_throws_for_all_value_categories();

  return 0;
}
