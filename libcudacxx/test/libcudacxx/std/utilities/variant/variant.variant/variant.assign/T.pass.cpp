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

// template <class ...Types> class variant;

// template <class T>
// variant& operator=(T&&) noexcept(see below);

#include <cuda/std/cassert>
// #include <cuda/std/string>
#include <cuda/std/type_traits>
#include <cuda/std/variant>
// #include <cuda/std/memory>

#include "test_macros.h"
#include "variant_test_helpers.h"

namespace MetaHelpers {

struct Dummy {
  Dummy() = default;
};

struct ThrowsCtorT {
  TEST_HOST_DEVICE
  ThrowsCtorT(int) noexcept(false) {}
  TEST_HOST_DEVICE
  ThrowsCtorT &operator=(int) noexcept { return *this; }
};

struct ThrowsAssignT {
  TEST_HOST_DEVICE
  ThrowsAssignT(int) noexcept {}
  TEST_HOST_DEVICE
  ThrowsAssignT &operator=(int) noexcept(false) { return *this; }
};

struct NoThrowT {
  TEST_HOST_DEVICE
  NoThrowT(int) noexcept {}
  TEST_HOST_DEVICE
  NoThrowT &operator=(int) noexcept { return *this; }
};

} // namespace MetaHelpers

namespace RuntimeHelpers {
#ifndef TEST_HAS_NO_EXCEPTIONS

struct ThrowsCtorT {
  int value;
  TEST_HOST_DEVICE
  ThrowsCtorT() : value(0) {}
  TEST_HOST_DEVICE
  ThrowsCtorT(int) noexcept(false) { throw 42; }
  TEST_HOST_DEVICE
  ThrowsCtorT &operator=(int v) noexcept {
    value = v;
    return *this;
  }
};

struct MoveCrashes {
  int value;
  TEST_HOST_DEVICE
  MoveCrashes(int v = 0) noexcept : value{v} {}
  TEST_HOST_DEVICE
  MoveCrashes(MoveCrashes &&) noexcept { assert(false); }
  TEST_HOST_DEVICE
  MoveCrashes &operator=(MoveCrashes &&) noexcept { assert(false); return *this; }
  TEST_HOST_DEVICE
  MoveCrashes &operator=(int v) noexcept {
    value = v;
    return *this;
  }
};

struct ThrowsCtorTandMove {
  int value;
  TEST_HOST_DEVICE
  ThrowsCtorTandMove() : value(0) {}
  TEST_HOST_DEVICE
  ThrowsCtorTandMove(int) noexcept(false) { throw 42; }
  TEST_HOST_DEVICE
  ThrowsCtorTandMove(ThrowsCtorTandMove &&) noexcept(false) { assert(false); }
  TEST_HOST_DEVICE
  ThrowsCtorTandMove &operator=(int v) noexcept {
    value = v;
    return *this;
  }
};

struct ThrowsAssignT {
  int value;
  TEST_HOST_DEVICE
  ThrowsAssignT() : value(0) {}
  TEST_HOST_DEVICE
  ThrowsAssignT(int v) noexcept : value(v) {}
  TEST_HOST_DEVICE
  ThrowsAssignT &operator=(int) noexcept(false) { throw 42; }
};

struct NoThrowT {
  int value;
  TEST_HOST_DEVICE
  NoThrowT() : value(0) {}
  TEST_HOST_DEVICE
  NoThrowT(int v) noexcept : value(v) {}
  TEST_HOST_DEVICE
  NoThrowT &operator=(int v) noexcept {
    value = v;
    return *this;
  }
};

#endif // !defined(TEST_HAS_NO_EXCEPTIONS)
} // namespace RuntimeHelpers

TEST_HOST_DEVICE
void test_T_assignment_noexcept() {
  using namespace MetaHelpers;
  {
    using V = cuda::std::variant<Dummy, NoThrowT>;
    static_assert(cuda::std::is_nothrow_assignable<V, int>::value, "");
  }
#if !defined(TEST_COMPILER_ICC)
  {
    using V = cuda::std::variant<Dummy, ThrowsCtorT>;
    static_assert(!cuda::std::is_nothrow_assignable<V, int>::value, "");
  }
  {
    using V = cuda::std::variant<Dummy, ThrowsAssignT>;
    static_assert(!cuda::std::is_nothrow_assignable<V, int>::value, "");
  }
#endif // !TEST_COMPILER_ICC
}

TEST_HOST_DEVICE
void test_T_assignment_sfinae() {
  {
    using V = cuda::std::variant<long, long long>;
    static_assert(!cuda::std::is_assignable<V, int>::value, "ambiguous");
  }
  /*{
    using V = cuda::std::variant<cuda::std::string, cuda::std::string>;
    static_assert(!cuda::std::is_assignable<V, const char *>::value, "ambiguous");
  }
  {
    using V = cuda::std::variant<cuda::std::string, void *>;
    static_assert(!cuda::std::is_assignable<V, int>::value, "no matching operator=");
  }
  {
    using V = cuda::std::variant<cuda::std::string, float>;
    static_assert(cuda::std::is_assignable<V, int>::value == VariantAllowsNarrowingConversions,
    "no matching operator=");
  }
  {
    using V = cuda::std::variant<cuda::std::unique_ptr<int>, bool>;
    static_assert(!cuda::std::is_assignable<V, cuda::std::unique_ptr<char>>::value,
                  "no explicit bool in operator=");
    struct X {
      operator void*();
    };
    static_assert(!cuda::std::is_assignable<V, X>::value,
                  "no boolean conversion in operator=");
    static_assert(!cuda::std::is_assignable<V, cuda::std::false_type>::value,
                  "no converted to bool in operator=");
  }*/
  {
    // mdominiak: this was originally not an aggregate and we should probably bring that back
    // eventually, except... https://www.godbolt.org/z/oanheq7bv
    struct X { X() = default; };
    struct Y {
      TEST_HOST_DEVICE operator X();
    };
    using V = cuda::std::variant<X>;
    static_assert(cuda::std::is_assignable<V, Y>::value,
                  "regression on user-defined conversions in operator=");
  }
#if !defined(TEST_VARIANT_HAS_NO_REFERENCES)
  {
    using V = cuda::std::variant<int, int &&>;
    static_assert(!cuda::std::is_assignable<V, int>::value, "ambiguous");
  }
  {
    using V = cuda::std::variant<int, const int &>;
    static_assert(!cuda::std::is_assignable<V, int>::value, "ambiguous");
  }
#endif // TEST_VARIANT_HAS_NO_REFERENCES
}

TEST_HOST_DEVICE
void test_T_assignment_basic() {
  {
    cuda::std::variant<int> v(43);
    v = 42;
    assert(v.index() == 0);
    assert(cuda::std::get<0>(v) == 42);
  }
  {
    cuda::std::variant<int, long> v(43l);
    v = 42;
    assert(v.index() == 0);
    assert(cuda::std::get<0>(v) == 42);
    v = 43l;
    assert(v.index() == 1);
    assert(cuda::std::get<1>(v) == 43);
  }
#ifndef TEST_VARIANT_ALLOWS_NARROWING_CONVERSIONS
  {
    cuda::std::variant<unsigned, long> v;
    v = 42;
    assert(v.index() == 1);
    assert(cuda::std::get<1>(v) == 42);
    v = 43u;
    assert(v.index() == 0);
    assert(cuda::std::get<0>(v) == 43);
  }
#endif
  /*{
    cuda::std::variant<cuda::std::string, bool> v = true;
    v = "bar";
    assert(v.index() == 0);
    assert(cuda::std::get<0>(v) == "bar");
  }
  {
    cuda::std::variant<bool, cuda::std::unique_ptr<int>> v;
    v = nullptr;
    assert(v.index() == 1);
    assert(cuda::std::get<1>(v) == nullptr);
  }*/
#if !defined(TEST_VARIANT_HAS_NO_REFERENCES)
  {
    using V = cuda::std::variant<int &, int &&, long>;
    int x = 42;
    V v(43l);
    v = x;
    assert(v.index() == 0);
    assert(&cuda::std::get<0>(v) == &x);
    v = cuda::std::move(x);
    assert(v.index() == 1);
    assert(&cuda::std::get<1>(v) == &x);
    // 'long' is selected by FUN(const int &) since 'const int &' cannot bind
    // to 'int&'.
    const int &cx = x;
    v = cx;
    assert(v.index() == 2);
    assert(cuda::std::get<2>(v) == 42);
  }
#endif // TEST_VARIANT_HAS_NO_REFERENCES
}

TEST_HOST_DEVICE
void test_T_assignment_performs_construction() {
  using namespace RuntimeHelpers;
#ifndef TEST_HAS_NO_EXCEPTIONS
  /*{
    using V = cuda::std::variant<cuda::std::string, ThrowsCtorT>;
    V v(cuda::std::in_place_type<cuda::std::string>, "hello");
    try {
      v = 42;
      assert(false);
    } catch (...) { / * ... * /
    }
    assert(v.index() == 0);
    assert(cuda::std::get<0>(v) == "hello");
  }
  {
    using V = cuda::std::variant<ThrowsAssignT, cuda::std::string>;
    V v(cuda::std::in_place_type<cuda::std::string>, "hello");
    v = 42;
    assert(v.index() == 0);
    assert(cuda::std::get<0>(v).value == 42);
  }*/
#endif // TEST_HAS_NO_EXCEPTIONS
}

TEST_HOST_DEVICE
void test_T_assignment_performs_assignment() {
  using namespace RuntimeHelpers;
#ifndef TEST_HAS_NO_EXCEPTIONS
  {
    using V = cuda::std::variant<ThrowsCtorT>;
    V v;
    v = 42;
    assert(v.index() == 0);
    assert(cuda::std::get<0>(v).value == 42);
  }
  /*{
    using V = cuda::std::variant<ThrowsCtorT, cuda::std::string>;
    V v;
    v = 42;
    assert(v.index() == 0);
    assert(cuda::std::get<0>(v).value == 42);
  }*/
  {
    using V = cuda::std::variant<ThrowsAssignT>;
    V v(100);
    try {
      v = 42;
      assert(false);
    } catch (...) { /* ... */
    }
    assert(v.index() == 0);
    assert(cuda::std::get<0>(v).value == 100);
  }
  {
    using V = cuda::std::variant<cuda::std::string, ThrowsAssignT>;
    V v(100);
    try {
      v = 42;
      assert(false);
    } catch (...) { /* ... */
    }
    assert(v.index() == 1);
    assert(cuda::std::get<1>(v).value == 100);
  }
#endif // TEST_HAS_NO_EXCEPTIONS
}

int main(int, char**) {
  test_T_assignment_basic();
  test_T_assignment_performs_construction();
  test_T_assignment_performs_assignment();
  test_T_assignment_noexcept();
  test_T_assignment_sfinae();

  return 0;
}
