//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// UNSUPPORTED: msvc-19.16
// UNSUPPORTED: clang-7, clang-8

// <cuda/std/variant>

// template <class ...Types> class variant;

// template <class T> constexpr variant(T&&) noexcept(see below);

#include <cuda/std/cassert>
// #include <cuda/std/string>
#include <cuda/std/type_traits>
#include <cuda/std/variant>
// #include <cuda/std/memory>

#include "test_macros.h"
#include "variant_test_helpers.h"

struct Dummy
{
  Dummy() = default;
};

struct ThrowsT
{
  __host__ __device__ ThrowsT(int) noexcept(false) {}
};

struct NoThrowT
{
  __host__ __device__ NoThrowT(int) noexcept(true) {}
};

struct AnyConstructible
{
  template <typename T>
  __host__ __device__ AnyConstructible(T&&)
  {}
};
struct NoConstructible
{
  NoConstructible() = delete;
};
template <class T>
struct RValueConvertibleFrom
{
  __host__ __device__ RValueConvertibleFrom(T&&) {}
};

__host__ __device__ void test_T_ctor_noexcept()
{
  {
    using V = cuda::std::variant<Dummy, NoThrowT>;
    static_assert(cuda::std::is_nothrow_constructible<V, int>::value, "");
  }
#if !defined(TEST_COMPILER_ICC)
  {
    using V = cuda::std::variant<Dummy, ThrowsT>;
    static_assert(!cuda::std::is_nothrow_constructible<V, int>::value, "");
  }
#endif // !TEST_COMPILER_ICC
}

__host__ __device__ void test_T_ctor_sfinae()
{
  {
    using V = cuda::std::variant<long, long long>;
    static_assert(!cuda::std::is_constructible<V, int>::value, "ambiguous");
  }
  /* {
    using V = cuda::std::variant<cuda::std::string, cuda::std::string>;
    static_assert(!cuda::std::is_constructible<V, const char *>::value, "ambiguous");
  }
  {
    using V = cuda::std::variant<cuda::std::string, void *>;
    static_assert(!cuda::std::is_constructible<V, int>::value,
                  "no matching constructor");
  }
  {
    using V = cuda::std::variant<cuda::std::string, float>;
    static_assert(cuda::std::is_constructible<V, int>::value == VariantAllowsNarrowingConversions,
                  "no matching constructor");
  }
  {
    using V = cuda::std::variant<cuda::std::unique_ptr<int>, bool>;
    static_assert(!cuda::std::is_constructible<V, cuda::std::unique_ptr<char>>::value,
                  "no explicit bool in constructor");
    struct X {
      __host__ __device__ operator void*();
    };
    static_assert(!cuda::std::is_constructible<V, X>::value,
                  "no boolean conversion in constructor");
    static_assert(!cuda::std::is_constructible<V, cuda::std::false_type>::value,
                  "no converted to bool in constructor");
  } */
  {
    struct X
    {};
    struct Y
    {
      __host__ __device__ operator X();
    };
    using V = cuda::std::variant<X>;
    static_assert(cuda::std::is_constructible<V, Y>::value, "regression on user-defined conversions in constructor");
  }
  {
    using V = cuda::std::variant<AnyConstructible, NoConstructible>;
    static_assert(!cuda::std::is_constructible<V, cuda::std::in_place_type_t<NoConstructible>>::value,
                  "no matching constructor");
    static_assert(!cuda::std::is_constructible<V, cuda::std::in_place_index_t<1>>::value, "no matching constructor");
  }

#if !defined(TEST_VARIANT_HAS_NO_REFERENCES)
  {
    using V = cuda::std::variant<int, int&&>;
    static_assert(!cuda::std::is_constructible<V, int>::value, "ambiguous");
  }
  {
    using V = cuda::std::variant<int, const int&>;
    static_assert(!cuda::std::is_constructible<V, int>::value, "ambiguous");
  }
#endif
}

__host__ __device__ void test_T_ctor_basic()
{
  {
    constexpr cuda::std::variant<int> v(42);
    static_assert(v.index() == 0, "");
    static_assert(cuda::std::get<0>(v) == 42, "");
  }
  {
    constexpr cuda::std::variant<int, long> v(42l);
    static_assert(v.index() == 1, "");
    static_assert(cuda::std::get<1>(v) == 42, "");
  }
#ifndef TEST_VARIANT_ALLOWS_NARROWING_CONVERSIONS
  {
    constexpr cuda::std::variant<unsigned, long> v(42);
    static_assert(v.index() == 1, "");
    static_assert(cuda::std::get<1>(v) == 42, "");
  }
#endif
  /* {
    cuda::std::variant<cuda::std::string, bool const> v = "foo";
    assert(v.index() == 0);
    assert(cuda::std::get<0>(v) == "foo");
  }
  {
    cuda::std::variant<bool volatile, cuda::std::unique_ptr<int>> v = nullptr;
    assert(v.index() == 1);
    assert(cuda::std::get<1>(v) == nullptr);
  } */
  {
    cuda::std::variant<bool volatile const, int> v = true;
    assert(v.index() == 0);
    assert(cuda::std::get<0>(v));
  }
  {
    cuda::std::variant<RValueConvertibleFrom<int>> v1 = 42;
    assert(v1.index() == 0);

    int x                                                               = 42;
    cuda::std::variant<RValueConvertibleFrom<int>, AnyConstructible> v2 = x;
    assert(v2.index() == 1);
  }
#if !defined(TEST_VARIANT_HAS_NO_REFERENCES)
  {
    using V = cuda::std::variant<const int&, int&&, long>;
    static_assert(cuda::std::is_convertible<int&, V>::value, "must be implicit");
    int x = 42;
    V v(x);
    assert(v.index() == 0);
    assert(&cuda::std::get<0>(v) == &x);
  }
  {
    using V = cuda::std::variant<const int&, int&&, long>;
    static_assert(cuda::std::is_convertible<int, V>::value, "must be implicit");
    int x = 42;
    V v(cuda::std::move(x));
    assert(v.index() == 1);
    assert(&cuda::std::get<1>(v) == &x);
  }
#endif
}

struct BoomOnAnything
{
  template <class T>
  __host__ __device__ constexpr BoomOnAnything(T)
  {
    static_assert(!cuda::std::is_same<T, T>::value, "");
  }
};

__host__ __device__ void test_no_narrowing_check_for_class_types()
{
  using V = cuda::std::variant<int, BoomOnAnything>;
  V v(42);
  assert(v.index() == 0);
  assert(cuda::std::get<0>(v) == 42);
}

struct Bar
{};
struct Baz
{};
__host__ __device__ void test_construction_with_repeated_types()
{
  using V = cuda::std::variant<int, Bar, Baz, int, Baz, int, int>;
#if !defined(TEST_COMPILER_GCC) || __GNUC__ >= 7
  static_assert(!cuda::std::is_constructible<V, int>::value, "");
  static_assert(!cuda::std::is_constructible<V, Baz>::value, "");
#endif // !gcc-6
  // OK, the selected type appears only once and so it shouldn't
  // be affected by the duplicate types.
  static_assert(cuda::std::is_constructible<V, Bar>::value, "");
}

int main(int, char**)
{
  test_T_ctor_basic();
  test_T_ctor_noexcept();
  test_T_ctor_sfinae();
  test_no_narrowing_check_for_class_types();
  test_construction_with_repeated_types();
  return 0;
}
