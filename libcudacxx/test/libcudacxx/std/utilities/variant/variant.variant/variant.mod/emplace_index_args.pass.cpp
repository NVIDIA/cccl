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

// template <size_t I, class ...Args>
//   variant_alternative_t<I, variant<Types...>>& emplace(Args&&... args);

#include <cuda/std/cassert>
// #include <cuda/std/string>
#include <cuda/std/type_traits>
#include <cuda/std/variant>

#include "archetypes.h"
#include "test_convertible.h"
#include "test_macros.h"
#include "variant_test_helpers.h"

template <class Var, size_t I, class... Args>
__host__ __device__ constexpr auto test_emplace_exists_imp(int)
  -> decltype(cuda::std::declval<Var>().template emplace<I>(cuda::std::declval<Args>()...), true)
{
  return true;
}

template <class, size_t, class...>
__host__ __device__ constexpr auto test_emplace_exists_imp(long) -> bool
{
  return false;
}

template <class Var, size_t I, class... Args>
__host__ __device__ constexpr bool emplace_exists()
{
  return test_emplace_exists_imp<Var, I, Args...>(0);
}

__host__ __device__ void test_emplace_sfinae()
{
  {
    using V = cuda::std::variant<int, void*, const void*, TestTypes::NoCtors>;
    static_assert(emplace_exists<V, 0>(), "");
    static_assert(emplace_exists<V, 0, int>(), "");
    static_assert(!emplace_exists<V, 0, decltype(nullptr)>(), "cannot construct");
    static_assert(emplace_exists<V, 1, decltype(nullptr)>(), "");
    static_assert(emplace_exists<V, 1, int*>(), "");
    static_assert(!emplace_exists<V, 1, const int*>(), "");
    static_assert(!emplace_exists<V, 1, int>(), "cannot construct");
    static_assert(emplace_exists<V, 2, const int*>(), "");
    static_assert(emplace_exists<V, 2, int*>(), "");
    static_assert(!emplace_exists<V, 3>(), "cannot construct");
  }
#if !defined(TEST_VARIANT_HAS_NO_REFERENCES)
  {
    using V = cuda::std::variant<int, int&, const int&, int&&, TestTypes::NoCtors>;
    static_assert(emplace_exists<V, 0>(), "");
    static_assert(emplace_exists<V, 0, int>(), "");
    static_assert(emplace_exists<V, 0, long long>(), "");
    static_assert(!emplace_exists<V, 0, int, int>(), "too many args");
    static_assert(emplace_exists<V, 1, int&>(), "");
    static_assert(!emplace_exists<V, 1>(), "cannot default construct ref");
    static_assert(!emplace_exists<V, 1, const int&>(), "cannot bind ref");
    static_assert(!emplace_exists<V, 1, int&&>(), "cannot bind ref");
    static_assert(emplace_exists<V, 2, int&>(), "");
    static_assert(emplace_exists<V, 2, const int&>(), "");
    static_assert(emplace_exists<V, 2, int&&>(), "");
    static_assert(!emplace_exists<V, 2, void*>(), "not constructible from void*");
    static_assert(emplace_exists<V, 3, int>(), "");
    static_assert(!emplace_exists<V, 3, int&>(), "cannot bind ref");
    static_assert(!emplace_exists<V, 3, const int&>(), "cannot bind ref");
    static_assert(!emplace_exists<V, 3, const int&&>(), "cannot bind ref");
    static_assert(!emplace_exists<V, 4>(), "no ctors");
  }
#endif
}

__host__ __device__ void test_basic()
{
  {
    using V = cuda::std::variant<int>;
    V v(42);
    auto& ref1 = v.emplace<0>();
    static_assert(cuda::std::is_same_v<int&, decltype(ref1)>, "");
    assert(cuda::std::get<0>(v) == 0);
    assert(&ref1 == &cuda::std::get<0>(v));
    auto& ref2 = v.emplace<0>(42);
    static_assert(cuda::std::is_same_v<int&, decltype(ref2)>, "");
    assert(cuda::std::get<0>(v) == 42);
    assert(&ref2 == &cuda::std::get<0>(v));
  }
  {
    using V     = cuda::std::variant<int, long, const void*, TestTypes::NoCtors>; //, cuda::std::string>;
    const int x = 100;
    V v(cuda::std::in_place_index<0>, -1);
    // default emplace a value
    auto& ref1 = v.emplace<1>();
    static_assert(cuda::std::is_same_v<long&, decltype(ref1)>, "");
    assert(cuda::std::get<1>(v) == 0);
    assert(&ref1 == &cuda::std::get<1>(v));
    auto& ref2 = v.emplace<2>(&x);
    static_assert(cuda::std::is_same_v<const void*&, decltype(ref2)>, "");
    assert(cuda::std::get<2>(v) == &x);
    assert(&ref2 == &cuda::std::get<2>(v));
    // emplace with multiple args
    /* auto& ref3 = v.emplace<4>(3u, 'a');
    static_assert(cuda::std::is_same_v<cuda::std::string&, decltype(ref3)>, "");
    assert(cuda::std::get<4>(v) == "aaa");
    assert(&ref3 == &cuda::std::get<4>(v));*/
  }
#if !defined(TEST_VARIANT_HAS_NO_REFERENCES)
  {
    using V = cuda::std::variant<int, long, const int&, int&&, TestTypes::NoCtors>; //,
                                                                                    // cuda::std::string>;
    const int x = 100;
    int y       = 42;
    int z       = 43;
    V v(cuda::std::in_place_index<0>, -1);
    // default emplace a value
    auto& ref1 = v.emplace<1>();
    static_assert(cuda::std::is_same_v<long&, decltype(ref1)>, "");
    assert(cuda::std::get<1>(v) == 0);
    assert(&ref1 == &cuda::std::get<1>(v));
    // emplace a reference
    auto& ref2 = v.emplace<2>(x);
    static_assert(cuda::std::is_same_v<&, decltype(ref)>, "");
    assert(&cuda::std::get<2>(v) == &x);
    assert(&ref2 == &cuda::std::get<2>(v));
    // emplace an rvalue reference
    auto& ref3 = v.emplace<3>(cuda::std::move(y));
    static_assert(cuda::std::is_same_v<&, decltype(ref)>, "");
    assert(&cuda::std::get<3>(v) == &y);
    assert(&ref3 == &cuda::std::get<3>(v));
    // re-emplace a new reference over the active member
    auto& ref4 = v.emplace<3>(cuda::std::move(z));
    static_assert(cuda::std::is_same_v<&, decltype(ref)>, "");
    assert(&cuda::std::get<3>(v) == &z);
    assert(&ref4 == &cuda::std::get<3>(v));
    // emplace with multiple args
    /*auto& ref5 = v.emplace<5>(3u, 'a');
    static_assert(cuda::std::is_same_v<cuda::std::string&, decltype(ref5)>, "");
    assert(cuda::std::get<5>(v) == "aaa");
    assert(&ref5 == &cuda::std::get<5>(v));*/
  }
#endif
}

int main(int, char**)
{
  test_basic();
  test_emplace_sfinae();

  return 0;
}
