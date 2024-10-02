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

//  template <class T, class... Types>
//  constexpr add_pointer_t<T> get_if(variant<Types...>* v) noexcept;
// template <class T, class... Types>
//  constexpr add_pointer_t<const T> get_if(const variant<Types...>* v)
//  noexcept;

#include <cuda/std/cassert>
#include <cuda/std/variant>

#include "test_macros.h"
#include "variant_test_helpers.h"

__host__ __device__ void test_const_get_if()
{
  {
    using V              = cuda::std::variant<int>;
    constexpr const V* v = nullptr;
    static_assert(cuda::std::get_if<int>(v) == nullptr, "");
  }
  {
    using V = cuda::std::variant<int, const long>;
    constexpr V v(42);
    ASSERT_NOEXCEPT(cuda::std::get_if<int>(&v));
    ASSERT_SAME_TYPE(decltype(cuda::std::get_if<int>(&v)), const int*);
#if TEST_STD_VER > 2014 && defined(_CCCL_BUILTIN_ADDRESSOF)
    static_assert(*cuda::std::get_if<int>(&v) == 42, "");
#endif
    static_assert(cuda::std::get_if<const long>(&v) == nullptr, "");
  }
  {
    using V = cuda::std::variant<int, const long>;
    constexpr V v(42l);
    ASSERT_SAME_TYPE(decltype(cuda::std::get_if<const long>(&v)), const long*);
#if TEST_STD_VER > 2014 && defined(_CCCL_BUILTIN_ADDRESSOF)
    static_assert(*cuda::std::get_if<const long>(&v) == 42, "");
#endif
    static_assert(cuda::std::get_if<int>(&v) == nullptr, "");
  }
// FIXME: Remove these once reference support is reinstated
#if !defined(TEST_VARIANT_HAS_NO_REFERENCES)
  {
    using V = cuda::std::variant<int&>;
    int x   = 42;
    const V v(x);
    ASSERT_SAME_TYPE(decltype(cuda::std::get_if<int&>(&v)), int*);
    assert(cuda::std::get_if<int&>(&v) == &x);
  }
  {
    using V = cuda::std::variant<int&&>;
    int x   = 42;
    const V v(cuda::std::move(x));
    ASSERT_SAME_TYPE(decltype(cuda::std::get_if<int&&>(&v)), int*);
    assert(cuda::std::get_if<int&&>(&v) == &x);
  }
  {
    using V = cuda::std::variant<const int&&>;
    int x   = 42;
    const V v(cuda::std::move(x));
    ASSERT_SAME_TYPE(decltype(cuda::std::get_if<const int&&>(&v)), const int*);
    assert(cuda::std::get_if<const int&&>(&v) == &x);
  }
#endif
}

__host__ __device__ void test_get_if()
{
  {
    using V = cuda::std::variant<int>;
    V* v    = nullptr;
    assert(cuda::std::get_if<int>(v) == nullptr);
  }
  {
    using V = cuda::std::variant<int, const long>;
    V v(42);
    ASSERT_NOEXCEPT(cuda::std::get_if<int>(&v));
    ASSERT_SAME_TYPE(decltype(cuda::std::get_if<int>(&v)), int*);
    assert(*cuda::std::get_if<int>(&v) == 42);
    assert(cuda::std::get_if<const long>(&v) == nullptr);
  }
  {
    using V = cuda::std::variant<int, const long>;
    V v(42l);
    ASSERT_SAME_TYPE(decltype(cuda::std::get_if<const long>(&v)), const long*);
    assert(*cuda::std::get_if<const long>(&v) == 42);
    assert(cuda::std::get_if<int>(&v) == nullptr);
  }
// FIXME: Remove these once reference support is reinstated
#if !defined(TEST_VARIANT_HAS_NO_REFERENCES)
  {
    using V = cuda::std::variant<int&>;
    int x   = 42;
    V v(x);
    ASSERT_SAME_TYPE(decltype(cuda::std::get_if<int&>(&v)), int*);
    assert(cuda::std::get_if<int&>(&v) == &x);
  }
  {
    using V = cuda::std::variant<const int&>;
    int x   = 42;
    V v(x);
    ASSERT_SAME_TYPE(decltype(cuda::std::get_if<const int&>(&v)), const int*);
    assert(cuda::std::get_if<const int&>(&v) == &x);
  }
  {
    using V = cuda::std::variant<int&&>;
    int x   = 42;
    V v(cuda::std::move(x));
    ASSERT_SAME_TYPE(decltype(cuda::std::get_if<int&&>(&v)), int*);
    assert(cuda::std::get_if<int&&>(&v) == &x);
  }
  {
    using V = cuda::std::variant<const int&&>;
    int x   = 42;
    V v(cuda::std::move(x));
    ASSERT_SAME_TYPE(decltype(cuda::std::get_if<const int&&>(&v)), const int*);
    assert(cuda::std::get_if<const int&&>(&v) == &x);
  }
#endif
}

int main(int, char**)
{
  test_const_get_if();
  test_get_if();

  return 0;
}
