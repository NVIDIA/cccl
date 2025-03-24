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

// template <class ...Types> class variant;

// constexpr variant() noexcept(see below);

#include <cuda/std/cassert>
#include <cuda/std/type_traits>
#include <cuda/std/variant>

#include "test_macros.h"
#include "variant_test_helpers.h"

struct NonDefaultConstructible
{
  __host__ __device__ constexpr NonDefaultConstructible(int) {}
};

struct NotNoexcept
{
  __host__ __device__ NotNoexcept() noexcept(false) {}
};

#if TEST_HAS_EXCEPTIONS()
struct DefaultCtorThrows
{
  DefaultCtorThrows()
  {
    throw 42;
  }
};
#endif // TEST_HAS_EXCEPTIONS()

__host__ __device__ void test_default_ctor_sfinae()
{
  {
    using V = cuda::std::variant<cuda::std::monostate, int>;
    static_assert(cuda::std::is_default_constructible<V>::value, "");
  }
  {
    using V = cuda::std::variant<NonDefaultConstructible, int>;
    static_assert(!cuda::std::is_default_constructible<V>::value, "");
  }
#if !defined(TEST_VARIANT_HAS_NO_REFERENCES)
  {
    using V = cuda::std::variant<int&, int>;
    static_assert(!cuda::std::is_default_constructible<V>::value, "");
  }
#endif
}

__host__ __device__ void test_default_ctor_noexcept()
{
  {
    using V = cuda::std::variant<int>;
    static_assert(cuda::std::is_nothrow_default_constructible<V>::value, "");
  }
  {
    using V = cuda::std::variant<NotNoexcept>;
    static_assert(!cuda::std::is_nothrow_default_constructible<V>::value, "");
  }
}

#if TEST_HAS_EXCEPTIONS()
void test_default_ctor_throws()
{
  using V = cuda::std::variant<DefaultCtorThrows, int>;
  try
  {
    V v;
    assert(false);
  }
  catch (const int& ex)
  {
    assert(ex == 42);
  }
  catch (...)
  {
    assert(false);
  }
}
#endif // TEST_HAS_EXCEPTIONS()

__host__ __device__ void test_default_ctor_basic()
{
  {
    cuda::std::variant<int> v;
    assert(v.index() == 0);
    assert(cuda::std::get<0>(v) == 0);
  }
  {
    cuda::std::variant<int, long> v;
    assert(v.index() == 0);
    assert(cuda::std::get<0>(v) == 0);
  }
  {
    cuda::std::variant<int, NonDefaultConstructible> v;
    assert(v.index() == 0);
    assert(cuda::std::get<0>(v) == 0);
  }
  {
    using V = cuda::std::variant<int, long>;
    constexpr V v;
    static_assert(v.index() == 0, "");
    static_assert(cuda::std::get<0>(v) == 0, "");
  }
  {
    using V = cuda::std::variant<int, long>;
    constexpr V v;
    static_assert(v.index() == 0, "");
    static_assert(cuda::std::get<0>(v) == 0, "");
  }
  {
    using V = cuda::std::variant<int, NonDefaultConstructible>;
    constexpr V v;
    static_assert(v.index() == 0, "");
    static_assert(cuda::std::get<0>(v) == 0, "");
  }
}

int main(int, char**)
{
  test_default_ctor_basic();
  test_default_ctor_sfinae();
  test_default_ctor_noexcept();
#if TEST_HAS_EXCEPTIONS()
  NV_IF_TARGET(NV_IS_HOST, (test_default_ctor_throws();))
#endif // TEST_HAS_EXCEPTIONS()

  return 0;
}
