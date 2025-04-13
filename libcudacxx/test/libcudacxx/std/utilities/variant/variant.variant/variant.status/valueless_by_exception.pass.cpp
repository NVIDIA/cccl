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

// constexpr bool valueless_by_exception() const noexcept;

#include <cuda/std/cassert>
#if defined(_LIBCUDACXX_HAS_STRING)
#  include <cuda/std/string>
#endif // _LIBCUDACXX_HAS_STRING
#include <cuda/std/type_traits>
#include <cuda/std/variant>

#include "archetypes.h"
#include "test_macros.h"
#include "variant_test_helpers.h"

#if TEST_HAS_EXCEPTIONS()
void test_exceptions()
{
  using V = cuda::std::variant<int, MakeEmptyT>;
  V v;
  assert(!v.valueless_by_exception());
  makeEmpty(v);
  assert(v.valueless_by_exception());
}
#endif // TEST_HAS_EXCEPTIONS()

int main(int, char**)
{
  {
    using V = cuda::std::variant<int, long>;
    constexpr V v;
    static_assert(!v.valueless_by_exception(), "");
  }
  {
    using V = cuda::std::variant<int, long>;
    V v;
    assert(!v.valueless_by_exception());
  }
#if defined(_LIBCUDACXX_HAS_STRING)
  {
    using V = cuda::std::variant<int, long, cuda::std::string>;
    const V v("abc");
    assert(!v.valueless_by_exception());
  }
#endif // _LIBCUDACXX_HAS_STRING
#if TEST_HAS_EXCEPTIONS()
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions();))
#endif // TEST_HAS_EXCEPTIONS()

  return 0;
}
