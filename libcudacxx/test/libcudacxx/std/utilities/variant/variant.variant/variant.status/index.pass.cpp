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

// constexpr size_t index() const noexcept;

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
  assert(v.index() == 0);
  makeEmpty(v);
  assert(v.index() == cuda::std::variant_npos);
}
#endif // TEST_HAS_EXCEPTIONS()

int main(int, char**)
{
  {
    using V = cuda::std::variant<int, long>;
    constexpr V v;
    static_assert(v.index() == 0, "");
  }
  {
    using V = cuda::std::variant<int, long>;
    V v;
    assert(v.index() == 0);
  }
  {
    using V = cuda::std::variant<int, long>;
    constexpr V v(cuda::std::in_place_index<1>);
    static_assert(v.index() == 1, "");
  }
#if defined(_LIBCUDACXX_HAS_STRING)
  {
    using V = cuda::std::variant<int, cuda::std::string>;
    V v("abc");
    assert(v.index() == 1);
    v = 42;
    assert(v.index() == 0);
  }
#endif // _LIBCUDACXX_HAS_STRING
#if TEST_HAS_EXCEPTIONS()
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions();))
#endif // TEST_HAS_EXCEPTIONS()

  return 0;
}
