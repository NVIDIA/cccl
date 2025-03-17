//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/type_traits>

// constexpr bool is_constant_evaluated() noexcept; // C++20

#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "test_macros.h"

#if TEST_STD_VER > 2017
#  ifndef __cccl_lib_is_constant_evaluated
#    if defined(_CCCL_BUILTIN_IS_CONSTANT_EVALUATED)
#      error __cccl_lib_is_constant_evaluated should be defined
#    endif // _CCCL_BUILTIN_IS_CONSTANT_EVALUATED
#  endif // __cccl_lib_is_constant_evaluated
#endif // TEST_STD_VER > 2017

template <bool>
struct InTemplate
{};

int main(int, char**)
{
#if defined(_CCCL_BUILTIN_IS_CONSTANT_EVALUATED)
  // Test the signature
  {
    static_assert(cuda::std::is_same_v<decltype(cuda::std::is_constant_evaluated()), bool>);
    static_assert(noexcept(cuda::std::is_constant_evaluated()));
    constexpr bool p = cuda::std::is_constant_evaluated();
    assert(p);
  }
  // Test the return value of the builtin for basic sanity only. It's the
  // compilers job to test tho builtin for correctness.
  {
    static_assert(cuda::std::is_constant_evaluated(), "");
    bool p = cuda::std::is_constant_evaluated();
    assert(!p);
    static_assert(cuda::std::is_same_v<InTemplate<cuda::std::is_constant_evaluated()>, InTemplate<true>>);
    static int local_static = cuda::std::is_constant_evaluated() ? 42 : -1;
    assert(local_static == 42);
  }
#endif
  return 0;
}
