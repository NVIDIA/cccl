//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/functional>

// negate

#include <cuda/std/functional>
#include <cuda/std/type_traits>
#include <cuda/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
    typedef cuda::std::negate<int> F;
    const F f = F();
#if _LIBCUDACXX_STD_VER <= 14 || defined(_LIBCUDACXX_ENABLE_CXX17_REMOVED_UNARY_BINARY_FUNCTION)
    static_assert((cuda::std::is_same<F::argument_type, int>::value), "" );
    static_assert((cuda::std::is_same<F::result_type, int>::value), "" );
#endif
    assert(f(36) == -36);
#if TEST_STD_VER > 11
    typedef cuda::std::negate<> F2;
    const F2 f2 = F2();
    assert(f2(36) == -36);
    assert(f2(36L) == -36);
    assert(f2(36.0) == -36);

    constexpr int foo = cuda::std::negate<int> () (3);
    static_assert ( foo == -3, "" );

    constexpr double bar = cuda::std::negate<> () (3.0);
    static_assert ( bar == -3.0, "" );
#endif

  return 0;
}
