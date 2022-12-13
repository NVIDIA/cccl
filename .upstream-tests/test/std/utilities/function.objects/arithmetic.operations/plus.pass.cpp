//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/functional>

// plus

#include <cuda/std/functional>
#include <cuda/std/type_traits>
#include <cuda/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
    typedef cuda::std::plus<int> F;
    const F f = F();
#if _LIBCUDACXX_STD_VER <= 14 || defined(_LIBCUDACXX_ENABLE_CXX17_REMOVED_UNARY_BINARY_FUNCTION)
    static_assert((cuda::std::is_same<int, F::first_argument_type>::value), "" );
    static_assert((cuda::std::is_same<int, F::second_argument_type>::value), "" );
    static_assert((cuda::std::is_same<int, F::result_type>::value), "" );
#endif
    assert(f(3, 2) == 5);
#if TEST_STD_VER > 11
    typedef cuda::std::plus<> F2;
    const F2 f2 = F2();
    assert(f2(3,2) == 5);
    assert(f2(3.0, 2) == 5);
    assert(f2(3, 2.5) == 5.5);

    constexpr int foo = cuda::std::plus<int> () (3, 2);
    static_assert ( foo == 5, "" );

    constexpr double bar = cuda::std::plus<> () (3.0, 2);
    static_assert ( bar == 5.0, "" );
#endif

  return 0;
}
