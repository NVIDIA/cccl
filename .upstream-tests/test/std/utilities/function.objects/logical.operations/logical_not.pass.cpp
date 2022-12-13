//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/functional>

// logical_not

#include <cuda/std/functional>
#include <cuda/std/type_traits>
#include <cuda/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
    typedef cuda::std::logical_not<int> F;
    const F f = F();
#if _LIBCUDACXX_STD_VER <= 14 || defined(_LIBCUDACXX_ENABLE_CXX17_REMOVED_UNARY_BINARY_FUNCTION)
    static_assert((cuda::std::is_same<F::argument_type, int>::value), "" );
    static_assert((cuda::std::is_same<F::result_type, bool>::value), "" );
#endif
    assert(!f(36));
    assert(f(0));
#if TEST_STD_VER > 11
    typedef cuda::std::logical_not<> F2;
    const F2 f2 = F2();
    assert(!f2(36));
    assert( f2(0));
    assert(!f2(36L));
    assert( f2(0L));

    constexpr bool foo = cuda::std::logical_not<int> () (36);
    static_assert ( !foo, "" );

    constexpr bool bar = cuda::std::logical_not<> () (36);
    static_assert ( !bar, "" );
#endif

  return 0;
}
