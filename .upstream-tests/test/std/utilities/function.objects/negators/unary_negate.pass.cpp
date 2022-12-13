//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/functional>

// unary_negate

#define _LIBCUDACXX_DISABLE_DEPRECATION_WARNINGS

#include <cuda/std/functional>
#include <cuda/std/type_traits>
#include <cuda/std/cassert>

int main(int, char**)
{
    typedef cuda::std::unary_negate<cuda::std::logical_not<int> > F;
    const F f = F(cuda::std::logical_not<int>());
#if _LIBCUDACXX_STD_VER <= 14 || defined(_LIBCUDACXX_ENABLE_CXX17_REMOVED_UNARY_BINARY_FUNCTION)
    static_assert((cuda::std::is_same<F::argument_type, int>::value), "" );
    static_assert((cuda::std::is_same<F::result_type, bool>::value), "" );
#endif
    assert(f(36));
    assert(!f(0));

  return 0;
}
