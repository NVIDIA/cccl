//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/tuple>

// template <class... Types> class tuple;

// template <class U1, class U2>
//   tuple& operator=(const pair<U1, U2>& u);

// UNSUPPORTED: c++98, c++03
// Internal compiler error in 14.24
// XFAIL: msvc-19.20, msvc-19.21, msvc-19.22, msvc-19.23, msvc-19.24, msvc-19.25

#include <cuda/std/tuple>
#include <cuda/std/utility>
#include <cuda/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        typedef cuda::std::pair<long, char> T0;
        typedef cuda::std::tuple<long long, short> T1;
        T0 t0(2, 'a');
        T1 t1;
        t1 = t0;
        assert(cuda::std::get<0>(t1) == 2);
        assert(cuda::std::get<1>(t1) == short('a'));
    }

  return 0;
}
