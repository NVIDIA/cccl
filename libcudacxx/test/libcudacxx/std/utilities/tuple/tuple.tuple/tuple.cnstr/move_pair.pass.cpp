//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/tuple>

// template <class... Types> class tuple;

// template <class U1, class U2> tuple(pair<U1, U2>&& u);

// UNSUPPORTED: c++98, c++03


#include <cuda/std/tuple>
#include <cuda/std/cassert>

#include "test_macros.h"

struct B
{
    int id_;

    TEST_HOST_DEVICE explicit B(int i) : id_(i) {}

    TEST_HOST_DEVICE virtual ~B() {}
};

struct D
    : B
{
    TEST_HOST_DEVICE explicit D(int i) : B(i) {}
};

int main(int, char**)
{
    // cuda::std::unique_ptr not supported
    /*
    {
        typedef cuda::std::pair<long, cuda::std::unique_ptr<D>> T0;
        typedef cuda::std::tuple<long long, cuda::std::unique_ptr<B>> T1;
        T0 t0(2, cuda::std::unique_ptr<D>(new D(3)));
        T1 t1 = cuda::std::move(t0);
        assert(cuda::std::get<0>(t1) == 2);
        assert(cuda::std::get<1>(t1)->id_ == 3);
    }
    */

  return 0;
}
