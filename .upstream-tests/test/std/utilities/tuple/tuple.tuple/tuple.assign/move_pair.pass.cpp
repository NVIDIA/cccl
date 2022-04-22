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
//   tuple& operator=(pair<U1, U2>&& u);

// UNSUPPORTED: c++98, c++03
// Internal compiler error in 14.24
// XFAIL: msvc-19.20, msvc-19.21, msvc-19.22, msvc-19.23, msvc-19.24, msvc-19.25

#include <cuda/std/tuple>
#include <cuda/std/utility>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "MoveOnly.h"

struct B
{
    int id_;

    __device__ __host__ explicit B(int i = 0) : id_(i) {}

    __device__ __host__ virtual ~B() {}
};

struct D
    : B
{
    __device__ __host__ explicit D(int i) : B(i) {}
};

int main(int, char**)
{
    {
        typedef cuda::std::pair<long, MoveOnly> T0;
        typedef cuda::std::tuple<long long, MoveOnly> T1;
        T0 t0(2, MoveOnly(3));
        T1 t1;
        t1 = cuda::std::move(t0);
        assert(cuda::std::get<0>(t1) == 2);
        assert(cuda::std::get<1>(t1).get() == 3);
    }

  return 0;
}
