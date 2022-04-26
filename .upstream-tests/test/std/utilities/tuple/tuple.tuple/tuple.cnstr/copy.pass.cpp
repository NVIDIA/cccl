//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/tuple>

// template <class... Types> class tuple;

// tuple(const tuple& u) = default;

// UNSUPPORTED: c++98, c++03
// Internal compiler error in 14.24
// XFAIL: msvc-19.20, msvc-19.21, msvc-19.22, msvc-19.23, msvc-19.24, msvc-19.25

#include <cuda/std/tuple>
#include <cuda/std/cassert>

#include "test_macros.h"

template <typename T>
__host__ __device__
constexpr bool unused(T &&) {return true;}

struct Empty {};

int main(int, char**)
{
    {
        typedef cuda::std::tuple<> T;
        T t0;
        T t = t0;
        unused(t); // Prevent unused warning
    }
    {
        typedef cuda::std::tuple<int> T;
        T t0(2);
        T t = t0;
        assert(cuda::std::get<0>(t) == 2);
    }
    {
        typedef cuda::std::tuple<int, char> T;
        T t0(2, 'a');
        T t = t0;
        assert(cuda::std::get<0>(t) == 2);
        assert(cuda::std::get<1>(t) == 'a');
    }
    // cuda::std::string not supported
    /*
    {
        typedef cuda::std::tuple<int, char, cuda::std::string> T;
        const T t0(2, 'a', "some text");
        T t = t0;
        assert(cuda::std::get<0>(t) == 2);
        assert(cuda::std::get<1>(t) == 'a');
        assert(cuda::std::get<2>(t) == "some text");
    }
    */
#if TEST_STD_VER > 11
    {
        typedef cuda::std::tuple<int> T;
        constexpr T t0(2);
        constexpr T t = t0;
        static_assert(cuda::std::get<0>(t) == 2, "");
    }
    {
        typedef cuda::std::tuple<Empty> T;
        constexpr T t0;
        constexpr T t = t0;
        constexpr Empty e = cuda::std::get<0>(t);
        ((void)e); // Prevent unused warning
    }
#endif

  return 0;
}
