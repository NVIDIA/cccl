//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc
// Internal compiler error in 14.24
// XFAIL: msvc-19.24

// uncomment for a really verbose output detailing what test steps are being launched
// #define DEBUG_TESTERS

#include "helpers.h"

#include <cuda/std/tuple>
#include <cuda/std/cassert>

struct pod {
    char val[10];
};

using tuple_t = cuda::std::tuple<int, pod, unsigned long long>;

template<int N>
struct write
{
    using async = cuda::std::false_type;

    template <typename Tuple>
    __host__ __device__
    static void perform(Tuple &t)
    {
        cuda::std::get<0>(t) = N;
        cuda::std::get<1>(t).val[0] = N;
        cuda::std::get<2>(t) = N;
    }
};

template<int N>
struct read
{
    using async = cuda::std::false_type;

    template <typename Tuple>
    __host__ __device__
    static void perform(Tuple &t)
    {
        assert(cuda::std::get<0>(t) == N);
        assert(cuda::std::get<1>(t).val[0] == N);
        assert(cuda::std::get<2>(t) == N);
    }
};

using w_r_w_r = performer_list<
  write<10>,
  read<10>,
  write<30>,
  read<30>
>;

void kernel_invoker()
{
    tuple_t t(0, {0}, 0);
    validate_not_movable<
        tuple_t,
        w_r_w_r
    >(t);
}

int main(int arg, char ** argv)
{
#ifndef __CUDA_ARCH__
    kernel_invoker();
#endif

    return 0;
}
