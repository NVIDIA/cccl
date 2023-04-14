//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc, pre-sm-70

// uncomment for a really verbose output detailing what test steps are being launched
// #define DEBUG_TESTERS

#include "helpers.h"

#include <cuda/std/latch>

template<int N>
struct count_down
{
    using async = cuda::std::true_type;

    template<typename Latch>
    __host__ __device__
    static void perform(Latch & latch)
    {
        latch.count_down(N);
    }
};

template<int N>
struct arrive_and_wait
{
    using async = cuda::std::true_type;

    template<typename Latch>
    __host__ __device__
    static void perform(Latch & latch)
    {
        latch.arrive_and_wait(N);
    }
};

// This one is named `latch_wait` because otherwise you get this on older systems:
// .../latch.pass.cpp(44): error: invalid redeclaration of type name "wait"
// /usr/include/bits/waitstatus.h(66): here
// Isn't software great?
struct latch_wait
{
    using async = cuda::std::true_type;

    template<typename Latch>
    __host__ __device__
    static void perform(Latch & latch)
    {
        latch.wait();
    }
};

template<int Expected>
struct reset
{
    template<typename Latch>
    __host__ __device__
    static void perform(Latch & latch)
    {
        new (&latch) Latch(Expected);
    }
};

using r0_w = performer_list<
    reset<0>,
    latch_wait
>;

using r5_cd1_aw2_w_cd2 = performer_list<
    reset<5>,
    count_down<1>,
    arrive_and_wait<2>,
    latch_wait,
    count_down<2>
>;

using r3_aw1_aw1_aw1 = performer_list<
    reset<3>,
    arrive_and_wait<1>,
    arrive_and_wait<1>,
    arrive_and_wait<1>
>;

void kernel_invoker()
{
    validate_not_movable<
        cuda::std::latch,
        r0_w
    >(0);
    validate_not_movable<
        cuda::latch<cuda::thread_scope_system>,
        r0_w
    >(0);

    validate_not_movable<
        cuda::std::latch,
        r5_cd1_aw2_w_cd2
    >(0);
    validate_not_movable<
        cuda::latch<cuda::thread_scope_system>,
        r5_cd1_aw2_w_cd2
    >(0);

    validate_not_movable<
        cuda::std::latch,
        r3_aw1_aw1_aw1
    >(0);
    validate_not_movable<
        cuda::latch<cuda::thread_scope_system>,
        r3_aw1_aw1_aw1
    >(0);
}

int main(int arg, char ** argv)
{
    NV_IF_TARGET(NV_IS_HOST,(
        kernel_invoker();
    ))

    return 0;
}
