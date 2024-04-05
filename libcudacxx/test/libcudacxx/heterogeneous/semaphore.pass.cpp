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

#include <cuda/std/semaphore>

template<int N>
struct release
{
    using async = cuda::std::true_type;

    template<typename Semaphore>
    __host__ __device__
    static void perform(Semaphore & semaphore)
    {
        semaphore.release(N);
    }
};

struct acquire
{
    using async = cuda::std::true_type;

    template<typename Semaphore>
    __host__ __device__
    static void perform(Semaphore & semaphore)
    {
        semaphore.acquire();
    }
};

using a_a_r2 = performer_list<
    acquire,
    acquire,
    release<2>
>;

using a_a_a_r1_r2 = performer_list<
    acquire,
    acquire,
    acquire,
    release<1>,
    release<2>
>;

using a_r3_a_a = performer_list<
    acquire,
    release<3>,
    acquire,
    acquire
>;

void kernel_invoker()
{
    validate_not_movable<
        cuda::std::counting_semaphore<3>,
        a_a_r2
    >(0);
    validate_not_movable<
        cuda::counting_semaphore<cuda::thread_scope_system, 3>,
        a_a_r2
    >(0);

    validate_not_movable<
        cuda::std::counting_semaphore<3>,
        a_a_a_r1_r2
    >(0);
    validate_not_movable<
        cuda::counting_semaphore<cuda::thread_scope_system, 3>,
        a_a_a_r1_r2
    >(0);

    validate_not_movable<
        cuda::std::counting_semaphore<3>,
        a_r3_a_a
    >(0);
    validate_not_movable<
        cuda::counting_semaphore<cuda::thread_scope_system, 3>,
        a_r3_a_a
    >(0);

}

int main(int arg, char ** argv)
{
    NV_IF_TARGET(NV_IS_HOST,(
        kernel_invoker();
    ))

    return 0;
}
