//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef TEST_SUPPORT_HETEROGENEOUS_TEST_THREAD_H
#define TEST_SUPPORT_HETEROGENEOUS_TEST_THREAD_H

#include <nv/target>

#include <cuda/std/chrono>

#ifndef __CUDACC_RTC__
#include <chrono>
#include <thread>
#endif // __CUDACC_RTC__

__host__ __device__
void test_sleep_thread(cuda::std::chrono::milliseconds dur) {
    constexpr cuda::std::chrono::nanoseconds max_sleep_duration{500000};
    cuda::std::chrono::nanoseconds dur_mu{dur};
    cuda::std::chrono::nanoseconds waited{0};
    const auto start = cuda::std::chrono::high_resolution_clock::now();
    while (dur_mu > waited) {
        NV_IF_TARGET(
            NV_IS_DEVICE,
            __libcpp_thread_sleep_for(dur_mu - waited < max_sleep_duration ? dur_mu - waited : max_sleep_duration);,
            std::this_thread::sleep_for(std::chrono::nanoseconds{(dur_mu - waited).count()});
        )
        waited = cuda::std::chrono::high_resolution_clock::now() - start;
    }
}

struct heterogeneous_thread_handler {
#ifndef __CUDACC_RTC__
    union { std::thread t_; };
#endif // __CUDACC_RTC__

    __host__ __device__
    heterogeneous_thread_handler() noexcept {}
    __host__ __device__
    ~heterogeneous_thread_handler() noexcept {}

    template <class F, class ...Args>
    __host__ __device__
    void run_on_first_thread(F&& f, Args&& ...args) {
        NV_IF_TARGET(
            NV_IS_DEVICE,
            if (threadIdx.x == 0) { cuda::std::__invoke(cuda::std::forward<F>(f), cuda::std::forward<Args>(args)...); },
            cuda::std::__invoke(cuda::std::forward<F>(f), cuda::std::forward<Args>(args)...);
        )
    }

    template <class F, class ...Args>
    __host__ __device__
    void run_on_second_thread(F&& f, Args&& ...args) {
        NV_IF_TARGET(
            NV_IS_DEVICE,
            if (threadIdx.x == 1) { cuda::std::__invoke(cuda::std::forward<F>(f), cuda::std::forward<Args>(args)...); },
            ::new((void*)std::addressof(t_)) std::thread(std::forward<F>(f), std::forward<Args>(args)...);
        )
    }

    __host__ __device__
    void sleep_first_thread(cuda::std::chrono::milliseconds dur) {
        test_sleep_thread(dur);
    }

    __host__ __device__
    void syncthreads() const {
        NV_IF_TARGET(
            NV_IS_DEVICE,
            __syncthreads();
        )
    }

    __host__ __device__
    void join_test_thread() {
        NV_IF_TARGET(
            NV_IS_DEVICE,
            (),
            t_.join();
        )
    }
};

#endif // TEST_SUPPORT_HETEROGENEOUS_TEST_THREAD_H
