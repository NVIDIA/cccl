//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// UNSUPPORTED: pre-sm-70

// <mutex>

// template <class Mutex> class unique_lock;

// mutex_type* release() noexcept;

#include<cuda/std/mutex>
#include<cuda/std/cassert>

#include "test_macros.h"

STATIC_TEST_GLOBAL_VAR int lock_count = 0;
STATIC_TEST_GLOBAL_VAR int unlock_count = 0;
struct mutex
{
    __host__ __device__ void lock() {++lock_count;}
    __host__ __device__ void unlock() {++unlock_count;}
};

STATIC_TEST_GLOBAL_VAR mutex m;

int main(int, char**)
{
    cuda::std::unique_lock<mutex> lk(m);
    assert(lk.mutex() == &m);
    assert(lk.owns_lock() == true);
    assert(lock_count == 1);
    assert(unlock_count == 0);
    assert(lk.release() == &m);
    assert(lk.mutex() == nullptr);
    assert(lk.owns_lock() == false);
    assert(lock_count == 1);
    assert(unlock_count == 0);

  return 0;
}
