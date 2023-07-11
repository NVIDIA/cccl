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

// void unlock();

#include<cuda/std/mutex>
#include<cuda/std/cassert>

#include "test_macros.h"

STATIC_TEST_GLOBAL_VAR bool unlock_called = false;

struct mutex
{
    __host__ __device__ void lock() {}
    __host__ __device__ void unlock() {unlock_called = true;}
};

STATIC_TEST_GLOBAL_VAR mutex m;

int main(int, char**)
{
    cuda::std::unique_lock<mutex> lk(m);
    lk.unlock();
    assert(unlock_called == true);
    assert(lk.owns_lock() == false);
#ifndef TEST_HAS_NO_EXCEPTIONS
    try
    {
        lk.unlock();
        assert(false);
    }
    catch (cuda::std::system_error& e)
    {
        assert(e.code().value() == EPERM);
    }
#endif
    lk.release();
#ifndef TEST_HAS_NO_EXCEPTIONS
    try
    {
        lk.unlock();
        assert(false);
    }
    catch (cuda::std::system_error& e)
    {
        assert(e.code().value() == EPERM);
    }
#endif

  return 0;
}
