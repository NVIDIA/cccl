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

// void swap(unique_lock& u);

#include<cuda/std/mutex>
#include<cuda/std/cassert>

#include "test_macros.h"

struct mutex
{
    __host__ __device__ void lock() {}
    __host__ __device__ void unlock() {}
};

STATIC_TEST_GLOBAL_VAR mutex m;

int main(int, char**)
{
    cuda::std::unique_lock<mutex> lk1(m);
    cuda::std::unique_lock<mutex> lk2;
    lk1.swap(lk2);
    assert(lk1.mutex() == nullptr);
    assert(lk1.owns_lock() == false);
    assert(lk2.mutex() == &m);
    assert(lk2.owns_lock() == true);

  return 0;
}
