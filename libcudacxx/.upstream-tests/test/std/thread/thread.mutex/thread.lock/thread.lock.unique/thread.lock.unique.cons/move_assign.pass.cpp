//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads, c++03
// UNSUPPORTED: pre-sm-70

// <mutex>

// template <class Mutex> class unique_lock;

// unique_lock& operator=(unique_lock&& u);

#include<cuda/std/mutex>
#include<cuda/std/cassert>
#include "nasty_containers.h"

#include "test_macros.h"

int main(int, char**)
{
    {
    typedef cuda::std::mutex M;
    M m0;
    M m1;
    cuda::std::unique_lock<M> lk0(m0);
    cuda::std::unique_lock<M> lk1(m1);
    lk1 = cuda::std::move(lk0);
    assert(lk1.mutex() == cuda::std::addressof(m0));
    assert(lk1.owns_lock() == true);
    assert(lk0.mutex() == nullptr);
    assert(lk0.owns_lock() == false);
    }
    {
    typedef nasty_mutex M;
    M m0;
    M m1;
    cuda::std::unique_lock<M> lk0(m0);
    cuda::std::unique_lock<M> lk1(m1);
    lk1 = cuda::std::move(lk0);
    assert(lk1.mutex() == cuda::std::addressof(m0));
    assert(lk1.owns_lock() == true);
    assert(lk0.mutex() == nullptr);
    assert(lk0.owns_lock() == false);
    }

  return 0;
}
