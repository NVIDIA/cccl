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

// unique_lock(unique_lock&& u);

#include<cuda/std/mutex>
#include<cuda/std/cassert>
#include "nasty_containers.h"

#include "test_macros.h"

int main(int, char**)
{
    {
    typedef cuda::std::mutex M;
    M m;
    cuda::std::unique_lock<M> lk0(m);
    cuda::std::unique_lock<M> lk = cuda::std::move(lk0);
    assert(lk.mutex() == cuda::std::addressof(m));
    assert(lk.owns_lock() == true);
    assert(lk0.mutex() == nullptr);
    assert(lk0.owns_lock() == false);
    }
    {
    typedef nasty_mutex M;
    M m;
    cuda::std::unique_lock<M> lk0(m);
    cuda::std::unique_lock<M> lk = cuda::std::move(lk0);
    assert(lk.mutex() == cuda::std::addressof(m));
    assert(lk.owns_lock() == true);
    assert(lk0.mutex() == nullptr);
    assert(lk0.owns_lock() == false);
    }

  return 0;
}
