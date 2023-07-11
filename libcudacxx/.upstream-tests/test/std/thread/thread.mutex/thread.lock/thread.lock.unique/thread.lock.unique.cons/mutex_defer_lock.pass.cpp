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

// unique_lock(mutex_type& m, defer_lock_t);

#include<cuda/std/mutex>
#include<cuda/std/cassert>
#include "nasty_containers.h"

#include "test_macros.h"

int main(int, char**)
{
    {
    typedef cuda::std::mutex M;
    M m;
    cuda::std::unique_lock<M> lk(m, cuda::std::defer_lock);
    assert(lk.mutex() == cuda::std::addressof(m));
    assert(lk.owns_lock() == false);
    }
    {
    typedef nasty_mutex M;
    M m;
    cuda::std::unique_lock<M> lk(m, cuda::std::defer_lock);
    assert(lk.mutex() == cuda::std::addressof(m));
    assert(lk.owns_lock() == false);
    }

  return 0;
}
