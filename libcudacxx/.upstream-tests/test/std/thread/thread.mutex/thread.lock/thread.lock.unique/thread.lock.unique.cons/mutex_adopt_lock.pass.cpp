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

// unique_lock(mutex_type& m, adopt_lock_t);

#include<cuda/std/mutex>
#include<cuda/std/cassert>
#include "nasty_containers.h"

#include "test_macros.h"

__host__ __device__ void test() {
    {
    typedef cuda::std::mutex M;
    M m;
    m.lock();
    cuda::std::unique_lock<M> lk(m, cuda::std::adopt_lock);
    assert(lk.mutex() == cuda::std::addressof(m));
    assert(lk.owns_lock() == true);
    }
    {
    typedef nasty_mutex M;
    M m;
    m.lock();
    cuda::std::unique_lock<M> lk(m, cuda::std::adopt_lock);
    assert(lk.mutex() == cuda::std::addressof(m));
    assert(lk.owns_lock() == true);
    }
}

int main(int, char**)
{
  test();
  return 0;
}
