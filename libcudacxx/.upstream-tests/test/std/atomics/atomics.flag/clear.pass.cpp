//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads, pre-sm-60
// UNSUPPORTED: windows && pre-sm-70

// <cuda/std/atomic>

// struct atomic_flag

// void clear(memory_order = memory_order_seq_cst);
// void clear(memory_order = memory_order_seq_cst) volatile;

#include <cuda/std/atomic>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "cuda_space_selector.h"

template<template<typename, typename> class Selector>
__host__ __device__
void test()
{
    {
        Selector<cuda::std::atomic_flag, default_initializer> sel;
        cuda::std::atomic_flag & f = *sel.construct();
        f.clear();
        assert(f.test_and_set() == 0);
        f.clear();
        assert(f.test_and_set() == 0);
    }
    {
        Selector<cuda::std::atomic_flag, default_initializer> sel;
        cuda::std::atomic_flag & f = *sel.construct();
        f.clear(cuda::std::memory_order_relaxed);
        assert(f.test_and_set() == 0);
        f.clear(cuda::std::memory_order_relaxed);
        assert(f.test_and_set() == 0);
    }
    {
        Selector<cuda::std::atomic_flag, default_initializer> sel;
        cuda::std::atomic_flag & f = *sel.construct();
        f.clear(cuda::std::memory_order_release);
        assert(f.test_and_set() == 0);
        f.clear(cuda::std::memory_order_release);
        assert(f.test_and_set() == 0);
    }
    {
        Selector<cuda::std::atomic_flag, default_initializer> sel;
        cuda::std::atomic_flag & f = *sel.construct();
        f.clear(cuda::std::memory_order_seq_cst);
        assert(f.test_and_set() == 0);
        f.clear(cuda::std::memory_order_seq_cst);
        assert(f.test_and_set() == 0);
    }
    {
        Selector<volatile cuda::std::atomic_flag, default_initializer> sel;
        volatile cuda::std::atomic_flag & f = *sel.construct();
        f.clear();
        assert(f.test_and_set() == 0);
        f.clear();
        assert(f.test_and_set() == 0);
    }
    {
        Selector<volatile cuda::std::atomic_flag, default_initializer> sel;
        volatile cuda::std::atomic_flag & f = *sel.construct();
        f.clear(cuda::std::memory_order_relaxed);
        assert(f.test_and_set() == 0);
        f.clear(cuda::std::memory_order_relaxed);
        assert(f.test_and_set() == 0);
    }
    {
        Selector<volatile cuda::std::atomic_flag, default_initializer> sel;
        volatile cuda::std::atomic_flag & f = *sel.construct();
        f.clear(cuda::std::memory_order_release);
        assert(f.test_and_set() == 0);
        f.clear(cuda::std::memory_order_release);
        assert(f.test_and_set() == 0);
    }
    {
        Selector<volatile cuda::std::atomic_flag, default_initializer> sel;
        volatile cuda::std::atomic_flag & f = *sel.construct();
        f.clear(cuda::std::memory_order_seq_cst);
        assert(f.test_and_set() == 0);
        f.clear(cuda::std::memory_order_seq_cst);
        assert(f.test_and_set() == 0);
    }
}

int main(int, char**)
{
    NV_DISPATCH_TARGET(
    NV_IS_HOST,(
        test<local_memory_selector>();
    ),
    NV_PROVIDES_SM_70,(
        test<local_memory_selector>();
    ))

    NV_IF_TARGET(NV_IS_DEVICE,(
        test<shared_memory_selector>();
        test<global_memory_selector>();
    ))

    return 0;
}
