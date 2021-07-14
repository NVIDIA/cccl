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

// atomic_flag() = default;

#include <cuda/std/atomic>
#include <cuda/std/cassert>

#include "test_macros.h"

#if !defined(TEST_COMPILER_C1XX)
  #include "placement_new.h"
#endif
#include "cuda_space_selector.h"

template<template<typename, typename> class Selector>
__host__ __device__
void test()
{
    Selector<cuda::std::atomic_flag, default_initializer> sel;
    cuda::std::atomic_flag & f = *sel.construct();
    f.clear();
    assert(f.test_and_set() == 0);
    {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
        typedef cuda::std::atomic_flag A;
        TEST_ALIGNAS_TYPE(A) char storage[sizeof(A)] = {1};
        A& zero = *new (storage) A();
        assert(!zero.test_and_set());
        // cudafe crashes on trying to interpret the line below when compiling with Clang
        // TODO: file a compiler bug
#if !(defined(__clang__) && defined(__CUDACC__))
        zero.~A();
#endif
#endif
    }
}

int main(int, char**)
{
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
    test<local_memory_selector>();
#endif
#ifdef __CUDA_ARCH__
    test<shared_memory_selector>();
    test<global_memory_selector>();
#endif

  return 0;
}
