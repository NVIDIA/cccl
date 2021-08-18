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

#include <cuda/std/atomic>
#include <cuda/std/type_traits>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "atomic_helpers.h"
#include "cuda_space_selector.h"

template <class T, template<typename, typename> typename Selector, cuda::thread_scope>
struct TestFn {
  __host__ __device__
  void operator()() const {
    // Test greater
    {
        typedef cuda::atomic<T> A;
        Selector<A, constructor_initializer> sel;
        A & t = *sel.construct();
        t = T(1);
        assert(t.fetch_max(2) == T(1));
        assert(t.load() == T(2));
    }
    {
        typedef cuda::atomic<T> A;
        Selector<volatile A, constructor_initializer> sel;
        volatile A & t = *sel.construct();
        t = T(1);
        assert(t.fetch_max(2) == T(1));
        assert(t.load() == T(2));
    }
    // Test not greater
    {
        typedef cuda::atomic<T> A;
        Selector<A, constructor_initializer> sel;
        A & t = *sel.construct();
        t = T(3);
        assert(t.fetch_max(2) == T(3));
        assert(t.load() == T(3));
    }
    {
        typedef cuda::atomic<T> A;
        Selector<volatile A, constructor_initializer> sel;
        volatile A & t = *sel.construct();
        t = T(3);
        assert(t.fetch_max(2) == T(3));
        assert(t.load() == T(3));
    }
  }
};

int main(int, char**)
{
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
    TestEachIntegralType<TestFn, local_memory_selector>()();
#endif
#ifdef __CUDA_ARCH__
    TestEachIntegralType<TestFn, shared_memory_selector>()();
    TestEachIntegralType<TestFn, global_memory_selector>()();
#endif

  return 0;
}
