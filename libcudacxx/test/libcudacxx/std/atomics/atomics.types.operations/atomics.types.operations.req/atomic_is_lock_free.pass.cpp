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

// template <class T>
//     bool
//     atomic_is_lock_free(const volatile atomic<T>* obj);
//
// template <class T>
//     bool
//     atomic_is_lock_free(const atomic<T>* obj);

#include <cuda/std/atomic>
#include <cuda/std/cassert>

#include "atomic_helpers.h"
#include "cuda_space_selector.h"
#include "test_macros.h"

template <class T, template <typename, typename> class, cuda::thread_scope>
struct TestFn
{
  __host__ __device__ void operator()() const
  {
    typedef cuda::std::atomic<T> A;
    A t{};
    bool b1 = cuda::std::atomic_is_lock_free(static_cast<const A*>(&t));
    volatile A vt{};
    bool b2 = cuda::std::atomic_is_lock_free(static_cast<const volatile A*>(&vt));
    assert(b1 == b2);
  }
};

struct A
{
  char _[4];
};

int main(int, char**)
{
  TestFn<A, local_memory_selector, cuda::thread_scope_system>()();
  TestEachAtomicType<TestFn, local_memory_selector>()();

  return 0;
}
