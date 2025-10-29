//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: pre-sm-70
// UNSUPPORTED: windows

// <cuda/atomic>

#include <cuda/atomic>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "cuda_space_selector.h"
#include "test_macros.h"

template <typename T>
__host__ __device__ constexpr T combine_literal(uint64_t lower, uint64_t upper)
{
  return T(lower) | (T(upper) << 64);
}

template <template <typename, typename> class Selector, cuda::thread_scope ThreadScope>
__host__ __device__ void test()
{
  {
    using T = __int128_t;
    typedef cuda::atomic_ref<T, ThreadScope> A;
    Selector<T, constructor_initializer> sel;
    T& t = *sel.construct();
    t    = T(0);
    A atom(t);
    auto test_v = combine_literal<T>(0x01234567DEADBEEF, 0x1337B33701234567);
    atom.store(test_v, cuda::std::memory_order_release);
    assert(atom.load() == test_v);
  }
  {
    using T = __uint128_t;
    typedef cuda::atomic_ref<T, ThreadScope> A;
    Selector<T, constructor_initializer> sel;
    T& t = *sel.construct();
    t    = T(0);
    A atom(t);
    auto test_v = combine_literal<T>(0x01234567DEADBEEF, 0x1337B33701234567);
    atom.store(test_v);
    assert(atom.load() == test_v);
  }
}

int main(int, char**)
{
#if __cccl_ptx_isa >= 840
  NV_DISPATCH_TARGET(
    NV_PROVIDES_SM_70,
    (test<local_memory_selector, cuda::thread_scope_thread>(); test<shared_memory_selector, cuda::thread_scope_block>();
     test<global_memory_selector, cuda::thread_scope_block>();
     test<global_memory_selector, cuda::thread_scope_device>();))
#endif
  return 0;
}
