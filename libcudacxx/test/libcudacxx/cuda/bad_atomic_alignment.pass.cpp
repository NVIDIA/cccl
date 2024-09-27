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

// <cuda/atomic>

// cuda::atomic<key>

// Original test issue:
// https://github.com/NVIDIA/libcudacxx/issues/160

#include <cuda/atomic>

#include "cuda_space_selector.h"
#include "test_macros.h"

template <template <typename, typename> class Selector>
struct TestFn
{
  __host__ __device__ void operator()() const
  {
    {
      struct key
      {
        int32_t a;
        int32_t b;
      };
      typedef cuda::std::atomic<key> A;
      Selector<A, constructor_initializer> sel;
      A& t = *sel.construct();
      cuda::std::atomic_init(&t, key{1, 2});
      auto r = t.load();
      auto d = key{5, 5};
      t.store(r);
      (void) t.exchange(r);
      (void) t.compare_exchange_weak(r, d, cuda::memory_order_seq_cst, cuda::memory_order_seq_cst);
      (void) t.compare_exchange_strong(d, r, cuda::memory_order_seq_cst, cuda::memory_order_seq_cst);
    }
    {
      struct alignas(8) key
      {
        int32_t a;
        int32_t b;
      };
      typedef cuda::std::atomic<key> A;
      Selector<A, constructor_initializer> sel;
      A& t = *sel.construct();
      cuda::std::atomic_init(&t, key{1, 2});
      auto r = t.load();
      auto d = key{5, 5};
      t.store(r);
      (void) t.exchange(r);
      (void) t.compare_exchange_weak(r, d, cuda::memory_order_seq_cst, cuda::memory_order_seq_cst);
      (void) t.compare_exchange_strong(d, r, cuda::memory_order_seq_cst, cuda::memory_order_seq_cst);
    }
  }
};

int main(int, char**)
{
  NV_DISPATCH_TARGET(NV_IS_HOST, TestFn<local_memory_selector>()();
                     , NV_PROVIDES_SM_70, TestFn<local_memory_selector>()();)

  NV_IF_TARGET(NV_IS_DEVICE, (TestFn<shared_memory_selector>()(); TestFn<global_memory_selector>()();))

  return 0;
}
