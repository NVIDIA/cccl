//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: enable-tile
// error: asm statement is unsupported in tile code

// UNSUPPORTED: libcpp-has-no-threads, pre-sm-60
// UNSUPPORTED: windows && pre-sm-70
//  ... test crashes clang

// <cuda/std/atomic>

// template <class Integral>
//     Integral
//     atomic_ref<Integral>::fetch_min(Integral op,
//                                     memory_order m = memory_order_seq_cst) const;

#include <cuda/std/atomic>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "atomic_helpers.h"
#include "cuda_space_selector.h"
#include "test_macros.h"

template <class T, template <typename, typename> class Selector, cuda::thread_scope>
struct TestFn
{
  TEST_FUNC void operator()() const
  {
    using A = cuda::std::atomic_ref<T>;
    // op less than the stored value: replaced, old value returned
    {
      Selector<T, constructor_initializer> sel;
      A a(*sel.construct(T(3)));
      assert(a.fetch_min(T(1)) == T(3));
      assert(a.load() == T(1));
    }
    // op not less than the stored value: unchanged, old value returned
    {
      Selector<T, constructor_initializer> sel;
      A a(*sel.construct(T(1)));
      assert(a.fetch_min(T(3)) == T(1));
      assert(a.load() == T(1));
    }
    // explicit memory order
    {
      Selector<T, constructor_initializer> sel;
      A a(*sel.construct(T(3)));
      assert(a.fetch_min(T(1), cuda::std::memory_order_relaxed) == T(3));
      assert(a.load() == T(1));
    }
  }
};

int main(int, char**)
{
  NV_DISPATCH_TARGET(NV_IS_HOST,
                     (TestEachIntegralRefType<TestFn, local_memory_selector>()();),
                     NV_PROVIDES_SM_70,
                     (TestEachIntegralRefType<TestFn, local_memory_selector>()();))

  NV_IF_TARGET(NV_IS_DEVICE,
               (TestEachIntegralRefType<TestFn, shared_memory_selector>()();
                TestEachIntegralRefType<TestFn, global_memory_selector>()();))

  return 0;
}
