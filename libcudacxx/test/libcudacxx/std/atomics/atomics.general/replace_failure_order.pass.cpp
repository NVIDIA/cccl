//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: libcpp-has-no-threads, pre-sm-60
// UNSUPPORTED: windows && pre-sm-70

// This test verifies behavior specified by [atomics.types.operations.req]/21:
//
//     When only one memory_order argument is supplied, the value of success is
//     order, and the value of failure is order except that a value of
//     memory_order_acq_rel shall be replaced by the value memory_order_acquire
//     and a value of memory_order_release shall be replaced by the value
//     memory_order_relaxed.
//
// Clang's atomic intrinsics do this for us, but GCC's do not. We don't actually
// have visibility to see what these memory orders are lowered to, but we can at
// least check that they are lowered at all (otherwise there is a compile
// failure with GCC).

#include <cuda/std/atomic>

#include "cuda_space_selector.h"
#include "test_macros.h"

template <template <typename, typename> class Selector>
__host__ __device__ void test()
{
  Selector<cuda::std::atomic<int>, default_initializer> sel;
  Selector<volatile cuda::std::atomic<int>, default_initializer> vsel;

  cuda::std::atomic<int>& i          = *sel.construct();
  volatile cuda::std::atomic<int>& v = *vsel.construct();
  int exp                            = 0;

  (void) i.compare_exchange_weak(exp, 0, cuda::std::memory_order_acq_rel);
  (void) i.compare_exchange_weak(exp, 0, cuda::std::memory_order_release);
  i.compare_exchange_strong(exp, 0, cuda::std::memory_order_acq_rel);
  i.compare_exchange_strong(exp, 0, cuda::std::memory_order_release);

  (void) v.compare_exchange_weak(exp, 0, cuda::std::memory_order_acq_rel);
  (void) v.compare_exchange_weak(exp, 0, cuda::std::memory_order_release);
  v.compare_exchange_strong(exp, 0, cuda::std::memory_order_acq_rel);
  v.compare_exchange_strong(exp, 0, cuda::std::memory_order_release);
}

int main(int, char**)
{
  NV_DISPATCH_TARGET(NV_IS_HOST, (test<local_memory_selector>();), NV_PROVIDES_SM_70, (test<local_memory_selector>();))

  NV_IF_TARGET(NV_IS_DEVICE, (test<shared_memory_selector>(); test<global_memory_selector>();))

  return 0;
}
