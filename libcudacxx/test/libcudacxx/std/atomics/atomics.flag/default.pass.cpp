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

#if !defined(TEST_COMPILER_MSVC)
#  include "placement_new.h"
#endif
#include "cuda_space_selector.h"

template <template <typename, typename> class Selector>
__host__ __device__ void test()
{
  Selector<cuda::std::atomic_flag, default_initializer> sel;
  cuda::std::atomic_flag& f = *sel.construct();
  f.clear();
  assert(f.test_and_set() == 0);

  NV_DISPATCH_TARGET(
    NV_IS_HOST,
    (typedef cuda::std::atomic_flag A; TEST_ALIGNAS_TYPE(A) char storage[sizeof(A)] = {1}; A& zero = *new (storage) A();
     assert(!zero.test_and_set());
     zero.~A();),
    NV_PROVIDES_SM_70,
    (typedef cuda::std::atomic_flag A; TEST_ALIGNAS_TYPE(A) char storage[sizeof(A)] = {1}; A& zero = *new (storage) A();
     assert(!zero.test_and_set());
     zero.~A();))
}

int main(int, char**)
{
  NV_DISPATCH_TARGET(NV_IS_HOST, (test<local_memory_selector>();), NV_PROVIDES_SM_70, (test<local_memory_selector>();))

  NV_IF_TARGET(NV_IS_DEVICE, (test<shared_memory_selector>(); test<global_memory_selector>();))

  return 0;
}
