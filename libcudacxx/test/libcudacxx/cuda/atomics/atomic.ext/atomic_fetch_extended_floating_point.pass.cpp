//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: libcpp-has-no-threads, pre-sm-60
// UNSUPPORTED: windows && pre-sm-70
// UNSUPPORTED: force-tile
// error: asm statement is unsupported in tile code

// <cuda/atomic>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cuda/atomic>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "atomic_helpers.h"
#include "cuda_space_selector.h"
#include "test_macros.h"

template <class T>
TEST_HOST_DEVICE_FUNC bool equal(T lhs, T rhs)
{
  return static_cast<float>(lhs) == static_cast<float>(rhs);
}

template <class T, template <typename, typename> class Selector, cuda::thread_scope ThreadScope>
struct TestFn
{
  TEST_HOST_DEVICE_FUNC void operator()() const
  {
    // Fetch min
    {
      using A = cuda::atomic<T, ThreadScope>;
      Selector<A, constructor_initializer> sel;
      A& t = *sel.construct();
      t    = T(-1);
      assert(equal(t.fetch_min(T(-5)), T(-1)));
      assert(equal(t.load(), T(-5)));
    }
    {
      using A = cuda::atomic<T, ThreadScope>;
      Selector<volatile A, constructor_initializer> sel;
      volatile A& t = *sel.construct();
      t             = T(-1);
      assert(equal(t.fetch_min(T(-5)), T(-1)));
      assert(equal(t.load(), T(-5)));
    }
    // Test not lesser
    {
      using A = cuda::atomic<T, ThreadScope>;
      Selector<A, constructor_initializer> sel;
      A& t = *sel.construct();
      t    = T(-1);
      assert(equal(t.fetch_min(4), T(-1)));
      assert(equal(t.load(), T(-1)));
    }
    {
      using A = cuda::atomic<T, ThreadScope>;
      Selector<volatile A, constructor_initializer> sel;
      volatile A& t = *sel.construct();
      t             = T(-1);
      assert(equal(t.fetch_min(4), T(-1)));
      assert(equal(t.load(), T(-1)));
    }
    // Fetch max
    {
      using A = cuda::atomic<T, ThreadScope>;
      Selector<A, constructor_initializer> sel;
      A& t = *sel.construct();
      t    = T(1);
      assert(equal(t.fetch_max(2), T(1)));
      assert(equal(t.load(), T(2)));
    }
    {
      using A = cuda::atomic<T, ThreadScope>;
      Selector<volatile A, constructor_initializer> sel;
      volatile A& t = *sel.construct();
      t             = T(1);
      assert(equal(t.fetch_max(2), T(1)));
      assert(equal(t.load(), T(2)));
    }
    // Test not greater
    {
      using A = cuda::atomic<T, ThreadScope>;
      Selector<A, constructor_initializer> sel;
      A& t = *sel.construct();
      t    = T(3);
      assert(equal(t.fetch_max(2), T(3)));
      assert(equal(t.load(), T(3)));
    }
    {
      using A = cuda::atomic<T, ThreadScope>;
      Selector<volatile A, constructor_initializer> sel;
      volatile A& t = *sel.construct();
      t             = T(3);
      assert(equal(t.fetch_max(2), T(3)));
      assert(equal(t.load(), T(3)));
    }
  }
};

int main(int, char**)
{
#if _CCCL_HAS_NVFP16()
  NV_DISPATCH_TARGET(NV_IS_HOST,
                     (TestFn<__half, local_memory_selector, cuda::thread_scope::thread_scope_thread>()();),
                     NV_PROVIDES_SM_70,
                     (TestFn<__half, local_memory_selector, cuda::thread_scope::thread_scope_thread>()();))

  NV_IF_TARGET(NV_IS_DEVICE,
               (TestFn<__half, shared_memory_selector, cuda::thread_scope::thread_scope_thread>()();
                TestFn<__half, global_memory_selector, cuda::thread_scope::thread_scope_thread>()();))
#endif // _CCCL_HAS_NVFP16()

#if _CCCL_HAS_NVBF16()
  NV_DISPATCH_TARGET(NV_IS_HOST,
                     (TestFn<__nv_bfloat16, local_memory_selector, cuda::thread_scope::thread_scope_thread>()();),
                     NV_PROVIDES_SM_70,
                     (TestFn<__nv_bfloat16, local_memory_selector, cuda::thread_scope::thread_scope_thread>()();))

  NV_IF_TARGET(NV_IS_DEVICE,
               (TestFn<__nv_bfloat16, shared_memory_selector, cuda::thread_scope::thread_scope_thread>()();
                TestFn<__nv_bfloat16, global_memory_selector, cuda::thread_scope::thread_scope_thread>()();))
#endif // _CCCL_HAS_NVBF16()

  return 0;
}
