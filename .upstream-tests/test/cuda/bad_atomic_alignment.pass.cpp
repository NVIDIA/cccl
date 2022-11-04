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

template <typename T>
__host__ __device__
constexpr bool unused(T &&) {return true;}

template <template<typename, typename> typename Selector>
struct TestFn {
  __host__ __device__
  void operator()() const {
    {
        struct key {
          int32_t a;
          int32_t b;
        };
        typedef cuda::std::atomic<key> A;
        Selector<A, constructor_initializer> sel;
        A & t = *sel.construct();
        cuda::std::atomic_init(&t, key{1,2});
        auto r = t.load();
        t.store(r);
        (void)t.exchange(r);
    }
    {
        struct alignas(8) key {
          int32_t a;
          int32_t b;
        };
        typedef cuda::std::atomic<key> A;
        Selector<A, constructor_initializer> sel;
        A & t = *sel.construct();
        cuda::std::atomic_init(&t, key{1,2});
        auto r = t.load();
        t.store(r);
        (void)t.exchange(r);
    }
  }
};

int main(int, char**)
{
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
    TestFn<local_memory_selector>()();
#endif
#ifdef __CUDA_ARCH__
    TestFn<shared_memory_selector>()();
    TestFn<global_memory_selector>()();
#endif

  return 0;
}
