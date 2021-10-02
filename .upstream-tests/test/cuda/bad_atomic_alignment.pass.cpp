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

template <typename T>
__host__ __device__
constexpr bool unused(T &&) {return true;}

int main(int argc, char ** argv)
{
  // Test default aligned user type
  {
    struct key {
      int32_t a;
      int32_t b;
    };
    static_assert(alignof(key) == 4, "");
    cuda::atomic<key> k(key{});
    auto r = k.load();
    k.store(r);
    (void)k.exchange(r);
    unused(r);
  }
  // Test forcibly aligned user type
  {
    struct alignas(8) key {
      int32_t a;
      int32_t b;
    };
    static_assert(alignof(key) == 8, "");
    cuda::atomic<key> k(key{});
    auto r = k.load();
    k.store(r);
    (void)k.exchange(r);
    unused(r);
  }
  return 0;
}
