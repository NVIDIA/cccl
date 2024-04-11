//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: pre-sm-70

// <cuda/std/semaphore>

#include <cuda/std/semaphore>

#include "test_macros.h"

int main(int, char**)
{
  static_assert(cuda::std::counting_semaphore<>::max() > 0, "");
  static_assert(cuda::std::counting_semaphore<1>::max() >= 1, "");
  static_assert(cuda::std::counting_semaphore<cuda::std::numeric_limits<int>::max()>::max() >= 1, "");
  static_assert(cuda::std::counting_semaphore<cuda::std::numeric_limits<unsigned>::max()>::max() >= 1, "");
  static_assert(cuda::std::counting_semaphore<cuda::std::numeric_limits<ptrdiff_t>::max()>::max() >= 1, "");
  static_assert(cuda::std::counting_semaphore<1>::max() == cuda::std::binary_semaphore::max(), "");

  static_assert(cuda::counting_semaphore<cuda::thread_scope_system>::max() > 0, "");
  static_assert(cuda::counting_semaphore<cuda::thread_scope_system, 1>::max() >= 1, "");
  static_assert(cuda::counting_semaphore<cuda::thread_scope_system, cuda::std::numeric_limits<int>::max()>::max() >= 1,
                "");
  static_assert(
    cuda::counting_semaphore<cuda::thread_scope_system, cuda::std::numeric_limits<unsigned>::max()>::max() >= 1, "");
  static_assert(
    cuda::counting_semaphore<cuda::thread_scope_system, cuda::std::numeric_limits<ptrdiff_t>::max()>::max() >= 1, "");
  static_assert(cuda::counting_semaphore<cuda::thread_scope_system, 1>::max()
                  == cuda::binary_semaphore<cuda::thread_scope_system>::max(),
                "");

  static_assert(cuda::counting_semaphore<cuda::thread_scope_device>::max() > 0, "");
  static_assert(cuda::counting_semaphore<cuda::thread_scope_device, 1>::max() >= 1, "");
  static_assert(cuda::counting_semaphore<cuda::thread_scope_device, cuda::std::numeric_limits<int>::max()>::max() >= 1,
                "");
  static_assert(
    cuda::counting_semaphore<cuda::thread_scope_device, cuda::std::numeric_limits<unsigned>::max()>::max() >= 1, "");
  static_assert(
    cuda::counting_semaphore<cuda::thread_scope_device, cuda::std::numeric_limits<ptrdiff_t>::max()>::max() >= 1, "");
  static_assert(cuda::counting_semaphore<cuda::thread_scope_device, 1>::max()
                  == cuda::binary_semaphore<cuda::thread_scope_device>::max(),
                "");

  static_assert(cuda::counting_semaphore<cuda::thread_scope_block>::max() > 0, "");
  static_assert(cuda::counting_semaphore<cuda::thread_scope_block, 1>::max() >= 1, "");
  static_assert(cuda::counting_semaphore<cuda::thread_scope_block, cuda::std::numeric_limits<int>::max()>::max() >= 1,
                "");
  static_assert(
    cuda::counting_semaphore<cuda::thread_scope_block, cuda::std::numeric_limits<unsigned>::max()>::max() >= 1, "");
  static_assert(
    cuda::counting_semaphore<cuda::thread_scope_block, cuda::std::numeric_limits<ptrdiff_t>::max()>::max() >= 1, "");
  static_assert(cuda::counting_semaphore<cuda::thread_scope_block, 1>::max()
                  == cuda::binary_semaphore<cuda::thread_scope_block>::max(),
                "");

  return 0;
}
