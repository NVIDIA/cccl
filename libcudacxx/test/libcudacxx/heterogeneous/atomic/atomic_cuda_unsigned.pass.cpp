//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc, pre-sm-60
// UNSUPPORTED: windows && pre-sm-70

#include <cuda/atomic>
#include <cuda/std/cassert>

#include "common.h"

void kernel_invoker()
{
  validate_pinned<cuda::atomic<unsigned char, cuda::thread_scope_system>, bitwise_atomic_testers>();
  validate_pinned<cuda::atomic<unsigned short, cuda::thread_scope_system>, bitwise_atomic_testers>();
  validate_pinned<cuda::atomic<unsigned int, cuda::thread_scope_system>, bitwise_atomic_testers>();
  validate_pinned<cuda::atomic<unsigned long, cuda::thread_scope_system>, bitwise_atomic_testers>();
  validate_pinned<cuda::atomic<unsigned long long, cuda::thread_scope_system>, bitwise_atomic_testers>();
}

int main(int arg, char** argv)
{
  NV_IF_TARGET(NV_IS_HOST, (kernel_invoker();))

  return 0;
}
