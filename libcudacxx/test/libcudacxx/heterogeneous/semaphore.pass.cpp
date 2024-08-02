//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc, pre-sm-70

// uncomment for a really verbose output detailing what test steps are being launched
#define DEBUG_TESTERS

#include <cuda/std/cassert>
#include <cuda/std/semaphore>

#include "helpers.h"

template <int N>
struct release
{
  static constexpr size_t threadcount = N;

  template <typename Semaphore>
  __host__ __device__ static void perform(Semaphore& semaphore)
  {
    semaphore.release(1);
  }
};

template <int N>
struct acquire
{
  static constexpr size_t threadcount = N;

  template <typename Semaphore>
  __host__ __device__ static void perform(Semaphore& semaphore)
  {
    semaphore.acquire();
  }
};

using a3_r3_a3_r3 = performer_list<acquire<3>, release<3>, acquire<3>, release<3>>;

void kernel_invoker()
{
  validate_pinned<cuda::std::counting_semaphore<3>, a3_r3_a3_r3>(3);
  validate_pinned<cuda::counting_semaphore<cuda::thread_scope_system, 3>, a3_r3_a3_r3>(3);
}

int main(int arg, char** argv)
{
  NV_IF_TARGET(NV_IS_HOST, (kernel_invoker();))

  return 0;
}
