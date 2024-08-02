//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc, pre-sm-70

// uncomment for a really verbose output detailing what test steps are being launched
// #define DEBUG_TESTERS

#include <cuda/std/cassert>
#include <cuda/std/latch>

#include "helpers.h"

template <int N>
struct count_down
{
  using async                         = cuda::std::true_type;
  static constexpr size_t threadcount = N;

  template <typename Latch>
  __host__ __device__ static void perform(Latch& latch)
  {
    latch.count_down(1);
  }
};

template <int N>
struct arrive_and_wait
{
  using async                         = cuda::std::true_type;
  static constexpr size_t threadcount = N;

  template <typename Latch>
  __host__ __device__ static void perform(Latch& latch)
  {
    latch.arrive_and_wait(1);
  }
};

// This one is named `latch_wait` because otherwise you get this on older systems:
// .../latch.pass.cpp(44): error: invalid redeclaration of type name "wait"
// /usr/include/bits/waitstatus.h(66): here
// Isn't software great?
struct latch_wait
{
  using async = cuda::std::true_type;

  template <typename Latch>
  __host__ __device__ static void perform(Latch& latch)
  {
    latch.wait();
  }
};

template <int Expected>
struct reset
{
  template <typename Latch>
  __host__ __device__ static void perform(Latch& latch)
  {
    new (&latch) Latch(Expected);
  }
};

using r0_w = performer_list<reset<0>, latch_wait>;

using r5_cd5_w_w = performer_list<reset<5>, count_down<5>, latch_wait, latch_wait>;

using r5_aw5_w_w = performer_list<reset<5>, arrive_and_wait<5>, latch_wait, latch_wait>;

void kernel_invoker()
{
  validate_pinned<cuda::std::latch, r0_w>(0);
  validate_pinned<cuda::latch<cuda::thread_scope_system>, r0_w>(0);

  validate_pinned<cuda::std::latch, r5_cd5_w_w>(0);
  validate_pinned<cuda::latch<cuda::thread_scope_system>, r5_cd5_w_w>(0);

  validate_pinned<cuda::std::latch, r5_aw5_w_w>(0);
  validate_pinned<cuda::latch<cuda::thread_scope_system>, r5_aw5_w_w>(0);
}

int main(int arg, char** argv)
{
  NV_IF_TARGET(NV_IS_HOST, (kernel_invoker();))

  return 0;
}
