//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc

// uncomment for a really verbose output detailing what test steps are being launched
// #define DEBUG_TESTERS

#include <cuda/std/cassert>
#include <cuda/std/utility>

#include "helpers.h"

struct pod
{
  char val[10];
};

using pair_t = cuda::std::pair<int, pod>;

template <int N>
struct Write
{
  using async = cuda::std::false_type;

  template <typename Pair>
  __host__ __device__ static void perform(Pair& p)
  {
    cuda::std::get<0>(p)        = N;
    cuda::std::get<1>(p).val[0] = N;
  }
};

template <int N>
struct Read
{
  using async = cuda::std::false_type;

  template <typename Pair>
  __host__ __device__ static void perform(Pair& p)
  {
    assert(cuda::std::get<0>(p) == N);
    assert(cuda::std::get<1>(p).val[0] == N);
  }
};

using w_r_w_r = performer_list<Write<10>, Read<10>, Write<30>, Read<30>>;

void kernel_invoker()
{
  pair_t p(0, {0});
  validate_pinned<pair_t, w_r_w_r>(p);
}

int main(int arg, char** argv)
{
  NV_IF_TARGET(NV_IS_HOST, (kernel_invoker();))

  return 0;
}
