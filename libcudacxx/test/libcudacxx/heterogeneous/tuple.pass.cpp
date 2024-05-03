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
#include <cuda/std/tuple>

#include "helpers.h"

struct pod
{
  char val[10];
};

using tuple_t = cuda::std::tuple<int, pod, unsigned long long>;

template <int N>
struct Write
{
  using async = cuda::std::false_type;

  template <typename Tuple>
  __host__ __device__ static void perform(Tuple& t)
  {
    cuda::std::get<0>(t)        = N;
    cuda::std::get<1>(t).val[0] = N;
    cuda::std::get<2>(t)        = N;
  }
};

template <int N>
struct Read
{
  using async = cuda::std::false_type;

  template <typename Tuple>
  __host__ __device__ static void perform(Tuple& t)
  {
    assert(cuda::std::get<0>(t) == N);
    assert(cuda::std::get<1>(t).val[0] == N);
    assert(cuda::std::get<2>(t) == N);
  }
};

using w_r_w_r = performer_list<Write<10>, Read<10>, Write<30>, Read<30>>;

void kernel_invoker()
{
  tuple_t t(0, {0}, 0);
  validate_pinned<tuple_t, w_r_w_r>(t);
}

int main(int arg, char** argv)
{
  NV_IF_TARGET(NV_IS_HOST, (kernel_invoker();))

  return 0;
}
