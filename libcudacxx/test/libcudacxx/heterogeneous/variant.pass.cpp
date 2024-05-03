//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++11
// UNSUPPORTED: msvc-19.16
// UNSUPPORTED: nvrtc

// uncomment for a really verbose output detailing what test steps are being launched
// #define DEBUG_TESTERS

#include <cuda/std/cassert>
#include <cuda/std/variant>

#include "helpers.h"

struct pod
{
  int val;

  __host__ __device__ friend bool operator==(pod lhs, pod rhs)
  {
    return lhs.val == rhs.val;
  }
};

using variant_t = cuda::std::variant<int, pod, double>;

template <typename T, int Val>
struct tester
{
  template <typename Variant>
  __host__ __device__ static void initialize(Variant&& v)
  {
    v = T{Val};
  }

  template <typename Variant>
  __host__ __device__ static void validate(Variant&& v)
  {
    assert(cuda::std::holds_alternative<T>(v));
    assert(cuda::std::get<T>(v) == T{Val});
  }
};

using testers =
  tester_list<tester<int, 10>, tester<int, 20>, tester<pod, 30>, tester<pod, 40>, tester<double, 50>, tester<double, 60>>;

void kernel_invoker()
{
  variant_t v;
  validate_pinned<variant_t, testers>(v);
}

int main(int arg, char** argv)
{
#ifndef __CUDA_ARCH__
  kernel_invoker();
#endif

  return 0;
}
