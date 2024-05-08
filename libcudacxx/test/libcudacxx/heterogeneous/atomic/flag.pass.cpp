//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc, pre-sm-60
// UNSUPPORTED: windows && pre-sm-70

#include <cuda/std/atomic>
#include <cuda/std/cassert>

#include "../helpers.h"

struct clear
{
  template <typename AF>
  __host__ __device__ static void initialize(AF& af)
  {
    af.clear();
  }
};

struct clear_tester : clear
{
  template <typename AF>
  __host__ __device__ static void validate(AF& af)
  {
    assert(af.test_and_set() == false);
  }
};

template <bool Previous>
struct test_and_set_tester
{
  template <typename AF>
  __host__ __device__ static void initialize(AF& af)
  {
    assert(af.test_and_set() == Previous);
  }

  template <typename AF>
  __host__ __device__ static void validate(AF& af)
  {
    assert(af.test_and_set() == true);
  }
};

using atomic_flag_testers = tester_list<clear_tester, clear, test_and_set_tester<false>, test_and_set_tester<true>>;

void kernel_invoker()
{
  validate_pinned<cuda::std::atomic_flag, atomic_flag_testers>();
}

int main(int argc, char** argv)
{
  NV_IF_TARGET(NV_IS_HOST, (kernel_invoker();))

  return 0;
}
