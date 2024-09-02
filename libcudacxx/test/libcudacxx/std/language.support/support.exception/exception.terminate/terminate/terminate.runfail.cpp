//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// test terminate
// UNSUPPORTED: no_execute
// UNSUPPORTED: nvrtc

#include <cuda/std/__exception/terminate.h>
#include <cuda/std/cassert>

#include "test_macros.h"

#if defined(TEST_COMPILER_MSVC)
#  pragma warning(disable : 4702) // unreachable code
#endif // TEST_COMPILER_MSVC

#if 0 // Wait until terminate handler is available
__host__ __device__ void f1()
{
  NV_IF_ELSE_TARGET(NV_IS_HOST, (::std::exit(0);),(__trap();))
}
#endif //

int main(int, char**)
{
  // cuda::std::set_terminate(f1);
  cuda::std::terminate();
  assert(false);
  return 0;
}
