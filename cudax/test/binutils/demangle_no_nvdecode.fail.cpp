//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// This test uses assert(...) for checking results
#undef NDEBUG

#include <cuda/std/cassert>

#include <cuda/experimental/binutils.cuh>

#if _CCCL_HAS_INCLUDE(<nv_decode.h>)
#  error "This test requires <nv_decode.h> not to be findable in PATH."
#endif

namespace cudax = cuda::experimental;

bool test()
{
  constexpr auto real_mangled_name = "_ZN8clstmp01I5cls01E13clstmp01_mf01Ev";
  const auto demangled             = cudax::demangle(real_mangled_name);
  assert(demangled == "clstmp01<cls01>::clstmp01_mf01()");

  return true;
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (test();))
  return 0;
}
