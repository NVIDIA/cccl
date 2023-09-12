// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads
// NVRTC_SKIP_KERNEL_RUN // do compile, but do not run under nvrtc

#include <cuda/std/cassert>
#include <nv/target>

// This is a test of the NVRTC_SKIP_KERNEL_RUN tag that indicates that a test
// should compiler under NVRTC, but should not be run.
int main(int, char**)
{
  NV_DISPATCH_TARGET(
        NV_IS_DEVICE, (
          // Ensure that code fails at runtime when run under NVRTC.
#ifdef _LIBCUDACXX_COMPILER_NVRTC
          assert(false);
#endif
        )
    );

  return 0;
}
