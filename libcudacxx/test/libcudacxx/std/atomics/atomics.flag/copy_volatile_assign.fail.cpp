//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: pre-sm-60
// UNSUPPORTED: windows && pre-sm-70

// .fail. expects compilation to fail, but this would only fail at runtime with NVRTC

// <cuda/std/atomic>

// struct atomic_flag

// atomic_flag& operator=(const atomic_flag&) = delete;

#include <cuda/std/atomic>
#include <cuda/std/cassert>

int main(int, char**)
{
  cuda::std::atomic_flag f0;
  volatile cuda::std::atomic_flag f;
  f = f0;

  return 0;
}
