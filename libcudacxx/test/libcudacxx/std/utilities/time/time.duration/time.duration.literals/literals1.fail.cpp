// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11

// .fail. expects compilation to fail, but this would only fail at runtime with NVRTC

#include <cuda/std/cassert>
#include <cuda/std/chrono>

int main(int, char**)
{
  cuda::std::chrono::hours h = 4h; // should fail w/conversion operator not found

  return 0;
}
