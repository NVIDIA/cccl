//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/chrono>

// duration

// template <class Rep2>
//   explicit duration(const Rep2& r);

// Rep2 shall be implicitly convertible to rep

// .fail. expects compilation to fail, but this would only fail at runtime with NVRTC

#include <cuda/std/chrono>

#include "../../rep.h"

int main(int, char**)
{
  cuda::std::chrono::duration<Rep> d(1);

  return 0;
}
