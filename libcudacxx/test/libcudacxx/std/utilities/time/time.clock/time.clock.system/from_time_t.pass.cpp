//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/chrono>

// system_clock

// static time_point from_time_t(time_t t);

#include <cuda/std/chrono>
#include <cuda/std/ctime>

int main(int, char**)
{
  typedef cuda::std::chrono::system_clock C;
  [[maybe_unused]] C::time_point t1 = C::from_time_t(C::to_time_t(C::now()));

  return 0;
}
