//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc

#include "host_device_comparison.h"

#include <cuda/std/cmath>

void test() {
  compare_host_device<__nv_bfloat16>([] __host__ __device__(cuda::std::size_t i) {
    auto raw = __nv_bfloat16_raw();
    raw.x = i;
    return cuda::std::sqrt(__nv_bfloat16(raw));
  });
}

int main(int argc, char** argv) {
  NV_IF_TARGET(NV_IS_HOST, { test(); })

  return 0;
}
