//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc

// <mdspan>

// template<class ElementType, class Extents, class LayoutPolicy = layout_right, class AccessorPolicy =
// default_accessor> class mdspan;
//
// Mandates:
//  - Extents is a specialization of extents

#include <cuda/mdspan>

#include "test_macros.h"

__host__ __device__ void not_extents()
{
  // expected-error-re@*:* {{{{(static_assert|static assertion)}} failed {{.*}}mdspan: Extents template parameter must
  // be a specialization of extents.}}
  cuda::shared_memory_mdspan<int, int> m;
  unused(m);
}

int main(int, char**)
{
  return 0;
}
