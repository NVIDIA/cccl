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
//  - LayoutPolicy shall meet the layout mapping policy requirements ([mdspan.layout.policy.reqmts])
#include <cuda/mdspan>

#include "test_macros.h"

__host__ __device__ void not_layout_policy()
{
  // expected-error-re@*:* {{{{(static_assert|static assertion)}} failed {{.*}}mdspan: LayoutPolicy template parameter
  // is invalid. A common mistake is to pass a layout mapping instead of a layout policy}}
  cuda::
    shared_memory_mdspan<int, cuda::std::extents<int>, cuda::std::layout_left::template mapping<cuda::std::extents<int>>>
      m;
  unused(m);
}

int main(int, char**)
{
  return 0;
}
