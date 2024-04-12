//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cuda/std/cstddef>
#include <cuda/std/type_traits>

// max_align_t is a trivial standard-layout type whose alignment requirement
//   is at least as great as that of every scalar type
#include "test_macros.h"

#ifndef TEST_COMPILER_NVRTC
#  include <stdio.h>
#endif // TEST_COMPILER_NVRTC

TEST_NV_DIAG_SUPPRESS(cuda_demote_unsupported_floating_point)

int main(int, char**)
{
#if TEST_STD_VER > 2017
  //  P0767
  static_assert(cuda::std::is_trivial<cuda::std::max_align_t>::value,
                "cuda::std::is_trivial<cuda::std::max_align_t>::value");
  static_assert(cuda::std::is_standard_layout<cuda::std::max_align_t>::value,
                "cuda::std::is_standard_layout<cuda::std::max_align_t>::value");
#else
  static_assert(cuda::std::is_pod<cuda::std::max_align_t>::value, "cuda::std::is_pod<cuda::std::max_align_t>::value");
#endif
  static_assert((cuda::std::alignment_of<cuda::std::max_align_t>::value >= cuda::std::alignment_of<long long>::value),
                "cuda::std::alignment_of<cuda::std::max_align_t>::value >= "
                "cuda::std::alignment_of<long long>::value");
  static_assert(cuda::std::alignment_of<cuda::std::max_align_t>::value >= cuda::std::alignment_of<long double>::value,
                "cuda::std::alignment_of<cuda::std::max_align_t>::value >= "
                "cuda::std::alignment_of<long double>::value");
  static_assert(cuda::std::alignment_of<cuda::std::max_align_t>::value >= cuda::std::alignment_of<void*>::value,
                "cuda::std::alignment_of<cuda::std::max_align_t>::value >= "
                "cuda::std::alignment_of<void*>::value");

  return 0;
}
